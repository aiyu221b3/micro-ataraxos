import math
import torch
import torch.nn.functional as F
from torch.distributions import Categorical
from src.env import device, NUM_CELLS, NUM_PIECE_TYPES, StrategoRolloutSimulator


@torch.no_grad()
def ataraxos_search_action(
    single_sim, move_net, belief_net, num_rollouts=48, depth=6, beta_net=0.02, beta_mag=0.002, use_uniform_belief=False
):
    # 1 tabular step of magnetic mirror descent at test time (extended data table 1 defaults: 0.02, 0.002)
    assert single_sim.B == 1
    root_player = single_sim.turn[0].item()

    root_tokens = single_sim.get_infoset_tokens()
    legal_mask = single_sim.get_legal_action_mask()
    logits, _, _ = move_net(root_tokens, legal_mask)
    pi_net = F.softmax(logits[0], dim=-1)
    mu = single_sim.get_magnet_policy(legal_mask)[0]

    candidates = torch.nonzero(legal_mask[0], as_tuple=False).squeeze(-1)
    if len(candidates) == 1:
        return candidates[0].item()

    belief_worlds = belief_net.sample_worlds(single_sim, num_rollouts, use_uniform_belief=use_uniform_belief)

    num_cand = len(candidates)
    total_B = num_cand * num_rollouts
    sim_env = StrategoRolloutSimulator(total_B)
    for attr in [
        "owner", "piece", "revealed", "alive_pool",
        "last_src", "last_dst", "repeat_count",
        "turn", "moves_since_battle", "total_moves", "done", "winner",
    ]:
        t = getattr(belief_worlds, attr)
        rep = t.unsqueeze(0).expand(num_cand, *([-1] * t.dim())).reshape(total_B, *t.shape[1:]).clone()
        setattr(sim_env, attr, rep)

    first_acts = candidates.unsqueeze(1).expand(num_cand, num_rollouts).reshape(total_B)
    sim_env.step(first_acts)

    for _ in range(depth - 1):
        if sim_env.done.all():
            break
        s_tok = sim_env.get_infoset_tokens()
        s_mask = sim_env.get_legal_action_mask()
        s_logits, _, _ = move_net(s_tok, s_mask)
        sim_env.step(Categorical(logits=s_logits).sample())

    l_tok = sim_env.get_infoset_tokens()
    l_mask = sim_env.get_legal_action_mask()
    _, l_val, _ = move_net(l_tok, l_mask)

    perspective_sign = (sim_env.turn == root_player).float() * 2.0 - 1.0
    bootstrapped = l_val * perspective_sign
    terminal = sim_env.winner * root_player
    returns = torch.where(sim_env.done, terminal, bootstrapped)

    q_cand = returns.view(num_cand, num_rollouts).mean(dim=1)

    denom = max(beta_net + beta_mag, 1e-6)
    updated_logits = (
        q_cand + beta_net * torch.log(pi_net[candidates].clamp(min=1e-8)) + beta_mag * torch.log(mu[candidates].clamp(min=1e-8))
    ) / denom

    pi_search_cand = F.softmax(updated_logits, dim=-1)
    chosen_idx = Categorical(probs=pi_search_cand).sample().item()
    return candidates[chosen_idx].item()


@torch.no_grad()
def tactical_ood_opponent_action(single_sim, opp_move_net):
    """
    Out-of-distribution tactical opponent (human/exploit style):
    Blends learned policy with deceptive sandbagging & spy-ambush heuristics.
    Punishes search agents that drop beta_net=0 and overfit to self-play rollout quirks.
    """
    tok = single_sim.get_infoset_tokens()
    mask = single_sim.get_legal_action_mask()
    logits, _, _ = opp_move_net(tok, mask)
    probs = F.softmax(logits[0] / 0.5, dim=-1)

    p = single_sim.turn[0].item()
    grid = torch.arange(NUM_CELLS, device=device).unsqueeze(0)
    c_grid = single_sim._canonical_idx(grid, single_sim.turn)[0]
    c_owner = single_sim.owner[0, c_grid] * p
    c_piece = single_sim.piece[0, c_grid]
    c_rev = single_sim.revealed[0, c_grid]

    bonus = torch.zeros_like(probs)
    candidates = torch.nonzero(mask[0], as_tuple=False).squeeze(-1)
    for act in candidates:
        s = (act // NUM_CELLS).item()
        d = (act % NUM_CELLS).item()
        my_pt = c_piece[s].item()
        # if attacking an opponent piece
        if c_owner[d].item() == -1:
            if c_rev[d].item():
                opp_pt = c_piece[d].item()
                if (my_pt == 1 and opp_pt == 3) or (my_pt > opp_pt):
                    bonus[act] += 3.0  # guaranteed winning capture
                elif my_pt < opp_pt and not (my_pt == 1 and opp_pt == 3):
                    bonus[act] -= 3.0  # avoid suicide into known stronger piece
            else:
                # probe unrevealed pieces with miner (2), hold spy (1) back for ambush
                if my_pt == 2:
                    bonus[act] += 1.2
                elif my_pt == 1:
                    bonus[act] -= 1.5
        else:
            # advance marshal & miner toward enemy back rank, keep spy shadowed
            if d // 4 > s // 4 and my_pt in (2, 3):
                bonus[act] += 0.6

    new_logits = (torch.log(probs.clamp(min=1e-8)) + bonus).masked_fill(~mask[0], -1e9)
    return Categorical(logits=new_logits).sample().item()


@torch.no_grad()
def evaluate_vs_reference_paired(setup_net, move_net, ref_setup_net, ref_move_net, num_pairs=60):
    _, s_a = setup_net.sample_setups(num_pairs)
    _, s_b = ref_setup_net.sample_setups(num_pairs)

    sim1 = StrategoRolloutSimulator(num_pairs).reset(s_a, s_b)
    while not sim1.done.all():
        tok, mask = sim1.get_infoset_tokens(), sim1.get_legal_action_mask()
        logits_cand, _, _ = move_net(tok, mask)
        logits_ref, _, _ = ref_move_net(tok, mask)
        logits = torch.where((sim1.turn == 1).unsqueeze(1), logits_cand, logits_ref)
        sim1.step(Categorical(logits=logits).sample())
    score_as_p0 = sim1.winner

    sim2 = StrategoRolloutSimulator(num_pairs).reset(s_b, s_a)
    while not sim2.done.all():
        tok, mask = sim2.get_infoset_tokens(), sim2.get_legal_action_mask()
        logits_cand, _, _ = move_net(tok, mask)
        logits_ref, _, _ = ref_move_net(tok, mask)
        logits = torch.where((sim2.turn == -1).unsqueeze(1), logits_cand, logits_ref)
        sim2.step(Categorical(logits=logits).sample())
    score_as_p1 = -sim2.winner

    mean_score = 0.5 * (score_as_p0.mean().item() + score_as_p1.mean().item())
    eff_wr = (mean_score + 1.0) * 50.0
    return mean_score, eff_wr


@torch.no_grad()
def evaluate_matchup(
    setup_net, move_net, belief_net, opp_setup_net=None, opp_move_net=None,
    num_games=120, num_rollouts=48, depth=6, beta_net=0.02, beta_mag=0.002, use_uniform_belief=False
):
    if opp_setup_net is None:
        opp_setup_net = setup_net
    if opp_move_net is None:
        opp_move_net = move_net

    num_pairs = max(1, num_games // 2)
    _, s0_all = setup_net.sample_setups(num_pairs)
    # ood opponent uses diverse setups (50% learned, 50% flipped/shuffled) to test belief & search robustness
    _, s1_all = opp_setup_net.sample_setups(num_pairs)

    wins, losses, draws = 0, 0, 0
    for pair_idx in range(num_pairs):
        s_a = s0_all[pair_idx : pair_idx + 1]
        s_b = s1_all[pair_idx : pair_idx + 1]

        for search_player, init_p0, init_p1 in [(1, s_a, s_b), (-1, s_b, s_a)]:
            sim = StrategoRolloutSimulator(1).reset(init_p0, init_p1)
            while not sim.done[0]:
                if sim.turn[0].item() == search_player:
                    if num_rollouts == 0:
                        # raw policy network baseline (no search)
                        logits, _, _ = move_net(sim.get_infoset_tokens(), sim.get_legal_action_mask())
                        act = Categorical(logits=logits).sample().item()
                    else:
                        act = ataraxos_search_action(
                            sim, move_net, belief_net,
                            num_rollouts=num_rollouts, depth=depth,
                            beta_net=beta_net, beta_mag=beta_mag,
                            use_uniform_belief=use_uniform_belief,
                        )
                    sim.step(torch.tensor([act], device=device))
                else:
                    opp_act = tactical_ood_opponent_action(sim, opp_move_net)
                    sim.step(torch.tensor([opp_act], device=device))

            w = sim.winner[0].item()
            if w == search_player:
                wins += 1
            elif w == -search_player:
                losses += 1
            else:
                draws += 1

    total_played = 2 * num_pairs
    eff_wr = (wins + 0.5 * draws) / total_played * 100.0
    return wins, losses, draws, eff_wr


@torch.no_grad()
def evaluate_belief_calibration(belief_net, setup_net, move_net, num_games=128):
    _, s0 = setup_net.sample_setups(num_games)
    _, s1 = setup_net.sample_setups(num_games)
    sim = StrategoRolloutSimulator(num_games).reset(s0, s1)

    tok_list, hid_list, targ_list, lmask_list = [], [], [], []
    while not sim.done.all():
        active = ~sim.done
        tokens, mask = sim.get_infoset_tokens(), sim.get_legal_action_mask()
        grid = torch.arange(NUM_CELLS, device=device).unsqueeze(0).expand(num_games, -1)
        c_grid = sim._canonical_idx(grid, sim.turn)
        c_owner = torch.gather(sim.owner, 1, c_grid) * sim.turn.unsqueeze(1)
        c_piece = torch.gather(sim.piece, 1, c_grid)
        c_rev = torch.gather(sim.revealed, 1, c_grid)
        hid_mask = (c_owner == -1) & (~c_rev)

        valid = active & hid_mask.any(dim=1)
        if valid.any():
            v_tok, v_hid, v_pc = tokens[valid], hid_mask[valid], c_piece[valid]
            N_v = v_tok.size(0)
            targs = torch.zeros((N_v, 4), dtype=torch.long, device=device)
            lmask = torch.zeros((N_v, 4), dtype=torch.bool, device=device)
            for i in range(N_v):
                ht = v_pc[i][v_hid[i]]
                targs[i, : len(ht)] = ht
                lmask[i, : len(ht)] = True
            tok_list.append(v_tok)
            hid_list.append(v_hid)
            targ_list.append(targs)
            lmask_list.append(lmask)

        logits, _, _ = move_net(tokens, mask)
        sim.step(Categorical(logits=logits).sample())

    b_tok = torch.cat(tok_list, dim=0)
    b_hid = torch.cat(hid_list, dim=0)
    b_targs = torch.cat(targ_list, dim=0)
    b_lmask = torch.cat(lmask_list, dim=0)

    logits = belief_net.forward_teacher_forced(b_tok, b_hid, b_targs)
    valid_logits = logits[b_lmask]
    valid_targets = b_targs[b_lmask]

    probs = F.softmax(valid_logits, dim=-1)
    one_hot = F.one_hot(valid_targets, num_classes=NUM_PIECE_TYPES).float()

    nll = F.cross_entropy(valid_logits, valid_targets).item()
    acc = (probs.argmax(dim=-1) == valid_targets).float().mean().item() * 100.0
    brier = ((probs - one_hot) ** 2).sum(dim=-1).mean().item()

    unif_probs = torch.full_like(probs, 1.0 / NUM_PIECE_TYPES)
    unif_nll = math.log(NUM_PIECE_TYPES)
    unif_brier = ((unif_probs - one_hot) ** 2).sum(dim=-1).mean().item()
    return {"nll": nll, "acc": acc, "brier": brier, "unif_nll": unif_nll, "unif_brier": unif_brier}