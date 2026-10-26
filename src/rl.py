import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
from src.env import device, NUM_CELLS, StrategoRolloutSimulator
from src.networks import BeliefNetwork


def get_damping_schedules(it: int, num_iters: int, reg_mode: str = "annealed", lr_anneal: bool = True):
    # distinct power-law schedules for setup entropy, move magnet kl, and adam lr
    progress = (it - 1) / max(1, num_iters - 1)
    tau_setup_init = 0.25
    tau_move_mag_init = 0.18
    base_lr = 1.5e-3

    if reg_mode == "annealed":
        tau_setup = tau_setup_init * ((1.0 - 0.88 * progress) ** 1.5)
        tau_move_mag = tau_move_mag_init * ((1.0 - 0.92 * progress) ** 2.0)
    elif reg_mode == "fixed_high":
        # ablation of annealing alone: hold initial regularization coeffs constant
        tau_setup = tau_setup_init
        tau_move_mag = tau_move_mag_init
    elif reg_mode == "zero":
        # unregularized baseline to show immediate entropy collapse
        tau_setup = 0.0
        tau_move_mag = 0.0
    else:
        raise ValueError(f"unknown reg_mode: {reg_mode}")

    # proximal kl coeff to data collection policy stays constant; step size anneals via lr
    tau_move_old = 0.15
    lr_move = base_lr * ((1.0 - 0.82 * progress) ** 1.0) if lr_anneal else base_lr
    return tau_setup, tau_move_mag, tau_move_old, lr_move


def compute_dual_lambda_targets(traj_values, traj_players, traj_active, outcomes_p0, lam_val=0.98, lam_adv=0.92):
    # distinct lambdas for value cumulants vs advantages, as ordained by sokota et al.
    T = len(traj_values)
    B = outcomes_p0.size(0)
    val_targets = [None] * T
    advantages = [None] * T

    next_val = torch.zeros(B, device=device)
    gae_val = torch.zeros(B, device=device)
    gae_adv = torch.zeros(B, device=device)

    for t in reversed(range(T)):
        pl = traj_players[t].float()
        z = outcomes_p0 * pl

        if t == T - 1:
            is_next_active = torch.zeros(B, dtype=torch.bool, device=device)
        else:
            is_next_active = traj_active[t + 1]

        bootstrap = torch.where(is_next_active, -next_val, z)
        delta = bootstrap - traj_values[t]

        active_float = is_next_active.float()
        gae_val = delta + lam_val * (-gae_val) * active_float
        gae_adv = delta + lam_adv * (-gae_adv) * active_float

        val_targets[t] = (traj_values[t] + gae_val).clamp(-1.0, 1.0)
        advantages[t] = gae_adv
        next_val = traj_values[t]

    return val_targets, advantages


@torch.no_grad()
def update_ema(target_net, source_net, decay=0.95):
    for tp, sp in zip(target_net.parameters(), source_net.parameters()):
        tp.data.mul_(decay).add_(sp.data, alpha=1.0 - decay)


def train_belief_network(setup_net, move_net, num_epochs=45, batch_games=128):
    belief_net = BeliefNetwork().to(device)
    belief_opt = torch.optim.Adam(belief_net.parameters(), lr=1.5e-3)

    for ep in range(1, num_epochs + 1):
        belief_net.train()
        _, board_s0 = setup_net.sample_setups(batch_games)
        _, board_s1 = setup_net.sample_setups(batch_games)
        sim = StrategoRolloutSimulator(batch_games).reset(board_s0, board_s1)

        ep_tokens, ep_hid_masks, ep_targets, ep_loss_masks = [], [], [], []

        while not sim.done.all():
            active = ~sim.done
            tokens = sim.get_infoset_tokens()
            mask = sim.get_legal_action_mask()

            grid = torch.arange(NUM_CELLS, device=device).unsqueeze(0).expand(batch_games, -1)
            c_grid = sim._canonical_idx(grid, sim.turn)
            c_owner = torch.gather(sim.owner, 1, c_grid) * sim.turn.unsqueeze(1)
            c_piece = torch.gather(sim.piece, 1, c_grid)
            c_rev = torch.gather(sim.revealed, 1, c_grid)
            hid_mask = (c_owner == -1) & (~c_rev)

            valid = active & hid_mask.any(dim=1)
            if valid.any():
                v_tok = tokens[valid]
                v_hid = hid_mask[valid]
                v_pc = c_piece[valid]
                N_v = v_tok.size(0)
                targs = torch.zeros((N_v, 4), dtype=torch.long, device=device)
                lmask = torch.zeros((N_v, 4), dtype=torch.bool, device=device)
                for i in range(N_v):
                    ht = v_pc[i][v_hid[i]]
                    m = len(ht)
                    targs[i, :m] = ht
                    lmask[i, :m] = True
                ep_tokens.append(v_tok)
                ep_hid_masks.append(v_hid)
                ep_targets.append(targs)
                ep_loss_masks.append(lmask)

            with torch.no_grad():
                logits, _, _ = move_net(tokens, mask)
                sim.step(Categorical(logits=logits).sample())

        b_tok = torch.cat(ep_tokens, dim=0)
        b_hid = torch.cat(ep_hid_masks, dim=0)
        b_targs = torch.cat(ep_targets, dim=0)
        b_lmask = torch.cat(ep_loss_masks, dim=0)

        perm = torch.randperm(b_tok.size(0), device=device)[:512]
        pred_logits = belief_net.forward_teacher_forced(b_tok[perm], b_hid[perm], b_targs[perm])
        loss = F.cross_entropy(pred_logits[b_lmask[perm]], b_targs[perm][b_lmask[perm]])

        belief_opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(belief_net.parameters(), 1.0)
        belief_opt.step()

    belief_net.eval()
    return belief_net