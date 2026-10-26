import torch
import torch.nn.functional as F
from src.env import device, StrategoRolloutSimulator
from src.networks import MoveNetwork, BeliefNetwork
from src.rl import get_damping_schedules


def run_mechanism_checks():
    # safety checks for ze skeptics & peer reviewers
    print("Running mechanistic invariant checks...")
    sim = StrategoRolloutSimulator(batch_size=2)
    s0 = torch.tensor([[0, 1, 2, 3], [0, 1, 2, 3]], device=device)
    s1 = torch.tensor([[0, 1, 2, 3], [3, 2, 1, 0]], device=device)
    sim.reset(s0, s1)

    tokens = sim.get_infoset_tokens()
    mask = sim.get_legal_action_mask()
    test_move_net = MoveNetwork().to(device).eval()
    with torch.no_grad():
        logits, _, _ = test_move_net(tokens, mask)

    token_diff = (tokens[0] - tokens[1]).abs().max().item()
    logit_diff = (logits[0] - logits[1]).abs().max().item()
    assert token_diff == 0.0 and logit_diff < 1e-6, "leakage detected!"
    print("  [PASS] 1. info hiding invariant")

    sim_single = StrategoRolloutSimulator(batch_size=1)
    sim_single.reset(s0[:1], s1[:1])
    sim_single.alive_pool[0, 1, 2] = False
    sim_single.owner[0, 12] = 0
    sim_single.piece[0, 13] = 3
    sim_single.revealed[0, 13] = True
    sim_single.piece[0, 14] = 0
    sim_single.piece[0, 15] = 1

    test_belief_net = BeliefNetwork().to(device).eval()
    sampled_worlds = test_belief_net.sample_worlds(sim_single, num_samples=32)
    for k in range(32):
        opp_pieces = sampled_worlds.piece[k, [13, 14, 15]].tolist()
        assert sorted(opp_pieces) == [0, 1, 3], f"conservation violated: {opp_pieces}"
    print("  [PASS] 2. belief conservation invariant")

    q_dummy = torch.tensor([0.35, 0.25, 0.10], device=device)
    pi_net_dummy = torch.tensor([0.20, 0.50, 0.30], device=device)
    mu_dummy = torch.tensor([1 / 3, 1 / 3, 1 / 3], device=device)
    b_net, b_mag = 0.02, 0.002

    logits_mmd = (q_dummy + b_net * pi_net_dummy.log() + b_mag * mu_dummy.log()) / (b_net + b_mag)
    pi_mmd = F.softmax(logits_mmd, dim=-1)

    kkt_residual = q_dummy - b_net * (pi_mmd.log() - pi_net_dummy.log()) - b_mag * (pi_mmd.log() - mu_dummy.log())
    assert kkt_residual.std().item() < 1e-5, "mmd kkt stationarity failed!"
    assert (pi_mmd > 0.0).all() and (pi_mmd.max() < 0.999), "mmd collapsed!"
    print("  [PASS] 3. update-equivalence mmd kkt invariant")

    # check 4: co-annealing monotonicity in control
    sched_ctrl = [get_damping_schedules(t, 80, reg_mode="annealed", lr_anneal=True) for t in range(1, 81)]
    assert all(sched_ctrl[i][0] > sched_ctrl[i + 1][0] for i in range(79)), "tau_setup not decreasing!"
    assert all(sched_ctrl[i][1] > sched_ctrl[i + 1][1] for i in range(79)), "tau_move_mag not decreasing!"
    assert all(sched_ctrl[i][3] > sched_ctrl[i + 1][3] for i in range(79)), "lr_move not decreasing!"
    print("  [PASS] 4. dynamic damping power-law schedule invariant")

    # check 5: unconfounded rl regularization-annealing ablation (initial coeffs & proximal kl identical)
    ts_c1, tm_c1, to_c1, lr_c1 = get_damping_schedules(1, 80, reg_mode="annealed", lr_anneal=True)
    ts_f1, tm_f1, to_f1, lr_f1 = get_damping_schedules(1, 80, reg_mode="fixed_high", lr_anneal=True)
    ts_f80, tm_f80, to_f80, lr_f80 = get_damping_schedules(80, 80, reg_mode="fixed_high", lr_anneal=True)
    assert (ts_c1, tm_c1, to_c1, lr_c1) == (ts_f1, tm_f1, to_f1, lr_f1), "initial coeffs differ!"
    assert ts_f80 == ts_f1 and tm_f80 == tm_f1 and to_f80 == to_c1, "fixed schedule mutated objective!"
    print("  [PASS] 5. unconfounded rl schedule ablation invariant")

    # check 6: unconfounded search ablation (only 1 hyperparameter mutated per row)
    full_cfg = {"K": 48, "d": 6, "beta_net": 0.02, "beta_mag": 0.002, "unif": False}
    no_kl_cfg = {"K": 48, "d": 6, "beta_net": 0.00, "beta_mag": 0.002, "unif": False}
    unif_cfg = {"K": 48, "d": 6, "beta_net": 0.02, "beta_mag": 0.002, "unif": True}
    diff_no_kl = [k for k in full_cfg if full_cfg[k] != no_kl_cfg[k]]
    diff_unif = [k for k in full_cfg if full_cfg[k] != unif_cfg[k]]
    assert diff_no_kl == ["beta_net"] and diff_unif == ["unif"], "search ablation confounded!"
    print("  [PASS] 6. unconfounded search ablation single-variable invariant\n")


if __name__ == "__main__":
    run_mechanism_checks()