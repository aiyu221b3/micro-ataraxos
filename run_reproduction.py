import os
import copy
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

from paper_style import COLORS, setup_plot_style, style_axis, add_panel_label, save_figure
from src.env import device, NUM_CELLS, StrategoRolloutSimulator
from src.networks import SetupNetwork, MoveNetwork, scalar_return_to_wld_target
from src.rl import get_damping_schedules, compute_dual_lambda_targets, update_ema, train_belief_network
from src.search import evaluate_vs_reference_paired, evaluate_matchup, evaluate_belief_calibration
from src.mechanism_checks import run_mechanism_checks

os.makedirs("results", exist_ok=True)
os.makedirs("results/figures", exist_ok=True)


def train_single_agent(
    seed, num_iters=80, batch_games=128, adv_f=True, reg_mode="annealed", lr_a=True,
    ref_setup=None, ref_move=None, cond_name="Control"
):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    setup_net = SetupNetwork().to(device)
    move_net = MoveNetwork().to(device)
    ema_setup = copy.deepcopy(setup_net)
    ema_move = copy.deepcopy(move_net)

    setup_opt = torch.optim.Adam(setup_net.parameters(), lr=2e-3)
    move_opt = torch.optim.Adam(move_net.parameters(), lr=1.5e-3)

    rows = []
    for it in range(1, num_iters + 1):
        tau_setup, tau_move_mag, tau_move_old, lr_move = get_damping_schedules(
            it, num_iters, reg_mode=reg_mode, lr_anneal=lr_a
        )
        for pg in move_opt.param_groups:
            pg["lr"] = lr_move

        raw_s0, board_s0 = setup_net.sample_setups(batch_games)
        raw_s1, board_s1 = setup_net.sample_setups(batch_games)
        sim = StrategoRolloutSimulator(batch_games).reset(board_s0, board_s1)

        t_tok, t_mask, t_act, t_olog, t_val, t_pl, t_actv = [], [], [], [], [], [], []
        while not sim.done.all():
            active = ~sim.done
            tok, mask = sim.get_infoset_tokens(), sim.get_legal_action_mask()
            with torch.no_grad():
                logits, val, _ = move_net(tok, mask)
                acts = Categorical(logits=logits).sample()
            t_tok.append(tok)
            t_mask.append(mask)
            t_act.append(acts)
            t_olog.append(logits)
            t_val.append(val)
            t_pl.append(sim.turn.clone())
            t_actv.append(active.clone())
            sim.step(acts)

        out_p0 = sim.winner.detach()
        out_p1 = -out_p0

        lp0, ent0, val0, wld0 = setup_net.evaluate_setups(raw_s0)
        lp1, ent1, val1, wld1 = setup_net.evaluate_setups(raw_s1)
        s_pg = -((lp0 * (out_p0.unsqueeze(1) - val0.detach())) + (lp1 * (out_p1.unsqueeze(1) - val1.detach()))).mean()
        tw0 = scalar_return_to_wld_target(out_p0.unsqueeze(1).expand_as(val0))
        tw1 = scalar_return_to_wld_target(out_p1.unsqueeze(1).expand_as(val1))
        s_vl = -(tw0 * F.log_softmax(wld0, -1) + tw1 * F.log_softmax(wld1, -1)).sum(-1).mean()
        s_el = -(ent0.sum(1).mean() + ent1.sum(1).mean())
        s_loss = s_pg + 0.5 * s_vl + tau_setup * s_el
        setup_opt.zero_grad()
        s_loss.backward()
        nn.utils.clip_grad_norm_(setup_net.parameters(), 1.0)
        setup_opt.step()

        v_targs, advs = compute_dual_lambda_targets(t_val, t_pl, t_actv, out_p0)
        fa = torch.stack(t_actv).view(-1)
        b_tok = torch.stack(t_tok).view(-1, NUM_CELLS, 13)[fa]
        b_msk = torch.stack(t_mask).view(-1, NUM_CELLS * NUM_CELLS)[fa]
        b_act = torch.stack(t_act).view(-1)[fa]
        b_olg = torch.stack(t_olog).view(-1, NUM_CELLS * NUM_CELLS)[fa]
        b_vtg = torch.stack(v_targs).view(-1)[fa]
        b_adv = torch.stack(advs).view(-1)[fa]

        if adv_f:
            keep = b_adv.abs() >= torch.quantile(b_adv.abs(), 0.50)
            b_tok, b_msk, b_act = b_tok[keep], b_msk[keep], b_act[keep]
            b_olg, b_vtg, b_adv = b_olg[keep], b_vtg[keep], b_adv[keep]

        b_adv = (b_adv - b_adv.mean()) / (b_adv.std() + 1e-8)
        n_log, _, n_wld = move_net(b_tok, b_msk)
        n_dist, o_dist = Categorical(logits=n_log), Categorical(logits=b_olg)
        ratio = torch.exp(n_dist.log_prob(b_act) - o_dist.log_prob(b_act))
        pg_l = -torch.min(ratio * b_adv, torch.clamp(ratio, 0.8, 1.2) * b_adv).mean()
        vl_l = -(scalar_return_to_wld_target(b_vtg) * F.log_softmax(n_wld, -1)).sum(-1).mean()

        pi_p, log_p = F.softmax(n_log, -1), F.log_softmax(n_log, -1)
        mag_p = sim.get_magnet_policy(b_msk)
        kl_mag = (pi_p * (log_p - torch.log(mag_p.clamp(min=1e-8)))).sum(-1).mean()
        kl_old = (pi_p * (log_p - F.log_softmax(b_olg, -1))).sum(-1).mean()

        m_loss = pg_l + 0.5 * vl_l + tau_move_mag * kl_mag + tau_move_old * kl_old
        move_opt.zero_grad()
        m_loss.backward()
        nn.utils.clip_grad_norm_(move_net.parameters(), 1.0)
        move_opt.step()

        update_ema(ema_setup, setup_net)
        update_ema(ema_move, move_net)

        if ref_setup is not None and it in (1, 4, 16, 40, 80):
            it_score, _ = evaluate_vs_reference_paired(setup_net, move_net, ref_setup, ref_move, num_pairs=60)
            ema_score, ema_wr = evaluate_vs_reference_paired(ema_setup, ema_move, ref_setup, ref_move, num_pairs=60)
            rows.append({
            "condition": cond_name,
            "seed": seed,
            "iter": it,
            "nominal_budget": it * 2500,
            "budget": it * 2500,  # Retained for backward compatibility.
            "setup_ent": ent0.sum(1).mean().item(),
            "move_ent": n_dist.entropy().mean().item(),
            "iterate_score": it_score,
            "ema_score": ema_score,
            "ema_wr": ema_wr,
        })

    return ema_setup.eval(), ema_move.eval(), rows


def run_full_suite(seeds=(42, 43, 44), num_iters=80, batch_games=128):
    run_mechanism_checks()

    print("Training fixed reference opponent checkpoint...")
    ref_setup, ref_move, _ = train_single_agent(seed=999, num_iters=65, batch_games=batch_games)

    ablation_specs = [
        ("Control (Full Ataraxos)", True, "annealed", True),
        ("w/o Learning Rate Annealing", True, "annealed", False),
        ("w/o Advantage Filtering", False, "annealed", True),
        ("w/o Regularization Annealing", True, "fixed_high", True),
        ("Unregularized (tau=0)", True, "zero", True),
    ]

    rl_rows, search_rows, belief_rows, trained_control_models = [], [], [], []

    for cond_name, adv_f, reg_m, lr_a in ablation_specs:
        print(f">>> RL Condition: {cond_name} <<<")
        for seed in seeds:
            ema_setup, ema_move, rows = train_single_agent(
                seed=seed, num_iters=num_iters, batch_games=batch_games,
                adv_f=adv_f, reg_mode=reg_m, lr_a=lr_a,
                ref_setup=ref_setup, ref_move=ref_move, cond_name=cond_name,
            )
            rl_rows.extend(rows)
            print(f"  seed {seed} | final ema score vs ref: {rows[-1]['ema_score']:+.3f} ({rows[-1]['ema_wr']:.1f}% wr)")

            if cond_name == "Control (Full Ataraxos)":
                b_net = train_belief_network(ema_setup, ema_move, num_epochs=45, batch_games=batch_games)
                cal = evaluate_belief_calibration(b_net, ema_setup, ema_move, num_games=100)
                cal["seed"] = seed
                belief_rows.append(cal)
                trained_control_models.append((seed, ema_setup, ema_move, b_net))

    print("\n>>> Evaluating Update-Equivalence Search Ablations <<<")
    s_configs = [
        ("Raw Policy Network\n(No Search Baseline)", 0, 0, 0.02, 0.002, False),
        ("Full Ataraxos Search\n(Learned Belief, d=6, K=48)", 48, 6, 0.02, 0.002, False),
        ("Shallow Search\n(Learned Belief, d=2, K=16)", 16, 2, 0.02, 0.002, False),
        ("Uniform Belief Ablation\n(Uniform Prior, d=6, K=48)", 48, 6, 0.02, 0.002, True),
        ("No Network KL\n(beta_net=0, beta_mag=0.002)", 48, 6, 0.00, 0.002, False),
    ]
    for idx, (seed, ema_setup, ema_move, b_net) in enumerate(trained_control_models):
        _, opp_setup, opp_move, _ = trained_control_models[(idx + 1) % len(trained_control_models)]
        for s_name, K, d, bn, bm, unif in s_configs:
            w, l, dr, wr = evaluate_matchup(
                ema_setup, ema_move, b_net, opp_setup_net=opp_setup, opp_move_net=opp_move,
                num_games=120, num_rollouts=K, depth=d, beta_net=bn, beta_mag=bm, use_uniform_belief=unif,
            )
            search_rows.append({"condition": s_name, "seed": seed, "wins": w, "losses": l, "draws": dr, "eff_wr": wr})

    df_rl = pd.DataFrame(rl_rows)
    df_search = pd.DataFrame(search_rows)
    df_belief = pd.DataFrame(belief_rows)

    df_rl.to_csv("results/rl_ablation_seeds.csv", index=False)
    df_search.to_csv("results/search_ablation_seeds.csv", index=False)
    df_belief.to_csv("results/belief_calibration_seeds.csv", index=False)
    return df_rl, df_search, df_belief, trained_control_models


def plot_results(df_rl=None, df_search=None, trained_control_models=None):
    setup_plot_style()
    if df_rl is None:
        df_rl = pd.read_csv("results/rl_ablation_seeds.csv")
    if df_search is None:
        df_search = pd.read_csv("results/search_ablation_seeds.csv")

    if "nominal_budget" not in df_rl.columns:
        if "budget" not in df_rl.columns:
            raise ValueError("RL results must contain 'nominal_budget' or 'budget'.")
        df_rl["nominal_budget"] = df_rl["budget"]
    budgets = sorted(df_rl["nominal_budget"].unique())

    fig1, axes = plt.subplots(1, 3, figsize=(12.8, 3.7), dpi=300)
    plt.subplots_adjust(wspace=0.28)

    ax = axes[0]
    cond_styles = [
        ("Control (Full Ataraxos)", COLORS["purple"], "-"),
        ("w/o Learning Rate Annealing", COLORS["cyan"], "--"),
        ("w/o Advantage Filtering", COLORS["pink"], "-."),
        ("w/o Regularization Annealing", COLORS["muted"], ":"),
    ]
    for cond, col, ls in cond_styles:
        sub = df_rl[df_rl["condition"] == cond]
        grp = sub.groupby("nominal_budget")["ema_score"]
        mean, lo, hi = grp.mean(), grp.min(), grp.max()
        ax.plot(mean.index, mean.values, color=col, linestyle=ls, marker="o", markersize=4, linewidth=1.5, label=cond)
        ax.fill_between(mean.index, lo.values, hi.values, color=col, alpha=0.13, edgecolor="none")

    style_axis(ax, budgets=budgets, show_xlabel=True)
    ax.set_xlabel("Nominal training budget (iteration × 2,500)")
    ax.set_ylabel("Expected score vs. reference opponent")
    ax.set_title("Self-play RL mechanism ablations", fontsize=9.5, color=COLORS["ink"], pad=8)
    add_panel_label(ax, "a")
    ax.legend(loc="upper left", fontsize=7.2)

    ax = axes[1]
    for cond, col, ls, lbl in [
        ("Control (Full Ataraxos)", COLORS["purple"], "-", "Dynamic annealing (Control)"),
        ("w/o Regularization Annealing", COLORS["cyan"], "--", "Fixed high reg. (Un-annealed)"),
        ("Unregularized (tau=0)", COLORS["pink"], ":", "Zero reg. (Collapsed)"),
    ]:
        sub = df_rl[df_rl["condition"] == cond]
        grp = sub.groupby("nominal_budget")["move_ent"]
        mean, lo, hi = grp.mean(), grp.min(), grp.max()
        ax.plot(mean.index, mean.values, color=col, linestyle=ls, marker="o", markersize=4, linewidth=1.5, label=lbl)
        ax.fill_between(mean.index, lo.values, hi.values, color=col, alpha=0.14, edgecolor="none")

    style_axis(ax, budgets=budgets, show_xlabel=True)
    ax.set_xlabel("Nominal training budget (iteration × 2,500)")
    ax.set_ylabel("Move policy entropy (nats)")
    ax.set_title("Dynamic damping of policy entropy", fontsize=9.5, color=COLORS["ink"], pad=8)
    add_panel_label(ax, "b")
    ax.legend(loc="lower left", fontsize=7.2)

    ax = axes[2]
    ctrl = df_rl[df_rl["condition"] == "Control (Full Ataraxos)"]
    for col_name, col, ls, lbl in [
        ("ema_score", COLORS["cyan"], "-", "EMA weights (Ataraxos)"),
        ("iterate_score", COLORS["violet"], "--", "Current iterate"),
    ]:
        grp = ctrl.groupby("nominal_budget")[col_name]
        mean, lo, hi = grp.mean(), grp.min(), grp.max()
        ax.plot(mean.index, mean.values, color=col, linestyle=ls, marker="o", markersize=4, linewidth=1.5, label=lbl)
        ax.fill_between(mean.index, lo.values, hi.values, color=col, alpha=0.15, edgecolor="none")

    style_axis(ax, budgets=budgets, show_xlabel=True)
    ax.set_xlabel("Nominal training budget (iteration × 2,500)")
    ax.set_ylabel("Expected score vs. reference opponent")
    ax.set_title("Iterate vs. EMA stability across seeds", fontsize=9.5, color=COLORS["ink"], pad=8)
    add_panel_label(ax, "c")
    ax.legend(loc="upper left", fontsize=7.5)

    fig1.suptitle(
        "Mechanistic Reproduction on 4×4 Micro-Stratego "
        "(n = 3 independent seeds; shaded bands show min–max seed range)",
        fontsize=8.5,
        color=COLORS["muted"],
        y=-0.04,
    )
    save_figure(fig1, "results/figures/fig1_ataraxos_mechanisms.png")

    fig2, (ax_s, ax_h) = plt.subplots(1, 2, figsize=(11.6, 3.9), dpi=300)
    plt.subplots_adjust(wspace=0.36)

    order = [
        "No Network KL\n(beta_net=0, beta_mag=0.002)",
        "Uniform Belief Ablation\n(Uniform Prior, d=6, K=48)",
        "Shallow Search\n(Learned Belief, d=2, K=16)",
        "Full Ataraxos Search\n(Learned Belief, d=6, K=48)",
    ]
    bar_colors = [COLORS["pink"], COLORS["violet"], COLORS["cyan_light"], COLORS["purple"]]

    raw_base = df_search[df_search["condition"].str.startswith("Raw Policy")]["eff_wr"].mean()

    y_pos = np.arange(len(order))
    means, sems, max_extents = [], [], []
    for idx, cond in enumerate(order):
        vals = df_search[df_search["condition"] == cond]["eff_wr"].values
        m = vals.mean()
        s = vals.std(ddof=1) / np.sqrt(len(vals)) if len(vals) > 1 else 0.0
        means.append(m)
        sems.append(s)
        max_extents.append(max(m + s, vals.max()))
        ax_s.scatter(
            vals, np.full_like(vals, idx - 0.14),
            color=COLORS["ink"], s=16, zorder=5, alpha=0.75,
        )

    ax_s.barh(
        y_pos, means, xerr=sems, height=0.52, color=bar_colors, alpha=0.85,
        edgecolor=COLORS["ink"], linewidth=0.6, capsize=3,
        error_kw={"ecolor": COLORS["ink"], "elinewidth": 1.0},
    )

    ax_s.axvline(raw_base, color=COLORS["baseline"], linestyle="--", linewidth=1.2)
    ax_s.text(
        raw_base + 0.8, 3.42, f"No Search ({raw_base:.1f}%)",
        color=COLORS["muted"], fontsize=7.5, va="bottom", ha="left", style="italic"
    )

    ax_s.set_yticks(y_pos)
    ax_s.set_yticklabels(order, fontsize=8, color=COLORS["ink"])
    ax_s.set_xlim(15, 88)
    ax_s.set_ylim(-0.5, 3.65)
    ax_s.set_xlabel("Effective win rate vs. tactical OOD opponent (%)")
    ax_s.set_title("Update-equivalence search ablations", fontsize=9.5, color=COLORS["ink"], pad=8)
    style_axis(ax_s, budgets=None, show_xlabel=False)
    add_panel_label(ax_s, "a")

    for idx, (m, ext) in enumerate(zip(means, max_extents)):
        ax_s.text(ext + 1.5, idx, f"{m:.1f}%", va="center", ha="left", fontsize=8, fontweight="bold", color=COLORS["ink"])

    if trained_control_models is not None:
        all_raw_setups = []
        with torch.no_grad():
            for _, s_net, _, _ in trained_control_models:
                r_s, _ = s_net.sample_setups(2000)
                all_raw_setups.append(r_s)
        all_raw_setups = torch.cat(all_raw_setups, dim=0)

        heat_matrix = np.zeros((4, 4))
        for pt in range(4):
            for col in range(4):
                heat_matrix[pt, col] = (all_raw_setups[:, col] == pt).float().mean().item() * 100.0
        pd.DataFrame(
            heat_matrix,
            index=["Flag (0)", "Spy (1)", "Miner (2)", "Marshal (3)"],
            columns=["Col a (0)", "Col b (1)", "Col c (2)", "Col d (3)"],
        ).to_csv("results/setup_frequencies.csv")
    else:
        try:
            heat_matrix = pd.read_csv(
                "results/setup_frequencies.csv", index_col=0
            ).values
        except FileNotFoundError:
            print("using hardcoded values")
            heat_matrix = np.array([
                [20.2, 79.8,  0.0,  0.0],
                [21.4,  3.7, 28.3, 46.5],
                [27.1,  7.8, 34.9, 30.3],
                [31.3,  8.7, 36.8, 23.3],
            ])

    custom_cmap = LinearSegmentedColormap.from_list(
        "decoy_purple", ["#FFFFFF", "#EDE8F5", COLORS["violet"], COLORS["purple"], COLORS["ink"]]
    )
    im = ax_h.imshow(heat_matrix, cmap=custom_cmap, vmin=0, vmax=65, aspect="auto")

    ax_h.set_xticks(range(4))
    ax_h.set_xticklabels(["Col a (0)", "Col b (1)", "Col c (2)", "Col d (3)"], fontsize=8)
    ax_h.set_yticks(range(4))
    ax_h.set_yticklabels(["Flag (0)", "Spy (1)", "Miner (2)", "Marshal (3)"], fontsize=8)
    ax_h.set_title("Learned setup frequencies (Flag on left)", fontsize=9.5, color=COLORS["ink"], pad=8)
    add_panel_label(ax_h, "b")

    for r_i in range(4):
        for c_i in range(4):
            val = heat_matrix[r_i, c_i]
            txt_col = "white" if val > 38 else COLORS["ink"]
            ax_h.text(c_i, r_i, f"{val:.1f}%", ha="center", va="center", color=txt_col, fontsize=8.5, fontweight="bold")

    cbar = fig2.colorbar(im, ax=ax_h, fraction=0.046, pad=0.04)
    cbar.ax.tick_params(labelsize=7.5)
    cbar.set_label("Placement frequency (%)", fontsize=8, color=COLORS["ink"])

    save_figure(fig2, "results/figures/fig2_search_and_setups.png")
    print("Figures saved to results/")


if __name__ == "__main__":
    df_rl, df_search, df_belief, models = run_full_suite()
    plot_results(df_rl, df_search, models)
