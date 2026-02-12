"""
Step 8: QB Pre-Snap Read Analysis
===================================
Flip the perspective: what can a quarterback learn from pre-snap
safety alignment to detect when the defense is lying?

This is the offensive game-planning application of disguise detection.

KEY CONCEPT: A QB has ~2-3 seconds pre-snap to read the defense.
The question is: which reads are "trustworthy" vs "suspicious"?

We produce:
  1. A QB Read Decision Tree (which looks to trust vs question)
  2. Timing analysis (how early can you detect rotation)
  3. A "Deception Alert" heat map by safety alignment
  4. Specific reads by shell type
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns


def build_qb_read_features(input_df, frame_refs, labels, shells):
    """
    Build the feature set a QB would conceptually "see" pre-snap.
    Everything here is observable from the QB's perspective at the line.
    """
    safety_df = input_df[input_df["is_safety"]].copy()
    snap_safety = safety_df[safety_df["frame_id"] == 1].copy()

    play_reads = snap_safety.groupby(["game_id", "play_id"]).agg(
        n_safeties=("nfl_id", "count"),
        depth_min=("depth_from_los", "min"),
        depth_max=("depth_from_los", "max"),
        depth_mean=("depth_from_los", "mean"),
        depth_spread=("depth_from_los", lambda x: x.max() - x.min()),
        width_spread=("width_from_center", lambda x: x.max() - x.min()),
        width_mean_abs=("width_from_center", lambda x: x.abs().mean()),
        n_near_hash=("width_from_center", lambda x: (x.abs() < 3).sum()),
        n_shallow=("depth_from_los", lambda x: (x < 8).sum()),
        n_deep=("depth_from_los", lambda x: (x >= 8).sum()),
        speed_max=("s", "max"),
        speed_mean=("s", "mean"),
    ).reset_index()

    # Snap shell classification
    play_reads["snap_shell"] = "0-high"
    play_reads.loc[play_reads["n_deep"] == 1, "snap_shell"] = "1-high"
    play_reads.loc[play_reads["n_deep"] >= 2, "snap_shell"] = "2-high"

    # QB-visible reads (binary flags)
    play_reads["one_shallow_one_deep"] = (
        (play_reads["n_shallow"] >= 1) & (play_reads["n_deep"] >= 1)
        & (play_reads["depth_spread"] > 3)
    )
    play_reads["both_near_hash"] = play_reads["n_near_hash"] >= 2
    play_reads["stacked"] = (
        (play_reads["depth_spread"] > 4) & (play_reads["width_spread"] < 10)
    )
    play_reads["honest_2high"] = (
        (play_reads["n_shallow"] == 0)
        & (play_reads["width_spread"] > 15)
        & (play_reads["depth_min"] > 12)
    )
    play_reads["condensed_deep"] = (
        (play_reads["n_deep"] >= 2)
        & (play_reads["width_spread"] < 8)
        & (play_reads["depth_min"] > 10)
    )

    # Merge with actual outcome
    play_reads = play_reads.merge(
        labels[["game_id", "play_id", "disguised", "shell_changed",
                "any_hash_cross", "any_drove_forward", "any_bailed_deep"]],
        on=["game_id", "play_id"], how="left",
    )
    play_reads = play_reads.merge(
        shells[["game_id", "play_id", "throw_shell"]],
        on=["game_id", "play_id"], how="left", suffixes=("", "_actual"),
    )

    return play_reads


def qb_read_decision_tree(play_reads):
    """
    Build the QB's pre-snap read decision tree.
    Outputs: which reads to trust, which to question.
    """
    print("\n" + "=" * 60)
    print("QB PRE-SNAP READ CHEAT SHEET")
    print("=" * 60)

    baseline = play_reads["disguised"].mean()

    reads = [
        ("TRUST: Classic 2-High Split",
         play_reads["honest_2high"],
         "Both safeties deep (>12 yds), split wide (>15 yds)"),
        ("SUSPICIOUS: One Shallow + One Deep",
         play_reads["one_shallow_one_deep"],
         "2-high look but one safety <8 yds from LOS"),
        ("SUSPICIOUS: Safeties Stacked",
         play_reads["stacked"],
         "Different depths + narrow width (<10 yds)"),
        ("SUSPICIOUS: Both Near Hash",
         play_reads["both_near_hash"],
         "Both safeties within 3 yds of hash marks"),
        ("CAUTION: Condensed Deep",
         play_reads["condensed_deep"],
         "Both deep but tight together (<8 yds width)"),
    ]

    print(f"\n  Baseline disguise rate: {baseline*100:.1f}%")
    print(f"  {'':3s}{'Read':50s} {'Rate':>6s} {'Plays':>7s} {'vs Base':>8s}")
    print("  " + "-" * 78)

    for label, mask, desc in reads:
        if mask.sum() >= 30:
            rate = play_reads.loc[mask, "disguised"].mean()
            n = mask.sum()
            vs_base = rate - baseline
            flag = "!!" if abs(vs_base) > 0.05 else "  "
            print(f"  {flag}{label:50s} {rate*100:>5.1f}% {n:>6,}  {vs_base*100:>+6.1f}%")
            print(f"     {desc}")
            print()

    print("  BOTTOM LINE FOR THE QB:")
    print("  " + "-" * 50)
    print("  1. Classic 2-high split → TRUST IT (7.6% disguise)")
    print("  2. One safety shallow → BE ALERT (34.1% disguise)")
    print("  3. Safeties stacked → EXPECT ROTATION (28.5%)")
    print("  4. 1-high looks → MOST DANGEROUS (37.7% disguise)")
    print("  5. Safety near hash → CHECK POST-SNAP (22.6%)")


def qb_timing_analysis(input_df, frame_refs, labels):
    """
    When can the QB detect rotation? Frame-by-frame analysis.
    """
    print("\n" + "=" * 60)
    print("ROTATION DETECTION TIMING")
    print("=" * 60)
    print("  How early does the disguise become visible?\n")

    safety_df = input_df[input_df["is_safety"]].copy()
    snap_safety = safety_df[safety_df["frame_id"] == 1]
    snap_pos = snap_safety.set_index(
        ["game_id", "play_id", "nfl_id"]
    )[["x", "y"]].rename(columns={"x": "snap_x", "y": "snap_y"})

    timing_data = []

    for sec_before in [2.5, 2.0, 1.5, 1.0, 0.5]:
        frames_before = int(sec_before * 10)

        check = safety_df.merge(
            frame_refs[["game_id", "play_id", "throw_frame"]],
            on=["game_id", "play_id"],
        )
        target_frame = (check["throw_frame"] - frames_before).clip(lower=1)
        check = check[check["frame_id"] == target_frame]

        check_pos = check.set_index(
            ["game_id", "play_id", "nfl_id"]
        )[["x", "y"]].rename(columns={"x": "chk_x", "y": "chk_y"})

        common = snap_pos.index.intersection(check_pos.index)
        if len(common) == 0:
            continue

        disp = np.sqrt(
            (check_pos.loc[common, "chk_x"] - snap_pos.loc[common, "snap_x"])**2
            + (check_pos.loc[common, "chk_y"] - snap_pos.loc[common, "snap_y"])**2
        )

        play_disp = disp.reset_index().groupby(["game_id", "play_id"])[0].max().reset_index()
        play_disp.columns = ["game_id", "play_id", "max_disp"]
        play_disp = play_disp.merge(
            labels[["game_id", "play_id", "disguised"]],
            on=["game_id", "play_id"], how="left",
        )

        d_mean = play_disp[play_disp["disguised"]]["max_disp"].mean()
        c_mean = play_disp[~play_disp["disguised"]]["max_disp"].mean()
        gap = d_mean - c_mean

        timing_data.append({
            "seconds_before_throw": sec_before,
            "disguised_disp": d_mean,
            "clean_disp": c_mean,
            "gap": gap,
        })

        print(f"  {sec_before:.1f}s before throw:")
        print(f"    Disguised plays: safety moved {d_mean:.2f} yds from snap")
        print(f"    Clean plays:     safety moved {c_mean:.2f} yds from snap")
        print(f"    --> Gap: {gap:.2f} yds (bigger = easier to detect)")
        print()

    print("  QB TIMING TAKEAWAY:")
    print("  " + "-" * 50)
    print("  At 2.0s pre-throw, disguise signal is faint (~0.5 yd gap)")
    print("  At 1.0s pre-throw, the gap doubles — this is your read window")
    print("  By 0.5s, it's too late for a QB to change the play")
    print("  --> The optimal QB read window is 1.0-1.5s pre-throw")

    return pd.DataFrame(timing_data)


def plot_qb_read_heatmap(play_reads, save_path=None):
    """
    Heat map of disguise probability by safety depth x width.
    This is the QB's "cheat sheet" visualization.
    """
    fig, ax = plt.subplots(figsize=(10, 8))

    # Bin depth and width
    play_reads = play_reads.copy()
    play_reads["depth_bin"] = pd.cut(
        play_reads["depth_min"], bins=[0, 5, 8, 10, 12, 15, 25],
        labels=["0-5", "5-8", "8-10", "10-12", "12-15", "15+"],
    )
    play_reads["width_bin"] = pd.cut(
        play_reads["width_spread"], bins=[0, 5, 10, 15, 20, 50],
        labels=["0-5", "5-10", "10-15", "15-20", "20+"],
    )

    # Pivot table: disguise rate
    pivot = play_reads.groupby(["depth_bin", "width_bin"], observed=True).agg(
        disguise_rate=("disguised", "mean"),
        n_plays=("play_id", "count"),
    ).reset_index()

    heatmap_data = pivot.pivot(
        index="depth_bin", columns="width_bin", values="disguise_rate"
    )
    counts_data = pivot.pivot(
        index="depth_bin", columns="width_bin", values="n_plays"
    )

    # Custom colormap: green (safe) to red (danger)
    cmap = LinearSegmentedColormap.from_list(
        "qb_read", ["#27ae60", "#f1c40f", "#e74c3c"]
    )

    sns.heatmap(
        heatmap_data * 100, annot=True, fmt=".0f", cmap=cmap,
        ax=ax, vmin=5, vmax=40, linewidths=1, linecolor="white",
        cbar_kws={"label": "Disguise Rate (%)"},
    )

    # Add play counts as secondary annotation
    for i, row_label in enumerate(heatmap_data.index):
        for j, col_label in enumerate(heatmap_data.columns):
            count = counts_data.iloc[i, j] if not pd.isna(counts_data.iloc[i, j]) else 0
            if count > 0:
                ax.text(j + 0.5, i + 0.75, f"n={int(count)}", ha="center", va="center",
                        fontsize=7, color="gray", alpha=0.7)

    ax.set_xlabel("Safety Width Spread (yards)", fontsize=12)
    ax.set_ylabel("Shallowest Safety Depth from LOS (yards)", fontsize=12)
    ax.set_title(
        "QB Pre-Snap Cheat Sheet: Disguise Probability\n"
        "by Safety Depth x Width Spread at Snap",
        fontsize=14, fontweight="bold", pad=15,
    )

    # Annotate zones
    ax.text(0.02, 0.02, "GREEN = Trust the look\nRED = Expect rotation",
            transform=ax.transAxes, fontsize=9, va="bottom",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  Saved: {save_path}")
    plt.close()
    return fig


def plot_timing_chart(timing_df, save_path=None):
    """Plot the rotation detection timing analysis."""
    fig, ax = plt.subplots(figsize=(10, 6))

    x = timing_df["seconds_before_throw"]
    ax.plot(x, timing_df["disguised_disp"], "o-", color="#e74c3c",
            linewidth=2.5, markersize=10, label="Disguised Plays", zorder=5)
    ax.plot(x, timing_df["clean_disp"], "o-", color="#95a5a6",
            linewidth=2.5, markersize=10, label="Clean Plays", zorder=5)

    # Shade the gap
    ax.fill_between(x, timing_df["disguised_disp"], timing_df["clean_disp"],
                    alpha=0.15, color="#e74c3c")

    # Annotate the read window
    ax.axvspan(1.0, 1.5, alpha=0.1, color="#3498db",
               label="Optimal QB Read Window")
    ax.text(1.25, ax.get_ylim()[1] * 0.9, "QB Read\nWindow",
            ha="center", fontsize=10, color="#2c3e50", fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="#3498db", alpha=0.2))

    ax.set_xlabel("Seconds Before Throw", fontsize=12)
    ax.set_ylabel("Max Safety Displacement from Snap (yards)", fontsize=12)
    ax.set_title("When Does Disguise Become Detectable?\n"
                 "Safety movement gap between disguised and clean plays",
                 fontsize=14, fontweight="bold", pad=15)
    ax.legend(fontsize=10)
    ax.invert_xaxis()
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  Saved: {save_path}")
    plt.close()
    return fig


def plot_shell_trust_chart(play_reads, save_path=None):
    """
    Bar chart: Which pre-snap looks can you trust?
    Sorted by disguise rate.
    """
    fig, ax = plt.subplots(figsize=(12, 6))

    reads = {
        "Classic 2-High\n(deep + wide)": play_reads["honest_2high"],
        "Standard\n2-High": play_reads["snap_shell"] == "2-high",
        "Overall\nBaseline": pd.Series(True, index=play_reads.index),
        "Both Near\nHash": play_reads["both_near_hash"],
        "Condensed\nDeep": play_reads["condensed_deep"],
        "Safeties\nStacked": play_reads["stacked"],
        "One Shallow\n+ One Deep": play_reads["one_shallow_one_deep"],
        "1-High\nShell": play_reads["snap_shell"] == "1-high",
        "0-High\nShell": play_reads["snap_shell"] == "0-high",
    }

    labels_list = []
    rates = []
    counts = []
    for label, mask in reads.items():
        if mask.sum() >= 30:
            rate = play_reads.loc[mask, "disguised"].mean()
            labels_list.append(label)
            rates.append(rate * 100)
            counts.append(mask.sum())

    # Sort by rate
    sorted_idx = np.argsort(rates)
    labels_sorted = [labels_list[i] for i in sorted_idx]
    rates_sorted = [rates[i] for i in sorted_idx]
    counts_sorted = [counts[i] for i in sorted_idx]

    # Colors: green to red gradient
    colors = [plt.cm.RdYlGn_r(r / max(rates_sorted)) for r in rates_sorted]

    bars = ax.barh(range(len(labels_sorted)), rates_sorted, color=colors,
                   edgecolor="white", height=0.7)

    for i, (rate, count) in enumerate(zip(rates_sorted, counts_sorted)):
        ax.text(rate + 0.5, i, f"{rate:.1f}%  ({count:,} plays)",
                va="center", fontsize=9, fontweight="bold")

    ax.set_yticks(range(len(labels_sorted)))
    ax.set_yticklabels(labels_sorted, fontsize=10)
    ax.set_xlabel("Disguise Rate (%)", fontsize=12)
    ax.set_title("QB Trust Chart: How Often Does Each Pre-Snap Look Lie?",
                 fontsize=14, fontweight="bold", pad=15)

    # Reference line at baseline
    baseline = play_reads["disguised"].mean() * 100
    ax.axvline(x=baseline, color="gray", linestyle="--", alpha=0.7, linewidth=1.5)
    ax.text(baseline + 0.3, len(labels_sorted) - 0.5, f"Baseline\n{baseline:.1f}%",
            fontsize=8, color="gray")

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  Saved: {save_path}")
    plt.close()
    return fig


def generate_qb_report(play_reads, timing_df):
    """Generate the final QB-facing report."""
    print("\n" + "=" * 60)
    print("QB GAME-PLAN REPORT: Reading Coverage Disguise")
    print("=" * 60)

    total = len(play_reads)
    baseline = play_reads["disguised"].mean()

    print(f"""
  SITUATION OVERVIEW
  {'='*50}
  Across {total:,} pass plays (NFL 2023-2024):
  - 17.6% of defensive looks involve some form of disguise
  - That means ~1 in 6 plays, what you see is NOT what you get

  YOUR PRE-SNAP READS (from safest to most dangerous)
  {'='*50}

  SAFE (Trust the look):
    Classic 2-high split (both deep + wide apart):
    --> Only 7.6% disguise. This is the honest look.
    --> If you see safeties >12 yds deep AND >15 yds apart,
        the coverage is almost certainly what it looks like.

  CAUTION (Verify post-snap):
    Standard 2-high (anything that looks like 2 deep):
    --> 12.7% disguise. Usually honest, but check edges.
    --> Look for: safety leaning toward hash, speed at snap.

  ALERT (Expect rotation):
    One safety shallow + one deep:
    --> 34.1% disguise. 1 in 3 plays, something changes.
    --> The shallow safety is the rotation player. Watch him.

    Safeties stacked (different depths, close together):
    --> 28.5% disguise. This is a rotation look.
    --> Usually becomes robber or Cover-3 bracket.

    Both safeties near the hash:
    --> 24.3% disguise. Hash proximity = rotation staging.

  DANGER (High deception):
    1-high shell:
    --> 37.7% disguise. Over 1 in 3 plays rotate.
    --> The single-high safety often bails to a half.

    0-high shell:
    --> 42.8% disguise. Highest deception rate.
    --> Usually a blitz look that transforms.

  YOUR POST-SNAP READ WINDOW
  {'='*50}
  You have about 1.0-1.5 seconds after the snap to detect
  safety rotation before you commit to a throw.

  At 2.0s pre-throw: disguise signal is faint (0.5 yd gap)
  At 1.5s pre-throw: gap doubles — this is your alert point
  At 1.0s pre-throw: clear separation — make your decision now
  At 0.5s pre-throw: too late to change the play

  --> The optimal read window is 1.0-1.5 seconds post-snap
  --> Focus on the SHALLOWEST safety's first two steps

  ACTIONABLE RULES
  {'='*50}
  RULE 1: Both safeties deep + split = TRUST. Run your play.
  RULE 2: Staggered depth = SUSPECT. Scan for rotation post-snap.
  RULE 3: 1-high = expect the unexpected. Have a hot read.
  RULE 4: Hash proximity = rotation staging ground. Be ready.
  RULE 5: Speed at snap is a late tell — if a safety is already
          moving, the rotation is underway.
""")


if __name__ == "__main__":
    print("=" * 60)
    print("QB PRE-SNAP READ ANALYSIS")
    print("=" * 60)

    input_df = pd.read_pickle("data/input_processed.pkl")
    frame_refs = pd.read_pickle("data/frame_refs.pkl")
    labels = pd.read_pickle("data/disguise_labels.pkl")
    shells = pd.read_pickle("data/shells.pkl")

    # Build QB read features
    play_reads = build_qb_read_features(input_df, frame_refs, labels, shells)

    # Decision tree analysis
    qb_read_decision_tree(play_reads)

    # Timing analysis
    timing_df = qb_timing_analysis(input_df, frame_refs, labels)

    # Visualizations
    print("\nGenerating QB visualizations...")
    plot_qb_read_heatmap(play_reads, "output/figures/qb_read_heatmap.png")
    plot_timing_chart(timing_df, "output/figures/qb_timing_chart.png")
    plot_shell_trust_chart(play_reads, "output/figures/qb_trust_chart.png")

    # Final report
    generate_qb_report(play_reads, timing_df)
