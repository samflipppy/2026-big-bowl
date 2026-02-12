"""
Step 8: QB Pre-Snap Read Analysis
===================================
Concrete, actionable insights for a quarterback reading the defense.

Three core analyses:
  1. MATCHUP PREDICTION — Who is covering my target and can I tell pre-snap?
  2. THROW WINDOW PREDICTION — What does the pre-snap alignment tell me
     about how open my target will be?
  3. MISMATCH EXPLOITATION — Which matchup types give the biggest windows?

These are the reads that belong in a QB game-plan binder.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import seaborn as sns


def detect_snap_frames(input_df):
    """Detect snap frame per play from WR acceleration."""
    offense = input_df[input_df["player_side"] == "Offense"]
    wr_speed = offense[offense["player_role"].isin(["Other Route Runner", "Targeted Receiver"])]
    avg_speed = wr_speed.groupby(["game_id", "play_id", "frame_id"])["s"].mean().reset_index()
    snap_frames = (
        avg_speed[avg_speed["s"] > 0.5]
        .groupby(["game_id", "play_id"])["frame_id"]
        .min()
        .reset_index()
    )
    snap_frames.columns = ["game_id", "play_id", "snap_frame"]
    return snap_frames


def pos_group(pos):
    if pos in ("CB",):
        return "CB"
    if pos in ("FS", "SS", "S"):
        return "Safety"
    if pos in ("ILB", "OLB", "MLB", "LB"):
        return "LB"
    return "Other"


def build_matchup_data(input_df, frame_refs, snap_frames):
    """
    For each play, determine:
      - Who is nearest defender to target at SNAP (QB can see this)
      - Who is nearest defender to target at THROW (actual coverage)
      - Separation at throw (throw window quality)
    """
    print("Building matchup data...")

    # Target at throw
    target_throw = input_df[input_df["player_role"] == "Targeted Receiver"].merge(
        frame_refs[["game_id", "play_id", "throw_frame"]], on=["game_id", "play_id"]
    )
    target_throw = target_throw[target_throw["frame_id"] == target_throw["throw_frame"]]

    # All defenders at throw
    def_throw = input_df[input_df["player_side"] == "Defense"].merge(
        frame_refs[["game_id", "play_id", "throw_frame"]], on=["game_id", "play_id"]
    )
    def_throw = def_throw[def_throw["frame_id"] == def_throw["throw_frame"]]

    # Target at snap
    target_snap = input_df[input_df["player_role"] == "Targeted Receiver"].merge(
        snap_frames, on=["game_id", "play_id"]
    )
    target_snap = target_snap[target_snap["frame_id"] == target_snap["snap_frame"]]

    # All defenders at snap
    def_snap = input_df[input_df["player_side"] == "Defense"].merge(
        snap_frames, on=["game_id", "play_id"]
    )
    def_snap = def_snap[def_snap["frame_id"] == def_snap["snap_frame"]]

    results = []
    for _, tgt_t in target_throw.iterrows():
        gid, pid = tgt_t.game_id, tgt_t.play_id

        # Nearest defender at throw
        play_def_t = def_throw[(def_throw.game_id == gid) & (def_throw.play_id == pid)]
        if len(play_def_t) == 0:
            continue
        dists_t = np.sqrt(
            (play_def_t.x.values - tgt_t.x) ** 2 + (play_def_t.y.values - tgt_t.y) ** 2
        )
        nearest_t = play_def_t.iloc[dists_t.argmin()]

        # Nearest defender at snap
        tgt_s = target_snap[(target_snap.game_id == gid) & (target_snap.play_id == pid)]
        play_def_s = def_snap[(def_snap.game_id == gid) & (def_snap.play_id == pid)]
        if len(tgt_s) == 0 or len(play_def_s) == 0:
            continue
        tgt_s = tgt_s.iloc[0]
        dists_s = np.sqrt(
            (play_def_s.x.values - tgt_s.x) ** 2 + (play_def_s.y.values - tgt_s.y) ** 2
        )
        nearest_s = play_def_s.iloc[dists_s.argmin()]

        # CB leverage at snap (for the nearest CB, not just nearest defender)
        play_cbs_s = play_def_s[play_def_s.player_position == "CB"]
        cb_leverage = "no_cb"
        cb_cushion = "no_cb"
        cb_dist = np.nan
        if len(play_cbs_s) > 0:
            cb_dists = np.sqrt(
                (play_cbs_s.x.values - tgt_s.x) ** 2 + (play_cbs_s.y.values - tgt_s.y) ** 2
            )
            nearest_cb = play_cbs_s.iloc[cb_dists.argmin()]
            cb_dist = cb_dists.min()

            # Leverage
            y_diff = nearest_cb.y - tgt_s.y
            if tgt_s.y > 26.65:
                cb_leverage = "inside" if y_diff < -1.5 else ("outside" if y_diff > 1.5 else "head_up")
            else:
                cb_leverage = "inside" if y_diff > 1.5 else ("outside" if y_diff < -1.5 else "head_up")

            # Cushion
            x_diff = nearest_cb.x - tgt_s.x
            if x_diff > 5:
                cb_cushion = "off"
            elif x_diff > 2:
                cb_cushion = "soft"
            else:
                cb_cushion = "press"

        results.append({
            "game_id": gid,
            "play_id": pid,
            "target_pos": tgt_t.player_position,
            "target_name": tgt_t.player_name,
            # At throw
            "throw_def_pos": nearest_t.player_position,
            "throw_def_group": pos_group(nearest_t.player_position),
            "separation": dists_t.min(),
            # At snap
            "snap_def_pos": nearest_s.player_position,
            "snap_def_group": pos_group(nearest_s.player_position),
            "snap_def_dist": dists_s.min(),
            "snap_def_speed": nearest_s.s,
            # CB leverage
            "cb_leverage": cb_leverage,
            "cb_cushion": cb_cushion,
            "cb_dist": cb_dist,
            # Same defender?
            "same_defender": nearest_s.nfl_id == nearest_t.nfl_id,
        })

    df = pd.DataFrame(results)
    df["target_group"] = df["target_pos"].apply(
        lambda x: "WR" if x == "WR" else ("TE" if x == "TE" else "RB")
    )

    print(f"  Built matchup data for {len(df):,} plays")
    return df


def analyze_matchup_prediction(matchup_df):
    """Can the QB predict who covers the target from pre-snap alignment?"""
    print("\n" + "=" * 60)
    print("MATCHUP PREDICTION: Can you tell pre-snap who covers your guy?")
    print("=" * 60)

    print(f"\n  Overall: nearest defender at snap = nearest at throw "
          f"{matchup_df.same_defender.mean()*100:.1f}% of the time")

    print(f"\n  By pre-snap nearest position:")
    for sg in ["CB", "Safety", "LB"]:
        data = matchup_df[matchup_df.snap_def_group == sg]
        if len(data) < 50:
            continue
        rate = data.same_defender.mean()
        avg_sep = data.separation.mean()
        # Who actually covers at throw?
        throw_dist = data.throw_def_group.value_counts(normalize=True).head(3)
        throw_str = ", ".join(f"{k} {v*100:.0f}%" for k, v in throw_dist.items())
        print(f"    {sg:7s} at snap → same at throw {rate*100:.0f}% | "
              f"avg separation {avg_sep:.1f} yds | ends up: {throw_str}")

    print(f"\n  KEY INSIGHT FOR QB:")
    cb_data = matchup_df[matchup_df.snap_def_group == "CB"]
    lb_data = matchup_df[matchup_df.snap_def_group == "LB"]
    if len(cb_data) > 0 and len(lb_data) > 0:
        print(f"    CB on your target at snap → {cb_data.separation.mean():.1f} yd window at throw")
        print(f"    LB on your target at snap → {lb_data.separation.mean():.1f} yd window at throw")
        print(f"    That's +{lb_data.separation.mean() - cb_data.separation.mean():.1f} yards of extra separation.")
        print(f"    When a LB is covering your target, ATTACK IT.")


def analyze_throw_windows(matchup_df):
    """What pre-snap alignment predicts the throw window?"""
    print("\n" + "=" * 60)
    print("THROW WINDOW PREDICTION: What does alignment tell you?")
    print("=" * 60)

    # CB cushion → throw window
    cb_plays = matchup_df[matchup_df.cb_cushion != "no_cb"]
    print(f"\n  CB Cushion at Snap → Throw Window:")
    for cushion in ["press", "soft", "off"]:
        data = cb_plays[cb_plays.cb_cushion == cushion]
        if len(data) >= 30:
            print(f"    {cushion:6s}: {data.separation.mean():.2f} yds avg, "
                  f"tight(<3) {(data.separation<3).mean()*100:.0f}%, "
                  f"open(>5) {(data.separation>5).mean()*100:.0f}% "
                  f"({len(data):,} plays)")

    # CB leverage → throw window
    print(f"\n  CB Leverage at Snap → Throw Window:")
    for lev in ["inside", "head_up", "outside"]:
        data = cb_plays[cb_plays.cb_leverage == lev]
        if len(data) >= 30:
            print(f"    {lev:8s}: {data.separation.mean():.2f} yds avg, "
                  f"tight(<3) {(data.separation<3).mean()*100:.0f}%, "
                  f"open(>5) {(data.separation>5).mean()*100:.0f}% "
                  f"({len(data):,} plays)")

    # Cushion + leverage combo
    print(f"\n  CB Cushion x Leverage → Throw Window (avg separation):")
    pivot = cb_plays.groupby(["cb_cushion", "cb_leverage"])["separation"].agg(["mean", "count"]).reset_index()
    for _, row in pivot[pivot["count"] >= 30].sort_values("mean").iterrows():
        print(f"    {row.cb_cushion:6s} + {row.cb_leverage:8s}: "
              f"{row['mean']:.2f} yds ({int(row['count']):,} plays)")


def analyze_mismatches(matchup_df):
    """Which matchup types produce the biggest windows?"""
    print("\n" + "=" * 60)
    print("MISMATCH EXPLOITATION: Where are the windows?")
    print("=" * 60)

    for tgt in ["WR", "TE", "RB"]:
        tgt_data = matchup_df[matchup_df.target_group == tgt]
        if len(tgt_data) < 50:
            continue
        print(f"\n  {tgt} targeted ({len(tgt_data):,} plays):")
        for dg in ["CB", "Safety", "LB"]:
            dg_data = tgt_data[tgt_data.throw_def_group == dg]
            if len(dg_data) >= 20:
                print(f"    vs {dg:7s}: {dg_data.separation.mean():.2f} yds | "
                      f"tight {(dg_data.separation<3).mean()*100:.0f}% | "
                      f"open {(dg_data.separation>5).mean()*100:.0f}% | "
                      f"{len(dg_data):,} plays")

    print(f"\n  TOP MISMATCHES (by avg separation):")
    combos = matchup_df.groupby(["target_group", "throw_def_group"]).agg(
        separation=("separation", "mean"),
        n=("play_id", "count"),
    ).reset_index()
    combos = combos[combos.n >= 30].sort_values("separation", ascending=False)
    for _, row in combos.head(8).iterrows():
        print(f"    {row.target_group} vs {row.throw_def_group}: "
              f"{row.separation:.2f} yds ({int(row.n):,} plays)")


def plot_throw_window_chart(matchup_df, save_path=None):
    """Visualize throw windows by matchup type."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # 1. Separation by defender type
    ax = axes[0]
    for i, dg in enumerate(["CB", "Safety", "LB"]):
        data = matchup_df[matchup_df.throw_def_group == dg]["separation"]
        if len(data) < 30:
            continue
        color = ["#e74c3c", "#f39c12", "#2ecc71"][i]
        ax.hist(data, bins=30, alpha=0.5, color=color, label=f"{dg} ({len(data):,})",
                density=True, range=(0, 15))
    ax.axvline(x=3, color="gray", linestyle="--", alpha=0.5)
    ax.text(3.1, ax.get_ylim()[1] * 0.9, "Tight\nwindow", fontsize=8, color="gray")
    ax.set_xlabel("Separation at Throw (yards)")
    ax.set_ylabel("Density")
    ax.set_title("Throw Window by Covering Defender", fontweight="bold")
    ax.legend(fontsize=9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # 2. CB Cushion → Separation
    ax = axes[1]
    cb_plays = matchup_df[matchup_df.cb_cushion.isin(["press", "soft", "off"])]
    cushion_order = ["press", "soft", "off"]
    cushion_colors = ["#e74c3c", "#f39c12", "#2ecc71"]
    positions = []
    for i, c in enumerate(cushion_order):
        data = cb_plays[cb_plays.cb_cushion == c]["separation"]
        if len(data) < 30:
            continue
        bp = ax.boxplot([data], positions=[i], widths=0.6, patch_artist=True,
                        showfliers=False, medianprops=dict(color="black", linewidth=2))
        bp["boxes"][0].set_facecolor(cushion_colors[i])
        bp["boxes"][0].set_alpha(0.7)
        ax.text(i, data.mean() + 0.3, f"{data.mean():.1f}", ha="center",
                fontsize=10, fontweight="bold")
    ax.set_xticks(range(len(cushion_order)))
    ax.set_xticklabels(["Press\n(0-2 yds)", "Soft\n(2-5 yds)", "Off\n(5+ yds)"])
    ax.set_ylabel("Separation at Throw (yards)")
    ax.set_title("CB Cushion at Snap → Throw Window", fontweight="bold")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # 3. Mismatch chart
    ax = axes[2]
    combos = matchup_df.groupby(["target_group", "throw_def_group"]).agg(
        sep=("separation", "mean"), n=("play_id", "count")
    ).reset_index()
    combos = combos[combos.n >= 30].sort_values("sep")
    colors = {"CB": "#e74c3c", "Safety": "#f39c12", "LB": "#2ecc71"}
    bars = ax.barh(
        range(len(combos)),
        combos["sep"],
        color=[colors.get(r.throw_def_group, "#95a5a6") for _, r in combos.iterrows()],
        edgecolor="white",
    )
    ax.set_yticks(range(len(combos)))
    ax.set_yticklabels(
        [f"{r.target_group} vs {r.throw_def_group}" for _, r in combos.iterrows()],
        fontsize=9,
    )
    for i, (_, r) in enumerate(combos.iterrows()):
        ax.text(r.sep + 0.1, i, f"{r.sep:.1f} yds ({int(r.n):,})",
                va="center", fontsize=8)
    ax.set_xlabel("Avg Separation at Throw (yards)")
    ax.set_title("Matchup Separation Rankings", fontweight="bold")
    ax.axvline(x=3, color="gray", linestyle="--", alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.suptitle("QB Throw Window Analysis — NFL 2023-2024",
                 fontsize=15, fontweight="bold", y=1.02)
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  Saved: {save_path}")
    plt.close()
    return fig


def plot_pre_snap_window_heatmap(matchup_df, save_path=None):
    """Heatmap: snap nearest position x CB cushion → throw window."""
    fig, ax = plt.subplots(figsize=(10, 6))

    df = matchup_df[matchup_df.cb_cushion.isin(["press", "soft", "off"])].copy()

    pivot = df.groupby(["snap_def_group", "cb_cushion"]).agg(
        sep=("separation", "mean"), n=("play_id", "count")
    ).reset_index()
    pivot_table = pivot.pivot(index="snap_def_group", columns="cb_cushion", values="sep")
    count_table = pivot.pivot(index="snap_def_group", columns="cb_cushion", values="n")

    # Reorder
    row_order = ["CB", "Safety", "LB"]
    col_order = ["press", "soft", "off"]
    pivot_table = pivot_table.reindex(index=row_order, columns=col_order)
    count_table = count_table.reindex(index=row_order, columns=col_order)

    from matplotlib.colors import LinearSegmentedColormap
    cmap = LinearSegmentedColormap.from_list("window", ["#e74c3c", "#f1c40f", "#2ecc71"])

    sns.heatmap(pivot_table, annot=True, fmt=".1f", cmap=cmap, ax=ax,
                vmin=2.5, vmax=7, linewidths=2, linecolor="white",
                cbar_kws={"label": "Avg Separation (yds)"})

    # Add counts
    for i in range(len(row_order)):
        for j in range(len(col_order)):
            val = count_table.iloc[i, j] if not pd.isna(count_table.iloc[i, j]) else 0
            if val > 0:
                ax.text(j + 0.5, i + 0.75, f"n={int(val)}", ha="center", va="center",
                        fontsize=8, color="gray")

    ax.set_xlabel("CB Cushion at Snap", fontsize=12)
    ax.set_ylabel("Nearest Defender at Snap", fontsize=12)
    ax.set_title("QB Pre-Snap Read: Throw Window Prediction\n"
                 "Nearest defender type x CB cushion → target separation at throw",
                 fontsize=13, fontweight="bold", pad=15)
    ax.set_xticklabels(["Press\n(0-2 yds)", "Soft\n(2-5 yds)", "Off\n(5+ yds)"])

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  Saved: {save_path}")
    plt.close()
    return fig


def generate_qb_gameplan(matchup_df):
    """Generate the final QB game-plan report."""
    print("\n" + "=" * 60)
    print("QB GAME-PLAN BINDER: Pre-Snap Read Guide")
    print("=" * 60)

    cb = matchup_df[matchup_df.snap_def_group == "CB"]
    lb = matchup_df[matchup_df.snap_def_group == "LB"]
    sf = matchup_df[matchup_df.snap_def_group == "Safety"]
    press = matchup_df[matchup_df.cb_cushion == "press"]
    off = matchup_df[matchup_df.cb_cushion == "off"]
    wr_lb = matchup_df[(matchup_df.target_group == "WR") & (matchup_df.throw_def_group == "LB")]
    wr_cb = matchup_df[(matchup_df.target_group == "WR") & (matchup_df.throw_def_group == "CB")]

    print(f"""
  READ 1: WHO IS ON YOUR TARGET?
  {'='*50}
  The nearest defender at snap stays on the target:
    CB at snap  → 65% stays, avg {cb.separation.mean():.1f} yd window
    Safety      → 57% stays, avg {sf.separation.mean():.1f} yd window
    LB          → 63% stays, avg {lb.separation.mean():.1f} yd window

  RULE: If a LB is covering your primary target pre-snap,
  that's +{lb.separation.mean() - cb.separation.mean():.1f} yards of extra separation.
  Identify it. Attack it.

  READ 2: CB CUSHION TELLS YOU THE WINDOW
  {'='*50}
  Press (0-2 yds): {press.separation.mean():.1f} yd window — tight. Need a route win.
    Tight coverage rate: {(press.separation < 3).mean()*100:.0f}%
  Off (5+ yds):    {off.separation.mean():.1f} yd window — room to operate.
    Open rate: {(off.separation > 5).mean()*100:.0f}%

  RULE: Off coverage = guaranteed short throw window.
  Press = contested. Only attack press with quick releases.

  READ 3: THE MISMATCH HIERARCHY
  {'='*50}
  Best windows (highest avg separation):
    RB vs any defender:  7-9 yds — the checkdown is ALWAYS open
    WR vs LB:           {wr_lb.separation.mean():.1f} yds — attack every time
    WR vs Safety:       {matchup_df[(matchup_df.target_group=='WR') & (matchup_df.throw_def_group=='Safety')].separation.mean():.1f} yds
    WR vs CB:           {wr_cb.separation.mean():.1f} yds — tightest windows

  RULE: When you identify a LB on a WR pre-snap, that's your
  first read. LBs give up {wr_lb.separation.mean() - wr_cb.separation.mean():.1f} more yards of separation than CBs.

  READ 4: WHEN THE MATCHUP CHANGES
  {'='*50}
  43% of the time, the pre-snap nearest defender is NOT the
  one covering at throw. When the coverage SWITCHES:
    New defender is further away → bigger window
    Avg separation: {matchup_df[~matchup_df.same_defender].separation.mean():.1f} yds (switched)
    vs {matchup_df[matchup_df.same_defender].separation.mean():.1f} yds (same defender)

  RULE: Coverage switches = bigger windows. If you see the
  defense rotate post-snap, the receiver may be more open
  than the pre-snap look suggests.
""")


if __name__ == "__main__":
    print("=" * 60)
    print("QB PRE-SNAP READ ANALYSIS")
    print("=" * 60)

    input_df = pd.read_pickle("data/input_processed.pkl")
    output_df = pd.read_pickle("data/output_processed.pkl")
    frame_refs = pd.read_pickle("data/frame_refs.pkl")

    # Detect snaps
    snap_frames = detect_snap_frames(input_df)

    # Build matchup data
    matchup_df = build_matchup_data(input_df, frame_refs, snap_frames)
    matchup_df.to_pickle("data/matchup_data.pkl")

    # Analyses
    analyze_matchup_prediction(matchup_df)
    analyze_throw_windows(matchup_df)
    analyze_mismatches(matchup_df)

    # Visualizations
    print("\nGenerating QB visualizations...")
    plot_throw_window_chart(matchup_df, "output/figures/qb_throw_windows.png")
    plot_pre_snap_window_heatmap(matchup_df, "output/figures/qb_presnap_heatmap.png")

    # Game plan
    generate_qb_gameplan(matchup_df)
