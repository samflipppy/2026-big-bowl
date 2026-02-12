"""
Step 5: Disguise Score Metric & Rankings
=========================================
Convert model outputs into coach-usable metrics.

Key Metrics:
  - Disguise Rate: % of snaps where defense showed one shell but executed another
  - Disguise Score: Model-predicted probability of disguise (continuous 0-1)
  - Shell Deception Index: Weighted metric combining pre-throw rotation + post-throw mismatch
  - Pursuit Efficiency: How well defenders close on the ball after disguise

Coaches LOVE rankings. We rank teams and individual safeties.
"""

import pandas as pd
import numpy as np


def compute_team_metrics(input_df, model_df, labels, shells):
    """
    Compute team-level disguise metrics.

    We need to map plays to defensive teams. Since we don't have team IDs directly,
    we infer them from the game matchups and defensive side.
    """
    print("Computing team-level disguise metrics...")

    # Get unique games and their teams from player names
    # We'll use the game_id to group plays and identify which defense was on field
    # For now, use game_id as the grouping unit

    # Get all defensive players per play
    defense_plays = (
        input_df[input_df["player_side"] == "Defense"]
        .drop_duplicates(["game_id", "play_id", "nfl_id"])[
            ["game_id", "play_id", "nfl_id", "player_name", "player_position"]
        ]
    )

    # Merge model predictions with labels
    play_metrics = model_df[["game_id", "play_id", "disguise_prob"]].merge(
        labels[["game_id", "play_id", "disguised", "shell_changed",
                "any_drove_forward", "any_bailed_deep", "any_hash_cross"]],
        on=["game_id", "play_id"],
        how="left",
    )

    # Merge shell info
    play_metrics = play_metrics.merge(
        shells[["game_id", "play_id", "throw_shell", "early_shell"]],
        on=["game_id", "play_id"],
        how="left",
    )

    # --- Game-level metrics (proxy for team) ---
    game_metrics = play_metrics.groupby("game_id").agg(
        n_plays=("play_id", "count"),
        disguise_rate=("disguised", "mean"),
        mean_disguise_prob=("disguise_prob", "mean"),
        shell_change_rate=("shell_changed", "mean"),
        hash_cross_rate=("any_hash_cross", "mean"),
        pct_2high=("throw_shell", lambda x: (x == "2-high").mean()),
        pct_1high=("throw_shell", lambda x: (x == "1-high").mean()),
    ).reset_index()

    game_metrics["disguise_score"] = (
        game_metrics["disguise_rate"] * 0.4
        + game_metrics["mean_disguise_prob"] * 0.3
        + game_metrics["shell_change_rate"] * 0.2
        + game_metrics["hash_cross_rate"] * 0.1
    ) * 100  # Scale to 0-100

    game_metrics = game_metrics.sort_values("disguise_score", ascending=False)

    print(f"\n  Game-Level Disguise Rankings (top 15):")
    print(f"  {'Game':>12s}  {'Plays':>5s}  {'Disg%':>6s}  {'Score':>6s}  "
          f"{'Shell%':>6s}  {'2-High%':>7s}")
    print("  " + "-" * 55)
    for _, row in game_metrics.head(15).iterrows():
        print(f"  {int(row.game_id):>12d}  {int(row.n_plays):>5d}  "
              f"{row.disguise_rate*100:>5.1f}%  {row.disguise_score:>5.1f}  "
              f"{row.shell_change_rate*100:>5.1f}%  {row.pct_2high*100:>6.1f}%")

    return play_metrics, game_metrics


def compute_safety_metrics(input_df, output_df, frame_refs, labels):
    """
    Compute individual safety-level disguise metrics.

    Which safeties tip coverage? Which are masters of deception?
    """
    print("\nComputing safety-level metrics...")

    safety_df = input_df[input_df["is_safety"]].copy()

    # Safety position at throw frame
    safety_throw = safety_df.merge(
        frame_refs[["game_id", "play_id", "throw_frame"]], on=["game_id", "play_id"]
    )
    safety_throw = safety_throw[safety_throw["frame_id"] == safety_throw["throw_frame"]]

    # Safety position at early frame
    safety_early = safety_df.merge(
        frame_refs[["game_id", "play_id", "early_frame"]], on=["game_id", "play_id"]
    )
    safety_early = safety_early[safety_early["frame_id"] == safety_early["early_frame"]]

    # Per-safety pre-throw movement
    early_pos = safety_early.set_index(
        ["game_id", "play_id", "nfl_id"]
    )[["x", "y", "s", "dir"]].rename(columns=lambda c: f"early_{c}")

    throw_pos = safety_throw.set_index(
        ["game_id", "play_id", "nfl_id"]
    )[["x", "y", "s", "dir", "depth_from_los", "width_from_center"]].rename(
        columns=lambda c: f"throw_{c}"
    )

    safety_movement = early_pos.join(throw_pos, how="inner").reset_index()

    # Compute per-safety metrics
    safety_movement["pre_throw_displacement"] = np.sqrt(
        (safety_movement["throw_x"] - safety_movement["early_x"])**2
        + (safety_movement["throw_y"] - safety_movement["early_y"])**2
    )
    dir_diff = safety_movement["throw_dir"] - safety_movement["early_dir"]
    safety_movement["dir_change"] = ((dir_diff + 180) % 360 - 180).abs()
    safety_movement["speed_change"] = safety_movement["throw_s"] - safety_movement["early_s"]

    # Join with play-level disguise label
    safety_movement = safety_movement.merge(
        labels[["game_id", "play_id", "disguised"]],
        on=["game_id", "play_id"],
        how="left",
    )

    # Get player names
    player_names = (
        input_df[input_df["is_safety"]]
        .drop_duplicates("nfl_id")[["nfl_id", "player_name", "player_position"]]
    )

    # Aggregate per safety
    safety_stats = safety_movement.groupby("nfl_id").agg(
        n_plays=("play_id", "count"),
        plays_disguised=("disguised", "sum"),
        disguise_rate=("disguised", "mean"),
        avg_pre_displacement=("pre_throw_displacement", "mean"),
        avg_dir_change=("dir_change", "mean"),
        avg_speed_at_throw=("throw_s", "mean"),
        avg_depth=("throw_depth_from_los", "mean"),
    ).reset_index()

    safety_stats = safety_stats.merge(player_names, on="nfl_id", how="left")

    # Deception score per safety
    safety_stats["deception_score"] = (
        safety_stats["avg_pre_displacement"] * 0.3
        + safety_stats["avg_dir_change"] / 180 * 0.3  # Normalize to 0-1
        + safety_stats["disguise_rate"] * 0.4
    ) * 100

    # Filter to safeties with enough snaps
    min_snaps = 50
    qualified = safety_stats[safety_stats["n_plays"] >= min_snaps].sort_values(
        "deception_score", ascending=False
    )

    print(f"\n  Top 20 Safeties by Deception Score (min {min_snaps} snaps):")
    print(f"  {'Player':25s} {'Pos':4s} {'Snaps':>5s} {'Disg%':>6s} "
          f"{'AvgDisp':>7s} {'DirChg':>6s} {'Score':>6s}")
    print("  " + "-" * 65)
    for _, row in qualified.head(20).iterrows():
        print(f"  {str(row.player_name):25s} {str(row.player_position):4s} "
              f"{int(row.n_plays):>5d} {row.disguise_rate*100:>5.1f}% "
              f"{row.avg_pre_displacement:>6.2f}  {row.avg_dir_change:>5.1f}  "
              f"{row.deception_score:>5.1f}")

    return safety_stats, qualified


def compute_shell_effectiveness(labels, shells):
    """
    Does disguise actually work? Compare outcomes when disguised vs not.
    """
    print("\nAnalyzing disguise effectiveness...")

    merged = labels.merge(
        shells[["game_id", "play_id", "throw_shell", "early_shell"]],
        on=["game_id", "play_id"],
        how="left",
        suffixes=("", "_shell"),
    )

    # Use the shell column (handle possible suffix from merge)
    shell_col = "throw_shell" if "throw_shell" in merged.columns else "throw_shell_shell"

    # Disguise rates by shell type
    print(f"\n  Disguise rate by pre-throw shell:")
    for shell in ["2-high", "1-high", "0-high"]:
        mask = merged[shell_col] == shell
        if mask.sum() > 0:
            rate = merged.loc[mask, "disguised"].mean()
            n = mask.sum()
            print(f"    {shell}: {rate*100:.1f}% disguised ({n:,} plays)")

    # Disguise breakdown
    print(f"\n  Disguise mechanism breakdown:")
    print(f"    Shell rotation (pre-throw): {labels.shell_changed.sum():,} plays")
    print(f"    Safety drove forward: {labels.any_drove_forward.sum():,} plays")
    print(f"    Safety bailed deep: {labels.any_bailed_deep.sum():,} plays")
    print(f"    Hash crossing: {labels.any_hash_cross.sum():,} plays")

    return merged


if __name__ == "__main__":
    print("=" * 60)
    print("DISGUISE METRICS & RANKINGS")
    print("=" * 60)

    input_df = pd.read_pickle("data/input_processed.pkl")
    output_df = pd.read_pickle("data/output_processed.pkl")
    frame_refs = pd.read_pickle("data/frame_refs.pkl")
    shells = pd.read_pickle("data/shells.pkl")
    labels = pd.read_pickle("data/disguise_labels.pkl")
    model_df = pd.read_pickle("data/model_predictions.pkl")

    # Team metrics
    play_metrics, game_metrics = compute_team_metrics(input_df, model_df, labels, shells)
    play_metrics.to_pickle("data/play_metrics.pkl")
    game_metrics.to_pickle("data/game_metrics.pkl")

    # Safety metrics
    safety_stats, qualified_safeties = compute_safety_metrics(
        input_df, output_df, frame_refs, labels
    )
    safety_stats.to_pickle("data/safety_stats.pkl")

    # Effectiveness
    effectiveness = compute_shell_effectiveness(labels, shells)

    print("\nMetrics and rankings complete.")
