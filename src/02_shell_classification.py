"""
Step 2: Pre-Throw Shell Classification
=======================================
Classify defensive shells (1-high, 2-high, etc.) at key moments:
  - At the throw release (last pre-throw frame)
  - At an earlier reference point (~1.5s before throw)

Shell classification uses safety positioning geometry:
  - 2-High: Two safeties aligned deep (both > 8 yards from LOS)
  - 1-High: One safety deep, one rolled down
  - 0-High: No deep safeties (heavy box / blitz shell)

This is the foundation for disguise detection.
"""

import pandas as pd
import numpy as np


DEEP_SAFETY_DEPTH_THRESHOLD = 8.0  # yards from LOS to be considered "deep"
SAFETY_SPLIT_WIDTH = 5.0  # yards from center to distinguish split vs MOF


def classify_shell_at_frame(input_df, game_id, play_id, frame_id):
    """
    Classify defensive shell for a single play at a specific frame.

    Returns dict with shell type and safety positioning details.
    """
    play_frame = input_df[
        (input_df["game_id"] == game_id)
        & (input_df["play_id"] == play_id)
        & (input_df["frame_id"] == frame_id)
    ]

    safeties = play_frame[play_frame["is_safety"]]

    if len(safeties) == 0:
        return {
            "shell": "unknown",
            "n_deep_safeties": 0,
            "safety_depth_mean": np.nan,
            "safety_width_spread": np.nan,
            "safety_mof_distance": np.nan,
        }

    # Depth from LOS (positive = deeper into defense)
    deep_safeties = safeties[safeties["depth_from_los"] > DEEP_SAFETY_DEPTH_THRESHOLD]
    n_deep = len(deep_safeties)

    # Shell classification
    if n_deep >= 2:
        shell = "2-high"
    elif n_deep == 1:
        shell = "1-high"
    else:
        shell = "0-high"

    # Safety geometry metrics
    safety_depth_mean = safeties["depth_from_los"].mean()
    safety_width_spread = safeties["width_from_center"].max() - safeties["width_from_center"].min()
    # Closest safety to middle of field
    safety_mof_distance = safeties["width_from_center"].abs().min()

    return {
        "shell": shell,
        "n_deep_safeties": n_deep,
        "safety_depth_mean": safety_depth_mean,
        "safety_width_spread": safety_width_spread,
        "safety_mof_distance": safety_mof_distance,
    }


def classify_shells_bulk(input_df, frame_refs):
    """
    Classify shells for all plays at both the throw frame and early frame.
    Vectorized for performance.
    """
    print("Classifying defensive shells (vectorized)...")

    # Get safety data only
    safety_df = input_df[input_df["is_safety"]].copy()

    # --- Shell at THROW frame (last pre-throw frame) ---
    throw_safety = safety_df.merge(
        frame_refs[["game_id", "play_id", "throw_frame"]],
        on=["game_id", "play_id"],
    )
    throw_safety = throw_safety[throw_safety["frame_id"] == throw_safety["throw_frame"]]

    throw_shells = _compute_shell_features(throw_safety, prefix="throw")

    # --- Shell at EARLY frame (~1.5s before throw) ---
    early_safety = safety_df.merge(
        frame_refs[["game_id", "play_id", "early_frame"]],
        on=["game_id", "play_id"],
    )
    early_safety = early_safety[early_safety["frame_id"] == early_safety["early_frame"]]

    early_shells = _compute_shell_features(early_safety, prefix="early")

    # Merge both
    shells = throw_shells.merge(early_shells, on=["game_id", "play_id"], how="outer")

    # Detect shell change (rotation indicator)
    shells["shell_changed"] = shells["throw_shell"] != shells["early_shell"]

    # Specific rotation patterns
    shells["two_to_one"] = (
        (shells["early_shell"] == "2-high") & (shells["throw_shell"] == "1-high")
    )
    shells["one_to_two"] = (
        (shells["early_shell"] == "1-high") & (shells["throw_shell"] == "2-high")
    )

    print(f"Shell classification complete for {len(shells):,} plays")
    print(f"  Shell distribution at throw:")
    print(f"    {shells.throw_shell.value_counts().to_dict()}")
    print(f"  Shell changed between early and throw: "
          f"{shells.shell_changed.sum():,} / {len(shells):,} "
          f"({shells.shell_changed.mean()*100:.1f}%)")

    return shells


def _compute_shell_features(safety_frame_df, prefix):
    """Compute shell classification features from safety positions at a given frame."""
    grouped = safety_frame_df.groupby(["game_id", "play_id"])

    result = grouped.agg(
        n_safeties=("nfl_id", "count"),
        safety_depth_mean=("depth_from_los", "mean"),
        safety_depth_max=("depth_from_los", "max"),
        safety_depth_min=("depth_from_los", "min"),
        safety_width_spread=("width_from_center", lambda x: x.max() - x.min()),
        safety_mof_distance=("width_from_center", lambda x: x.abs().min()),
        safety_speed_mean=("s", "mean"),
        safety_speed_max=("s", "max"),
        safety_accel_mean=("a", "mean"),
    ).reset_index()

    # Count deep safeties
    deep_counts = (
        safety_frame_df[safety_frame_df["depth_from_los"] > DEEP_SAFETY_DEPTH_THRESHOLD]
        .groupby(["game_id", "play_id"])["nfl_id"]
        .count()
        .reset_index()
        .rename(columns={"nfl_id": "n_deep_safeties"})
    )
    result = result.merge(deep_counts, on=["game_id", "play_id"], how="left")
    result["n_deep_safeties"] = result["n_deep_safeties"].fillna(0).astype(int)

    # Shell type
    result["shell"] = "0-high"
    result.loc[result["n_deep_safeties"] == 1, "shell"] = "1-high"
    result.loc[result["n_deep_safeties"] >= 2, "shell"] = "2-high"

    # Rename all columns with prefix
    rename_map = {
        col: f"{prefix}_{col}"
        for col in result.columns
        if col not in ["game_id", "play_id"]
    }
    result = result.rename(columns=rename_map)

    return result


if __name__ == "__main__":
    input_df = pd.read_pickle("data/input_processed.pkl")
    frame_refs = pd.read_pickle("data/frame_refs.pkl")
    shells = classify_shells_bulk(input_df, frame_refs)
    shells.to_pickle("data/shells.pkl")
    print("\nSample shell data:")
    print(shells.head())
