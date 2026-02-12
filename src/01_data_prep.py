"""
Step 1: Data Preparation
========================
Load tracking data, standardize coordinates, identify key defensive players,
and merge pre-throw + ball-in-air frames.
"""

import pickle
import pandas as pd
import numpy as np


def load_data(input_path, output_path):
    """Load pre-throw (input) and ball-in-air (output) tracking data."""
    with open(input_path, "rb") as f:
        input_df = pickle.load(f)
    with open(output_path, "rb") as f:
        output_df = pickle.load(f)
    return input_df, output_df


def standardize_direction(df):
    """
    Standardize all plays so offense moves left-to-right.
    This makes spatial features comparable across plays.
    """
    df = df.copy()
    left_mask = df["play_direction"] == "left"

    # Flip x coordinate (field is 120 yards)
    df.loc[left_mask, "x"] = 120.0 - df.loc[left_mask, "x"]
    # Flip y coordinate (field is 53.3 yards wide)
    df.loc[left_mask, "y"] = 53.3 - df.loc[left_mask, "y"]
    # Flip direction (0-360 degrees)
    if "dir" in df.columns:
        df.loc[left_mask, "dir"] = (360.0 - df.loc[left_mask, "dir"]) % 360.0
    # Flip orientation
    if "o" in df.columns:
        df.loc[left_mask, "o"] = (360.0 - df.loc[left_mask, "o"]) % 360.0
    # Flip ball landing coordinates
    if "ball_land_x" in df.columns:
        df.loc[left_mask, "ball_land_x"] = 120.0 - df.loc[left_mask, "ball_land_x"]
    if "ball_land_y" in df.columns:
        df.loc[left_mask, "ball_land_y"] = 53.3 - df.loc[left_mask, "ball_land_y"]

    return df


def compute_los_and_relative_positions(df):
    """
    Compute line of scrimmage (LOS) and player positions relative to it.
    Depth = yards behind LOS (positive = deeper in own territory for defense).
    Width = yards from center of field (26.65).
    """
    df = df.copy()

    # LOS is the absolute yardline number, standardized
    # After standardization, offense goes left->right, so LOS is the snap x
    # Use the QB position at frame 1 as a proxy for LOS
    qb_los = (
        df[(df["player_role"] == "Passer") & (df["frame_id"] == 1)]
        .groupby(["game_id", "play_id"])["x"]
        .first()
        .reset_index()
        .rename(columns={"x": "los_x"})
    )
    df = df.merge(qb_los, on=["game_id", "play_id"], how="left")

    # Depth relative to LOS (positive = deeper into defensive territory)
    # Defense is on the right side (higher x), offense on left
    df["depth_from_los"] = df["x"] - df["los_x"]

    # Width from center of field
    df["width_from_center"] = df["y"] - 26.65

    return df


def identify_safeties(df):
    """
    Tag safeties in the data. Safeties are the key rotation players.
    Positions: FS, SS, S. Also tag nickel/slot CBs by depth.
    """
    df = df.copy()
    safety_positions = {"FS", "SS", "S"}
    df["is_safety"] = df["player_position"].isin(safety_positions)
    df["is_cb"] = df["player_position"] == "CB"
    df["is_lb"] = df["player_position"].isin({"ILB", "OLB", "MLB", "LB"})
    df["is_dl"] = df["player_position"].isin({"DE", "DT", "NT"})
    df["is_defensive_back"] = df["is_safety"] | df["is_cb"]
    return df


def get_last_pre_throw_frame(df):
    """Get the last frame of pre-throw data for each play (throw release moment)."""
    last_frames = (
        df.groupby(["game_id", "play_id"])["frame_id"]
        .max()
        .reset_index()
        .rename(columns={"frame_id": "throw_frame"})
    )
    return last_frames


def get_early_frame(df, seconds_before_throw=1.5):
    """
    Get frame ~1.5 seconds before throw for delta calculations.
    At 10fps, 1.5 seconds = 15 frames before the last frame.
    """
    last_frames = get_last_pre_throw_frame(df)
    last_frames["early_frame"] = (
        last_frames["throw_frame"] - int(seconds_before_throw * 10)
    ).clip(lower=1)
    return last_frames


def prepare_data():
    """Main data preparation pipeline."""
    print("Loading data...")
    input_df, output_df = load_data(
        "data/serialized_data/input_data.pkl", "data/serialized_data/output_data.pkl"
    )

    print(f"Input: {input_df.shape[0]:,} rows, {input_df.game_id.nunique()} games, "
          f"{input_df[['game_id','play_id']].drop_duplicates().shape[0]:,} plays")
    print(f"Output: {output_df.shape[0]:,} rows")

    print("Standardizing play direction...")
    input_df = standardize_direction(input_df)

    # For output data, we need to join play_direction from input
    play_dirs = (
        input_df[["game_id", "play_id", "play_direction"]]
        .drop_duplicates(["game_id", "play_id"])
    )
    output_df = output_df.merge(play_dirs, on=["game_id", "play_id"], how="left")
    left_mask = output_df["play_direction"] == "left"
    output_df.loc[left_mask, "x"] = 120.0 - output_df.loc[left_mask, "x"]
    output_df.loc[left_mask, "y"] = 53.3 - output_df.loc[left_mask, "y"]

    print("Computing relative positions...")
    input_df = compute_los_and_relative_positions(input_df)

    print("Identifying safeties and DBs...")
    input_df = identify_safeties(input_df)

    print("Computing frame references...")
    frame_refs = get_early_frame(input_df)

    # Save processed data
    input_df.to_pickle("data/input_processed.pkl")
    output_df.to_pickle("data/output_processed.pkl")
    frame_refs.to_pickle("data/frame_refs.pkl")

    print("Data preparation complete.")
    print(f"  Safeties in data: {input_df[input_df.is_safety].nfl_id.nunique()}")
    print(f"  CBs in data: {input_df[input_df.is_cb].nfl_id.nunique()}")

    return input_df, output_df, frame_refs


if __name__ == "__main__":
    prepare_data()
