"""
Step 3: Feature Engineering
============================
This is where the project wins.

Three categories of features:
  1. PRE-THROW STRUCTURE - defensive alignment at throw release
  2. DELTA FEATURES - movement changes in the 1.5s window before throw
  3. POST-THROW PURSUIT - how defenders move once the ball is in the air

Focus on safeties first, then all defensive coverage players.
"""

import pandas as pd
import numpy as np
from scipy.spatial.distance import cdist


def compute_pre_throw_features(input_df, frame_refs):
    """
    Compute features for each defensive player at the throw release frame.
    These capture the defensive structure at the moment of commitment.
    """
    print("Computing pre-throw features...")

    # Get defensive coverage players at throw frame
    defense = input_df[input_df["player_side"] == "Defense"].copy()
    throw_data = defense.merge(
        frame_refs[["game_id", "play_id", "throw_frame"]],
        on=["game_id", "play_id"],
    )
    throw_data = throw_data[throw_data["frame_id"] == throw_data["throw_frame"]]

    # --- Per-player features at throw ---
    player_features = throw_data[[
        "game_id", "play_id", "nfl_id", "player_position",
        "is_safety", "is_cb", "is_lb",
        "x", "y", "s", "a", "dir", "o",
        "depth_from_los", "width_from_center",
        "ball_land_x", "ball_land_y",
    ]].copy()

    # Movement direction components
    player_features["dir_rad"] = np.radians(player_features["dir"])
    player_features["vel_x"] = player_features["s"] * np.cos(player_features["dir_rad"])
    player_features["vel_y"] = player_features["s"] * np.sin(player_features["dir_rad"])

    # Orientation components (where player is facing)
    player_features["o_rad"] = np.radians(player_features["o"])

    # Angle to ball landing point
    dx_ball = player_features["ball_land_x"] - player_features["x"]
    dy_ball = player_features["ball_land_y"] - player_features["y"]
    player_features["angle_to_ball"] = np.degrees(np.arctan2(dy_ball, dx_ball)) % 360
    player_features["dist_to_ball_land"] = np.sqrt(dx_ball**2 + dy_ball**2)

    # Angle between movement direction and ball direction
    # This reveals if the player is already heading toward the ball
    angle_diff = player_features["dir"] - player_features["angle_to_ball"]
    player_features["dir_ball_alignment"] = np.abs(
        (angle_diff + 180) % 360 - 180
    )  # 0 = heading straight at ball, 180 = away

    # Angle between orientation and ball direction
    # This reveals if the player is LOOKING at the ball
    o_diff = player_features["o"] - player_features["angle_to_ball"]
    player_features["orient_ball_alignment"] = np.abs(
        (o_diff + 180) % 360 - 180
    )

    # Angle toward middle of field (MOF)
    # Safeties tipping rotation often lean toward MOF early
    player_features["angle_to_mof"] = np.degrees(
        np.arctan2(26.65 - player_features["y"], 0)
    ) % 360

    return player_features


def compute_delta_features(input_df, frame_refs):
    """
    Compute movement changes between early reference frame and throw frame.
    These SCREAM disguise - a safety who changes heading late is rotating.
    """
    print("Computing delta features (early -> throw)...")

    defense = input_df[input_df["player_side"] == "Defense"].copy()

    # Get data at early frame
    early_data = defense.merge(
        frame_refs[["game_id", "play_id", "early_frame"]],
        on=["game_id", "play_id"],
    )
    early_data = early_data[early_data["frame_id"] == early_data["early_frame"]]
    early_data = early_data.set_index(["game_id", "play_id", "nfl_id"])

    # Get data at throw frame
    throw_data = defense.merge(
        frame_refs[["game_id", "play_id", "throw_frame"]],
        on=["game_id", "play_id"],
    )
    throw_data = throw_data[throw_data["frame_id"] == throw_data["throw_frame"]]
    throw_data = throw_data.set_index(["game_id", "play_id", "nfl_id"])

    # Compute deltas for players present in both frames
    common_idx = early_data.index.intersection(throw_data.index)
    early = early_data.loc[common_idx]
    throw = throw_data.loc[common_idx]

    deltas = pd.DataFrame(index=common_idx)

    # Position changes
    deltas["delta_x"] = throw["x"] - early["x"]
    deltas["delta_y"] = throw["y"] - early["y"]
    deltas["delta_depth"] = throw["depth_from_los"] - early["depth_from_los"]
    deltas["delta_width"] = throw["width_from_center"] - early["width_from_center"]
    deltas["total_displacement"] = np.sqrt(deltas["delta_x"]**2 + deltas["delta_y"]**2)

    # Speed/acceleration changes
    deltas["delta_speed"] = throw["s"] - early["s"]
    deltas["delta_accel"] = throw["a"] - early["a"]
    deltas["accel_spike"] = (deltas["delta_accel"] > 1.0).astype(int)

    # Direction changes (THIS IS KEY FOR ROTATION DETECTION)
    dir_change = throw["dir"] - early["dir"]
    deltas["delta_dir"] = ((dir_change + 180) % 360 - 180).abs()  # 0-180 scale
    deltas["major_dir_change"] = (deltas["delta_dir"] > 45).astype(int)

    # Orientation changes (hip flip / head turn)
    o_change = throw["o"] - early["o"]
    deltas["delta_orientation"] = ((o_change + 180) % 360 - 180).abs()

    # Hash drift - lateral movement toward/away from ball side
    deltas["hash_drift"] = deltas["delta_y"].abs()

    # Movement toward MOF (middle of field)
    # Positive = moved toward center
    deltas["mof_drift"] = early["width_from_center"].abs() - throw["width_from_center"].abs()

    deltas = deltas.reset_index()
    return deltas


def compute_post_throw_features(input_df, output_df, frame_refs):
    """
    Compute features from ball-in-air tracking data.
    Only available for player_to_predict=True players.
    """
    print("Computing post-throw (ball-in-air) features...")

    # Get player info from input data
    player_info = (
        input_df[input_df["player_side"] == "Defense"]
        .drop_duplicates(["game_id", "play_id", "nfl_id"])[
            ["game_id", "play_id", "nfl_id", "player_position",
             "is_safety", "is_cb", "is_lb", "ball_land_x", "ball_land_y"]
        ]
    )

    # Join with output data
    flight = output_df.merge(player_info, on=["game_id", "play_id", "nfl_id"], how="inner")

    if len(flight) == 0:
        print("  No matching defensive players in output data.")
        return pd.DataFrame()

    # Get first and last ball-in-air frame per player per play
    flight_agg = flight.groupby(["game_id", "play_id", "nfl_id"]).agg(
        flight_x_start=("x", "first"),
        flight_y_start=("y", "first"),
        flight_x_end=("x", "last"),
        flight_y_end=("y", "last"),
        n_flight_frames=("frame_id", "count"),
    ).reset_index()

    flight_agg = flight_agg.merge(
        player_info, on=["game_id", "play_id", "nfl_id"], how="left"
    )

    # Total displacement during flight
    flight_agg["flight_displacement"] = np.sqrt(
        (flight_agg["flight_x_end"] - flight_agg["flight_x_start"])**2
        + (flight_agg["flight_y_end"] - flight_agg["flight_y_start"])**2
    )

    # Movement direction during flight
    dx = flight_agg["flight_x_end"] - flight_agg["flight_x_start"]
    dy = flight_agg["flight_y_end"] - flight_agg["flight_y_start"]
    flight_agg["flight_direction"] = np.degrees(np.arctan2(dy, dx)) % 360

    # Distance to ball landing point at start vs end of flight
    flight_agg["dist_to_ball_start"] = np.sqrt(
        (flight_agg["ball_land_x"] - flight_agg["flight_x_start"])**2
        + (flight_agg["ball_land_y"] - flight_agg["flight_y_start"])**2
    )
    flight_agg["dist_to_ball_end"] = np.sqrt(
        (flight_agg["ball_land_x"] - flight_agg["flight_x_end"])**2
        + (flight_agg["ball_land_y"] - flight_agg["flight_y_end"])**2
    )

    # Closing rate - did the player get closer to the ball?
    flight_agg["ball_closing_distance"] = (
        flight_agg["dist_to_ball_start"] - flight_agg["dist_to_ball_end"]
    )
    flight_agg["pursuit_efficiency"] = (
        flight_agg["ball_closing_distance"] / flight_agg["flight_displacement"].clip(lower=0.1)
    ).clip(-1, 1)

    # Did safety drive forward (toward LOS)?
    flight_agg["drove_forward"] = (
        flight_agg["flight_x_end"] < flight_agg["flight_x_start"]
    ).astype(int)

    # Lateral movement (hash crossing)
    flight_agg["lateral_flight_dist"] = (
        flight_agg["flight_y_end"] - flight_agg["flight_y_start"]
    ).abs()

    print(f"  Post-throw features for {len(flight_agg):,} player-plays")

    return flight_agg


def compute_play_level_features(input_df, frame_refs, shells):
    """
    Aggregate player-level features to play level.
    This creates the feature matrix for modeling.
    """
    print("Computing play-level features...")

    defense = input_df[
        (input_df["player_side"] == "Defense") & (input_df["player_role"] == "Defensive Coverage")
    ].copy()

    # Get data at throw frame
    throw_data = defense.merge(
        frame_refs[["game_id", "play_id", "throw_frame"]],
        on=["game_id", "play_id"],
    )
    throw_data = throw_data[throw_data["frame_id"] == throw_data["throw_frame"]]

    # Play-level aggregations of defensive geometry at throw
    play_features = throw_data.groupby(["game_id", "play_id"]).agg(
        # Overall defense
        n_coverage_players=("nfl_id", "count"),
        mean_def_depth=("depth_from_los", "mean"),
        max_def_depth=("depth_from_los", "max"),
        std_def_depth=("depth_from_los", "std"),
        mean_def_speed=("s", "mean"),
        max_def_speed=("s", "max"),
        mean_def_accel=("a", "mean"),
    ).reset_index()

    # Safety-specific aggregations
    safety_throw = throw_data[throw_data["is_safety"]]
    safety_agg = safety_throw.groupby(["game_id", "play_id"]).agg(
        n_safeties=("nfl_id", "count"),
        safety_mean_depth=("depth_from_los", "mean"),
        safety_mean_width=("width_from_center", lambda x: x.abs().mean()),
        safety_max_speed=("s", "max"),
        safety_mean_accel=("a", "mean"),
    ).reset_index()

    play_features = play_features.merge(safety_agg, on=["game_id", "play_id"], how="left")

    # Merge shell classification
    play_features = play_features.merge(shells, on=["game_id", "play_id"], how="left")

    print(f"  Play-level features: {play_features.shape}")

    return play_features


def build_rotation_label(input_df, output_df, frame_refs):
    """
    Automatically label plays with late safety rotation.

    A rotation is detected when:
    - A safety moves significantly (>= 3 yards) during ball flight
    - AND changes their lateral position (crosses hash trajectory)
    - OR drives forward significantly (>= 2 yards toward LOS)

    This is the TARGET VARIABLE for our model.
    """
    print("Building rotation labels...")

    # Get safety info
    safety_info = (
        input_df[input_df["is_safety"]]
        .drop_duplicates(["game_id", "play_id", "nfl_id"])[
            ["game_id", "play_id", "nfl_id", "player_position"]
        ]
    )

    # Get safety position at throw frame (last pre-throw frame)
    safety_at_throw = input_df[input_df["is_safety"]].merge(
        frame_refs[["game_id", "play_id", "throw_frame"]],
        on=["game_id", "play_id"],
    )
    safety_at_throw = safety_at_throw[
        safety_at_throw["frame_id"] == safety_at_throw["throw_frame"]
    ][["game_id", "play_id", "nfl_id", "x", "y", "depth_from_los", "width_from_center"]]
    safety_at_throw = safety_at_throw.rename(columns={
        "x": "pre_x", "y": "pre_y",
        "depth_from_los": "pre_depth", "width_from_center": "pre_width"
    })

    # Get safety positions during ball flight (output data)
    flight_safety = output_df.merge(safety_info, on=["game_id", "play_id", "nfl_id"])

    if len(flight_safety) == 0:
        print("  No safeties found in flight data.")
        # Fall back to pre-throw delta-based rotation detection
        return _label_rotation_from_deltas(input_df, frame_refs)

    # Get end-of-flight positions
    flight_end = flight_safety.groupby(["game_id", "play_id", "nfl_id"]).agg(
        post_x=("x", "last"),
        post_y=("y", "last"),
    ).reset_index()

    # Merge pre and post
    rotation_data = safety_at_throw.merge(
        flight_end, on=["game_id", "play_id", "nfl_id"], how="inner"
    )

    # Compute movement
    rotation_data["displacement"] = np.sqrt(
        (rotation_data["post_x"] - rotation_data["pre_x"])**2
        + (rotation_data["post_y"] - rotation_data["pre_y"])**2
    )
    rotation_data["lateral_movement"] = (
        rotation_data["post_y"] - rotation_data["pre_y"]
    ).abs()
    rotation_data["forward_drive"] = rotation_data["pre_x"] - rotation_data["post_x"]

    # Label rotation: significant movement + lateral or forward component
    rotation_data["rotated"] = (
        (rotation_data["displacement"] >= 3.0)
        & (
            (rotation_data["lateral_movement"] >= 2.0)
            | (rotation_data["forward_drive"] >= 2.0)
        )
    )

    # Aggregate to play level: did ANY safety rotate?
    play_rotation = (
        rotation_data.groupby(["game_id", "play_id"])
        .agg(
            any_safety_rotated=("rotated", "any"),
            max_safety_displacement=("displacement", "max"),
            max_lateral_movement=("lateral_movement", "max"),
            max_forward_drive=("forward_drive", "max"),
        )
        .reset_index()
    )

    print(f"  Rotation labeled for {len(play_rotation):,} plays")
    print(f"  Plays with safety rotation: {play_rotation.any_safety_rotated.sum():,} "
          f"({play_rotation.any_safety_rotated.mean()*100:.1f}%)")

    return play_rotation


def _label_rotation_from_deltas(input_df, frame_refs):
    """
    Fallback: Label rotation using pre-throw delta movement.
    Uses the movement between early and throw frames to detect late adjustments.
    """
    print("  Using pre-throw deltas as fallback for rotation labeling...")

    safety_df = input_df[input_df["is_safety"]].copy()

    # Early frame data
    early = safety_df.merge(
        frame_refs[["game_id", "play_id", "early_frame"]], on=["game_id", "play_id"]
    )
    early = early[early["frame_id"] == early["early_frame"]]
    early = early.set_index(["game_id", "play_id", "nfl_id"])[
        ["x", "y", "depth_from_los", "width_from_center", "dir", "s"]
    ].rename(columns=lambda c: f"early_{c}")

    # Throw frame data
    throw = safety_df.merge(
        frame_refs[["game_id", "play_id", "throw_frame"]], on=["game_id", "play_id"]
    )
    throw = throw[throw["frame_id"] == throw["throw_frame"]]
    throw = throw.set_index(["game_id", "play_id", "nfl_id"])[
        ["x", "y", "depth_from_los", "width_from_center", "dir", "s"]
    ].rename(columns=lambda c: f"throw_{c}")

    combined = early.join(throw, how="inner").reset_index()

    # Movement metrics
    combined["displacement"] = np.sqrt(
        (combined["throw_x"] - combined["early_x"])**2
        + (combined["throw_y"] - combined["early_y"])**2
    )
    combined["lateral_movement"] = (combined["throw_y"] - combined["early_y"]).abs()
    combined["depth_change"] = combined["throw_depth_from_los"] - combined["early_depth_from_los"]

    # Direction change
    dir_diff = combined["throw_dir"] - combined["early_dir"]
    combined["dir_change"] = ((dir_diff + 180) % 360 - 180).abs()

    # Late rotation: significant movement + direction change
    combined["rotated"] = (
        (combined["displacement"] >= 2.5)
        & (combined["dir_change"] >= 30)
    )

    play_rotation = (
        combined.groupby(["game_id", "play_id"])
        .agg(
            any_safety_rotated=("rotated", "any"),
            max_safety_displacement=("displacement", "max"),
            max_lateral_movement=("lateral_movement", "max"),
            max_forward_drive=("depth_change", lambda x: (-x).max()),
        )
        .reset_index()
    )

    print(f"  Rotation labeled for {len(play_rotation):,} plays (delta method)")
    print(f"  Plays with safety rotation: {play_rotation.any_safety_rotated.sum():,} "
          f"({play_rotation.any_safety_rotated.mean()*100:.1f}%)")

    return play_rotation


if __name__ == "__main__":
    input_df = pd.read_pickle("data/input_processed.pkl")
    output_df = pd.read_pickle("data/output_processed.pkl")
    frame_refs = pd.read_pickle("data/frame_refs.pkl")
    shells = pd.read_pickle("data/shells.pkl")

    # Pre-throw features
    player_features = compute_pre_throw_features(input_df, frame_refs)
    player_features.to_pickle("data/player_features.pkl")

    # Delta features
    deltas = compute_delta_features(input_df, frame_refs)
    deltas.to_pickle("data/deltas.pkl")

    # Post-throw features
    post_throw = compute_post_throw_features(input_df, output_df, frame_refs)
    post_throw.to_pickle("data/post_throw_features.pkl")

    # Play-level features
    play_features = compute_play_level_features(input_df, frame_refs, shells)
    play_features.to_pickle("data/play_features.pkl")

    # Rotation labels
    rotation_labels = build_rotation_label(input_df, output_df, frame_refs)
    rotation_labels.to_pickle("data/rotation_labels.pkl")

    print("\nFeature engineering complete.")
