"""
Step 4: Disguise Detection Model
=================================
Core insight: Coverage disguise = pre-throw structure ≠ post-throw execution.

We refine the label to detect STRUCTURAL shell mismatch, not just movement.
Then train a Random Forest to identify what pre-throw features predict disguise.

Variable importance is the key output — that's what coaches care about.
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import classification_report, roc_auc_score
import warnings
warnings.filterwarnings("ignore")


def refine_disguise_label(input_df, output_df, frame_refs, shells):
    """
    Build a refined disguise label based on structural shell mismatch.

    Disguise detected when:
    1. Pre-throw shell is 2-high BUT a safety drives forward/down during flight
       (showing Cover-2 shell but executing Cover-1/Cover-3)
    2. Pre-throw shell is 1-high BUT the low safety bails deep during flight
       (showing Cover-1 shell but executing Cover-2)
    3. Pre-throw alignment shows one structure but safety movement pattern
       contradicts it significantly

    Additionally uses pre-throw DELTA features to detect late pre-throw rotation.
    """
    print("Building refined disguise labels...")

    # --- Method 1: Shell change in pre-throw window ---
    # Already computed in shells: shell_changed, two_to_one, one_to_two
    shell_disguise = shells[["game_id", "play_id", "shell_changed",
                             "two_to_one", "one_to_two",
                             "early_shell", "throw_shell"]].copy()

    # --- Method 2: Post-throw structural mismatch ---
    safety_info = (
        input_df[input_df["is_safety"]]
        .drop_duplicates(["game_id", "play_id", "nfl_id"])[
            ["game_id", "play_id", "nfl_id", "player_position"]
        ]
    )

    # Safety positions at throw
    safety_throw = input_df[input_df["is_safety"]].merge(
        frame_refs[["game_id", "play_id", "throw_frame"]],
        on=["game_id", "play_id"],
    )
    safety_throw = safety_throw[safety_throw["frame_id"] == safety_throw["throw_frame"]]

    # Safety positions during flight
    flight_safety = output_df.merge(safety_info, on=["game_id", "play_id", "nfl_id"])
    flight_end = (
        flight_safety.groupby(["game_id", "play_id", "nfl_id"])
        .agg(post_x=("x", "last"), post_y=("y", "last"))
        .reset_index()
    )

    # Merge pre and post for safeties
    safety_movement = safety_throw[
        ["game_id", "play_id", "nfl_id", "x", "y",
         "depth_from_los", "width_from_center", "los_x"]
    ].merge(flight_end, on=["game_id", "play_id", "nfl_id"], how="inner")

    # Compute structural movement
    safety_movement["depth_change"] = (
        (safety_movement["post_x"] - safety_movement["los_x"])
        - safety_movement["depth_from_los"]
    )
    safety_movement["drove_toward_los"] = safety_movement["depth_change"] < -3.0
    safety_movement["bailed_deep"] = safety_movement["depth_change"] > 3.0
    safety_movement["hash_crossing"] = (
        (safety_movement["width_from_center"] * (safety_movement["post_y"] - 26.65)) < 0
    )  # Changed sides of field

    # Play-level post-throw behavior
    play_post = safety_movement.groupby(["game_id", "play_id"]).agg(
        any_drove_forward=("drove_toward_los", "any"),
        any_bailed_deep=("bailed_deep", "any"),
        any_hash_cross=("hash_crossing", "any"),
        max_depth_change=("depth_change", lambda x: x.abs().max()),
    ).reset_index()

    # --- Combine into final disguise label ---
    label_df = shell_disguise.merge(play_post, on=["game_id", "play_id"], how="left")

    # Fill NaN for plays without flight safety data
    label_df["any_drove_forward"] = label_df["any_drove_forward"].fillna(False)
    label_df["any_bailed_deep"] = label_df["any_bailed_deep"].fillna(False)
    label_df["any_hash_cross"] = label_df["any_hash_cross"].fillna(False)

    # DISGUISE = shell change OR structural post-throw mismatch
    # 2-high shown → safety drives down = disguise
    # 1-high shown → safety bails deep = disguise
    # Any shell → pre-throw shell change = disguise
    label_df["disguised"] = (
        label_df["shell_changed"]  # Pre-throw rotation detected
        | (  # 2-high look but safety attacks
            (label_df["throw_shell"] == "2-high") & label_df["any_drove_forward"]
        )
        | (  # 1-high look but safety bails
            (label_df["throw_shell"] == "1-high") & label_df["any_bailed_deep"]
        )
        | label_df["any_hash_cross"]  # Safety crossed the hash
    )

    print(f"  Total plays: {len(label_df):,}")
    print(f"  Disguised plays: {label_df.disguised.sum():,} "
          f"({label_df.disguised.mean()*100:.1f}%)")
    print(f"  Breakdown:")
    print(f"    Shell changed pre-throw: {label_df.shell_changed.sum():,}")
    print(f"    Safety drove forward (2-high): "
          f"{((label_df.throw_shell == '2-high') & label_df.any_drove_forward).sum():,}")
    print(f"    Safety bailed deep (1-high): "
          f"{((label_df.throw_shell == '1-high') & label_df.any_bailed_deep).sum():,}")
    print(f"    Hash crossing: {label_df.any_hash_cross.sum():,}")

    return label_df


def build_model_dataset(input_df, frame_refs, shells, labels):
    """
    Build the final feature matrix for the disguise model.
    Play-level features predicting whether disguise occurred.
    """
    print("Building model dataset...")

    # Get defensive data at throw frame
    defense = input_df[
        (input_df["player_side"] == "Defense")
        & (input_df["player_role"] == "Defensive Coverage")
    ]
    throw_data = defense.merge(
        frame_refs[["game_id", "play_id", "throw_frame"]], on=["game_id", "play_id"]
    )
    throw_data = throw_data[throw_data["frame_id"] == throw_data["throw_frame"]]

    # Get defensive data at early frame
    early_data = defense.merge(
        frame_refs[["game_id", "play_id", "early_frame"]], on=["game_id", "play_id"]
    )
    early_data = early_data[early_data["frame_id"] == early_data["early_frame"]]

    # --- Safety features at throw ---
    safety_throw = throw_data[throw_data["is_safety"]]
    sf = safety_throw.groupby(["game_id", "play_id"]).agg(
        safety_depth_mean=("depth_from_los", "mean"),
        safety_depth_max=("depth_from_los", "max"),
        safety_depth_min=("depth_from_los", "min"),
        safety_depth_spread=("depth_from_los", lambda x: x.max() - x.min()),
        safety_width_mean=("width_from_center", lambda x: x.abs().mean()),
        safety_width_spread=("width_from_center", lambda x: x.max() - x.min()),
        safety_speed_mean=("s", "mean"),
        safety_speed_max=("s", "max"),
        safety_accel_mean=("a", "mean"),
        safety_accel_max=("a", "max"),
        n_safeties=("nfl_id", "count"),
    ).reset_index()

    # --- Safety features at early frame ---
    safety_early = early_data[early_data["is_safety"]]
    sf_early = safety_early.groupby(["game_id", "play_id"]).agg(
        early_safety_depth_mean=("depth_from_los", "mean"),
        early_safety_width_spread=("width_from_center", lambda x: x.max() - x.min()),
        early_safety_speed_mean=("s", "mean"),
    ).reset_index()

    # --- CB features at throw ---
    cb_throw = throw_data[throw_data["is_cb"]]
    cf = cb_throw.groupby(["game_id", "play_id"]).agg(
        cb_depth_mean=("depth_from_los", "mean"),
        cb_depth_max=("depth_from_los", "max"),
        cb_speed_mean=("s", "mean"),
        cb_width_mean=("width_from_center", lambda x: x.abs().mean()),
        n_cbs=("nfl_id", "count"),
    ).reset_index()

    # --- LB features at throw ---
    lb_throw = throw_data[throw_data["is_lb"]]
    lf = lb_throw.groupby(["game_id", "play_id"]).agg(
        lb_depth_mean=("depth_from_los", "mean"),
        lb_speed_mean=("s", "mean"),
        n_lbs=("nfl_id", "count"),
    ).reset_index()

    # --- Overall defense at throw ---
    df_all = throw_data.groupby(["game_id", "play_id"]).agg(
        defense_depth_std=("depth_from_los", "std"),
        defense_speed_mean=("s", "mean"),
        defense_accel_mean=("a", "mean"),
        n_defenders=("nfl_id", "count"),
    ).reset_index()

    # --- Delta features (safety movement early -> throw) ---
    deltas = pd.read_pickle("data/deltas.pkl")
    safety_ids = input_df[input_df["is_safety"]][["nfl_id"]].drop_duplicates()
    safety_deltas = deltas[deltas["nfl_id"].isin(safety_ids["nfl_id"])]

    delta_agg = safety_deltas.groupby(["game_id", "play_id"]).agg(
        delta_depth_mean=("delta_depth", "mean"),
        delta_depth_max=("delta_depth", lambda x: x.abs().max()),
        delta_width_mean=("delta_width", lambda x: x.abs().mean()),
        delta_speed_mean=("delta_speed", "mean"),
        delta_dir_mean=("delta_dir", "mean"),
        delta_dir_max=("delta_dir", "max"),
        any_major_dir_change=("major_dir_change", "max"),
        displacement_mean=("total_displacement", "mean"),
        displacement_max=("total_displacement", "max"),
        mof_drift_mean=("mof_drift", "mean"),
        accel_spike_any=("accel_spike", "max"),
    ).reset_index()

    # --- Merge everything ---
    model_df = labels[["game_id", "play_id", "disguised"]].copy()
    for feat_df in [sf, sf_early, cf, lf, df_all, delta_agg]:
        model_df = model_df.merge(feat_df, on=["game_id", "play_id"], how="left")

    # Add shell info
    shell_features = shells[[
        "game_id", "play_id", "throw_shell",
        "throw_safety_depth_mean", "throw_safety_width_spread",
        "throw_safety_mof_distance",
    ]].copy()
    shell_features["shell_2high"] = (shell_features["throw_shell"] == "2-high").astype(int)
    shell_features["shell_1high"] = (shell_features["throw_shell"] == "1-high").astype(int)
    shell_features["shell_0high"] = (shell_features["throw_shell"] == "0-high").astype(int)
    model_df = model_df.merge(
        shell_features.drop(columns=["throw_shell"]),
        on=["game_id", "play_id"],
        how="left",
    )

    # Drop NaN rows
    feature_cols = [c for c in model_df.columns if c not in ["game_id", "play_id", "disguised"]]
    model_df = model_df.dropna(subset=feature_cols)

    print(f"  Model dataset: {model_df.shape[0]:,} plays, {len(feature_cols)} features")
    print(f"  Disguise rate: {model_df.disguised.mean()*100:.1f}%")

    return model_df, feature_cols


def train_model(model_df, feature_cols):
    """
    Train a Random Forest classifier — exactly Dom-style.
    Interpretability > squeezing 2% accuracy.
    """
    print("\nTraining Random Forest model...")

    X = model_df[feature_cols].values
    y = model_df["disguised"].astype(int).values

    # Random Forest with OOB estimation (like caret OOB)
    rf = RandomForestClassifier(
        n_estimators=500,
        max_depth=12,
        min_samples_leaf=20,
        max_features="sqrt",
        oob_score=True,
        random_state=42,
        n_jobs=-1,
        class_weight="balanced",
    )
    rf.fit(X, y)

    print(f"  OOB Accuracy: {rf.oob_score_:.3f}")

    # Cross-validation AUC
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    auc_scores = cross_val_score(rf, X, y, cv=cv, scoring="roc_auc")
    print(f"  5-Fold CV AUC: {auc_scores.mean():.3f} (+/- {auc_scores.std():.3f})")

    # Feature importance
    importance = pd.DataFrame({
        "feature": feature_cols,
        "importance": rf.feature_importances_,
    }).sort_values("importance", ascending=False)

    print("\n  Top 15 Most Important Features:")
    print("  " + "-" * 50)
    for _, row in importance.head(15).iterrows():
        bar = "█" * int(row["importance"] * 200)
        print(f"  {row['feature']:35s} {row['importance']:.4f} {bar}")

    return rf, importance


def generate_predictions(rf, model_df, feature_cols):
    """Generate disguise probability for each play."""
    X = model_df[feature_cols].values
    model_df = model_df.copy()
    model_df["disguise_prob"] = rf.predict_proba(X)[:, 1]
    return model_df


if __name__ == "__main__":
    print("=" * 60)
    print("COVERAGE DISGUISE DETECTION MODEL")
    print("=" * 60)

    input_df = pd.read_pickle("data/input_processed.pkl")
    output_df = pd.read_pickle("data/output_processed.pkl")
    frame_refs = pd.read_pickle("data/frame_refs.pkl")
    shells = pd.read_pickle("data/shells.pkl")

    # Build refined labels
    labels = refine_disguise_label(input_df, output_df, frame_refs, shells)
    labels.to_pickle("data/disguise_labels.pkl")

    # Build model dataset
    model_df, feature_cols = build_model_dataset(input_df, frame_refs, shells, labels)

    # Train model
    rf, importance = train_model(model_df, feature_cols)

    # Generate predictions
    model_df = generate_predictions(rf, model_df, feature_cols)
    model_df.to_pickle("data/model_predictions.pkl")
    importance.to_pickle("data/feature_importance.pkl")

    # Save model
    import pickle
    with open("data/rf_model.pkl", "wb") as f:
        pickle.dump(rf, f)

    print("\nModel training complete.")
