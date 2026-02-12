"""
Step 6: Visualizations
=======================
"Most analytics die because nobody can see them."

Key plots:
  1. Field plot: Pre-throw positions (dots) → Post-throw movement (arrows)
  2. Feature importance bar chart
  3. Team disguise rankings
  4. Safety deception leaderboard
  5. Shell distribution with disguise overlay
  6. Disguise mechanism breakdown
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyArrowPatch
import seaborn as sns

# Style: clean, professional, football-native
plt.rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor": "#f8f9fa",
    "axes.grid": True,
    "grid.alpha": 0.3,
    "font.family": "sans-serif",
    "font.size": 11,
})

FIELD_GREEN = "#2d5016"
FIELD_WHITE = "#ffffff"
SAFETY_COLOR = "#e74c3c"  # Red for safeties
CB_COLOR = "#3498db"      # Blue for CBs
WR_COLOR = "#f39c12"      # Orange for receivers
QB_COLOR = "#2ecc71"      # Green for QB
ARROW_COLOR = "#e74c3c"   # Red for movement arrows
DISGUISE_COLOR = "#e74c3c"
NO_DISGUISE_COLOR = "#95a5a6"


def draw_football_field(ax, ymin=0, ymax=53.3, xmin=0, xmax=120):
    """Draw a football field background."""
    ax.set_facecolor(FIELD_GREEN)

    # Yard lines
    for x in range(10, 111, 5):
        lw = 1.5 if x % 10 == 0 else 0.5
        alpha = 0.8 if x % 10 == 0 else 0.3
        ax.axvline(x=x, color="white", linewidth=lw, alpha=alpha)

    # Yard numbers
    for yard in range(10, 100, 10):
        display_yard = yard if yard <= 50 else 100 - yard
        x_pos = yard + 10
        ax.text(x_pos, 5, str(display_yard), fontsize=16, color="white",
                alpha=0.5, ha="center", va="center", fontweight="bold")
        ax.text(x_pos, 48.3, str(display_yard), fontsize=16, color="white",
                alpha=0.5, ha="center", va="center", fontweight="bold")

    # Hash marks
    for x in range(10, 111):
        ax.plot([x, x], [23.36, 23.56], color="white", linewidth=0.5, alpha=0.5)
        ax.plot([x, x], [29.74, 29.94], color="white", linewidth=0.5, alpha=0.5)

    # End zones
    ax.axvline(x=10, color="white", linewidth=2)
    ax.axvline(x=110, color="white", linewidth=2)

    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_aspect("equal")


def plot_disguise_play(input_df, output_df, frame_refs, game_id, play_id, save_path=None):
    """
    Plot a single play showing pre-throw structure and post-throw movement.
    BEFORE: dots (positions at throw)
    AFTER: arrows (movement during ball flight)
    """
    fig, ax = plt.subplots(figsize=(16, 8))

    # Get data for this play
    play_input = input_df[
        (input_df["game_id"] == game_id) & (input_df["play_id"] == play_id)
    ]
    play_output = output_df[
        (output_df["game_id"] == game_id) & (output_df["play_id"] == play_id)
    ]

    throw_frame = play_input["frame_id"].max()

    # Pre-throw positions (at throw release)
    at_throw = play_input[play_input["frame_id"] == throw_frame]

    # Get field bounds
    all_x = at_throw["x"].tolist()
    if len(play_output) > 0:
        all_x.extend(play_output["x"].tolist())
    x_center = np.mean(all_x)
    x_min = max(0, x_center - 25)
    x_max = min(120, x_center + 25)

    draw_football_field(ax, xmin=x_min, xmax=x_max)

    # Plot LOS
    if "los_x" in at_throw.columns and not at_throw["los_x"].isna().all():
        los = at_throw["los_x"].iloc[0]
        ax.axvline(x=los, color="yellow", linewidth=2, alpha=0.7, linestyle="--",
                   label="Line of Scrimmage")

    # Plot ball landing point
    if "ball_land_x" in at_throw.columns:
        ball_x = at_throw["ball_land_x"].iloc[0]
        ball_y = at_throw["ball_land_y"].iloc[0]
        ax.scatter(ball_x, ball_y, s=200, c="yellow", marker="*", zorder=10,
                   edgecolors="black", linewidth=1, label="Ball Landing")

    # Plot pre-throw positions (DOTS)
    for _, player in at_throw.iterrows():
        if player["player_side"] == "Defense":
            if player.get("is_safety", False):
                color = SAFETY_COLOR
                marker = "s"
                size = 120
            elif player.get("is_cb", False):
                color = CB_COLOR
                marker = "o"
                size = 100
            else:
                color = "#7f8c8d"
                marker = "o"
                size = 80
        else:
            if player["player_role"] == "Passer":
                color = QB_COLOR
                marker = "^"
                size = 120
            elif player["player_role"] == "Targeted Receiver":
                color = WR_COLOR
                marker = "D"
                size = 120
            else:
                color = "#bdc3c7"
                marker = "o"
                size = 60

        ax.scatter(player["x"], player["y"], s=size, c=color, marker=marker,
                   edgecolors="white", linewidth=1, zorder=5)

        # Add player name (abbreviated)
        name = str(player.get("player_name", ""))
        if name and name != "nan":
            short_name = name.split()[-1][:6] if " " in name else name[:6]
            ax.text(player["x"], player["y"] + 1.2, short_name,
                    fontsize=6, color="white", ha="center", va="bottom",
                    fontweight="bold", zorder=6)

    # Plot post-throw movement (ARROWS)
    if len(play_output) > 0:
        for nfl_id in play_output["nfl_id"].unique():
            player_flight = play_output[play_output["nfl_id"] == nfl_id].sort_values("frame_id")

            if len(player_flight) < 2:
                continue

            start_x = player_flight["x"].iloc[0]
            start_y = player_flight["y"].iloc[0]
            end_x = player_flight["x"].iloc[-1]
            end_y = player_flight["y"].iloc[-1]

            # Check if this is a safety
            player_info = at_throw[at_throw["nfl_id"] == nfl_id]
            is_safety = False
            if len(player_info) > 0:
                is_safety = player_info["is_safety"].iloc[0] if "is_safety" in player_info.columns else False

            arrow_color = SAFETY_COLOR if is_safety else "#3498db"
            arrow_width = 2.5 if is_safety else 1.5

            # Draw trajectory
            ax.annotate(
                "", xy=(end_x, end_y), xytext=(start_x, start_y),
                arrowprops=dict(
                    arrowstyle="->", color=arrow_color,
                    lw=arrow_width, alpha=0.8,
                ),
                zorder=7,
            )

    # Legend
    legend_elements = [
        plt.scatter([], [], s=120, c=SAFETY_COLOR, marker="s", edgecolors="white", label="Safety"),
        plt.scatter([], [], s=100, c=CB_COLOR, marker="o", edgecolors="white", label="CB"),
        plt.scatter([], [], s=120, c=WR_COLOR, marker="D", edgecolors="white", label="Targeted WR"),
        plt.scatter([], [], s=120, c=QB_COLOR, marker="^", edgecolors="white", label="QB"),
    ]
    ax.legend(handles=legend_elements, loc="upper right", framealpha=0.9,
              fontsize=9, facecolor="white")

    ax.set_title(f"Coverage Disguise Detection — Game {game_id}, Play {play_id}\n"
                 f"Dots = Pre-throw positions | Arrows = Ball-in-air movement",
                 fontsize=13, fontweight="bold", color="#2c3e50", pad=15)
    ax.set_xlabel("Field Position (yards)", fontsize=10)
    ax.set_ylabel("Field Width (yards)", fontsize=10)

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  Saved: {save_path}")
    plt.close()
    return fig


def plot_feature_importance(importance_df, save_path=None):
    """Plot top feature importances — this is what coaches care about."""
    fig, ax = plt.subplots(figsize=(10, 8))

    top_n = 20
    top = importance_df.head(top_n).sort_values("importance")

    # Clean up feature names for readability
    name_map = {
        "safety_depth_min": "Safety Min Depth (from LOS)",
        "early_safety_depth_mean": "Safety Depth 1.5s Pre-Throw",
        "safety_depth_spread": "Safety Depth Spread",
        "safety_speed_max": "Safety Max Speed at Throw",
        "throw_safety_mof_distance": "Safety Distance from MOF",
        "defense_speed_mean": "Overall Defense Speed",
        "throw_safety_depth_mean": "Safety Mean Depth at Throw",
        "safety_speed_mean": "Safety Mean Speed",
        "safety_depth_mean": "Safety Avg Depth",
        "cb_speed_mean": "CB Mean Speed",
        "cb_depth_max": "CB Max Depth",
        "delta_depth_mean": "Safety Depth Change (delta)",
        "safety_accel_mean": "Safety Mean Acceleration",
        "displacement_mean": "Safety Pre-throw Displacement",
        "defense_depth_std": "Defensive Depth Variation",
        "delta_dir_mean": "Safety Direction Change (delta)",
        "safety_width_spread": "Safety Width Spread",
        "safety_accel_max": "Safety Max Acceleration",
        "delta_dir_max": "Max Direction Change",
        "safety_width_mean": "Safety Width from Center",
    }

    labels = [name_map.get(f, f) for f in top["feature"]]
    colors = ["#e74c3c" if "safety" in f.lower() or "delta" in f.lower()
              else "#3498db" for f in top["feature"]]

    bars = ax.barh(range(len(top)), top["importance"], color=colors, edgecolor="white")
    ax.set_yticks(range(len(top)))
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlabel("Feature Importance (Gini)", fontsize=11)
    ax.set_title("What Predicts Coverage Disguise?\n"
                 "Random Forest Variable Importance",
                 fontsize=14, fontweight="bold", pad=15)

    # Add legend for colors
    ax.scatter([], [], c="#e74c3c", s=100, label="Safety / Movement Features")
    ax.scatter([], [], c="#3498db", s=100, label="Other Defensive Features")
    ax.legend(loc="lower right", fontsize=9)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  Saved: {save_path}")
    plt.close()
    return fig


def plot_safety_leaderboard(safety_stats, save_path=None):
    """Plot safety deception leaderboard."""
    fig, ax = plt.subplots(figsize=(12, 8))

    qualified = safety_stats[safety_stats["n_plays"] >= 50].sort_values(
        "deception_score", ascending=False
    ).head(20)

    colors = plt.cm.RdYlGn_r(np.linspace(0.2, 0.8, len(qualified)))

    bars = ax.barh(
        range(len(qualified)),
        qualified["deception_score"],
        color=colors,
        edgecolor="white",
    )
    ax.set_yticks(range(len(qualified)))
    ax.set_yticklabels(
        [f"{row.player_name} ({row.player_position})" for _, row in qualified.iterrows()],
        fontsize=9,
    )

    # Add snap counts
    for i, (_, row) in enumerate(qualified.iterrows()):
        ax.text(
            row.deception_score + 0.5, i,
            f"{int(row.n_plays)} snaps | {row.disguise_rate*100:.0f}% disg",
            va="center", fontsize=8, color="#555",
        )

    ax.set_xlabel("Deception Score", fontsize=11)
    ax.set_title("Safety Deception Leaderboard\n"
                 "Which safeties are hardest to read pre-throw?",
                 fontsize=14, fontweight="bold", pad=15)
    ax.invert_yaxis()
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  Saved: {save_path}")
    plt.close()
    return fig


def plot_shell_disguise_rates(labels, shells, save_path=None):
    """Plot disguise rates by shell type with breakdown."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Left: Disguise rate by shell
    merged = labels.merge(
        shells[["game_id", "play_id", "throw_shell"]],
        on=["game_id", "play_id"], how="left",
        suffixes=("", "_s"),
    )
    shell_col = "throw_shell" if "throw_shell" in merged.columns else "throw_shell_s"

    shell_rates = merged.groupby(shell_col).agg(
        disguise_rate=("disguised", "mean"),
        n_plays=("play_id", "count"),
    ).reset_index()

    colors = ["#e74c3c", "#f39c12", "#2ecc71"]
    ax = axes[0]
    bars = ax.bar(shell_rates[shell_col], shell_rates["disguise_rate"] * 100,
                  color=colors, edgecolor="white", width=0.6)
    for bar, (_, row) in zip(bars, shell_rates.iterrows()):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f"{row['disguise_rate']*100:.1f}%\n({int(row['n_plays']):,})",
                ha="center", va="bottom", fontsize=10, fontweight="bold")
    ax.set_ylabel("Disguise Rate (%)", fontsize=11)
    ax.set_xlabel("Pre-Throw Shell", fontsize=11)
    ax.set_title("Disguise Rate by Shell Type", fontsize=13, fontweight="bold")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Right: Disguise mechanism breakdown
    ax2 = axes[1]
    mechanisms = {
        "Shell\nRotation": labels["shell_changed"].sum(),
        "Safety Drove\nForward": labels["any_drove_forward"].sum(),
        "Safety Bailed\nDeep": labels["any_bailed_deep"].sum(),
        "Hash\nCrossing": labels["any_hash_cross"].sum(),
    }
    mech_colors = ["#e74c3c", "#3498db", "#2ecc71", "#f39c12"]
    bars2 = ax2.bar(mechanisms.keys(), mechanisms.values(), color=mech_colors,
                    edgecolor="white", width=0.6)
    for bar, count in zip(bars2, mechanisms.values()):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 20,
                 f"{count:,}", ha="center", va="bottom", fontsize=10, fontweight="bold")
    ax2.set_ylabel("Number of Plays", fontsize=11)
    ax2.set_title("How Defenses Disguise Coverage", fontsize=13, fontweight="bold")
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)

    plt.suptitle("Coverage Disguise Analysis — NFL 2023-2024",
                 fontsize=15, fontweight="bold", y=1.02)
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  Saved: {save_path}")
    plt.close()
    return fig


def plot_model_performance(model_df, save_path=None):
    """Plot model performance - predicted vs actual disguise distribution."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left: Distribution of disguise probability
    ax = axes[0]
    ax.hist(model_df[model_df["disguised"]]["disguise_prob"], bins=40,
            alpha=0.7, color=DISGUISE_COLOR, label="Actually Disguised", density=True)
    ax.hist(model_df[~model_df["disguised"]]["disguise_prob"], bins=40,
            alpha=0.7, color=NO_DISGUISE_COLOR, label="Not Disguised", density=True)
    ax.set_xlabel("Predicted Disguise Probability", fontsize=11)
    ax.set_ylabel("Density", fontsize=11)
    ax.set_title("Model Separation", fontsize=13, fontweight="bold")
    ax.legend()
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Right: Calibration-style plot
    ax2 = axes[1]
    model_df_sorted = model_df.copy()
    model_df_sorted["prob_bin"] = pd.qcut(model_df_sorted["disguise_prob"], 10, duplicates="drop")
    cal = model_df_sorted.groupby("prob_bin", observed=True).agg(
        mean_predicted=("disguise_prob", "mean"),
        mean_actual=("disguised", "mean"),
    ).reset_index()
    ax2.plot([0, 1], [0, 1], "k--", alpha=0.5, label="Perfect Calibration")
    ax2.scatter(cal["mean_predicted"], cal["mean_actual"],
                s=100, c=DISGUISE_COLOR, zorder=5, edgecolors="white")
    ax2.plot(cal["mean_predicted"], cal["mean_actual"],
             c=DISGUISE_COLOR, alpha=0.7)
    ax2.set_xlabel("Predicted Disguise Probability", fontsize=11)
    ax2.set_ylabel("Observed Disguise Rate", fontsize=11)
    ax2.set_title("Model Calibration", fontsize=13, fontweight="bold")
    ax2.legend()
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  Saved: {save_path}")
    plt.close()
    return fig


def find_best_disguise_plays(input_df, labels, shells, n=5):
    """Find the most dramatic disguise plays for visualization."""
    merged = labels.merge(
        shells[["game_id", "play_id", "throw_shell", "early_shell",
                "throw_safety_width_spread", "throw_safety_depth_mean"]],
        on=["game_id", "play_id"],
        how="left",
        suffixes=("", "_s"),
    )

    # Focus on plays where shell changed AND there was significant movement
    dramatic = merged[merged["disguised"]].copy()
    dramatic["drama_score"] = (
        dramatic["shell_changed"].astype(int) * 2
        + dramatic["any_hash_cross"].astype(int) * 1.5
        + dramatic["any_drove_forward"].astype(int)
        + dramatic["any_bailed_deep"].astype(int)
    )

    # Must have flight data
    flight_plays = input_df[input_df["player_to_predict"]].drop_duplicates(
        ["game_id", "play_id"]
    )[["game_id", "play_id"]]
    dramatic = dramatic.merge(flight_plays, on=["game_id", "play_id"])

    top_plays = dramatic.nlargest(n, "drama_score")[["game_id", "play_id"]].values
    return top_plays


if __name__ == "__main__":
    print("=" * 60)
    print("GENERATING VISUALIZATIONS")
    print("=" * 60)

    input_df = pd.read_pickle("data/input_processed.pkl")
    output_df = pd.read_pickle("data/output_processed.pkl")
    frame_refs = pd.read_pickle("data/frame_refs.pkl")
    shells = pd.read_pickle("data/shells.pkl")
    labels = pd.read_pickle("data/disguise_labels.pkl")
    model_df = pd.read_pickle("data/model_predictions.pkl")
    importance = pd.read_pickle("data/feature_importance.pkl")
    safety_stats = pd.read_pickle("data/safety_stats.pkl")

    # 1. Feature importance
    print("\n1. Feature Importance Plot")
    plot_feature_importance(importance, "output/figures/feature_importance.png")

    # 2. Safety leaderboard
    print("2. Safety Deception Leaderboard")
    plot_safety_leaderboard(safety_stats, "output/figures/safety_leaderboard.png")

    # 3. Shell disguise rates
    print("3. Shell Disguise Rates")
    plot_shell_disguise_rates(labels, shells, "output/figures/shell_disguise_rates.png")

    # 4. Model performance
    print("4. Model Performance")
    plot_model_performance(model_df, "output/figures/model_performance.png")

    # 5. Example disguise plays (field plots)
    print("5. Example Disguise Plays")
    best_plays = find_best_disguise_plays(input_df, labels, shells)
    for i, (gid, pid) in enumerate(best_plays):
        plot_disguise_play(
            input_df, output_df, frame_refs, gid, pid,
            save_path=f"output/figures/disguise_play_{i+1}.png",
        )

    print("\nAll visualizations saved to output/figures/")
