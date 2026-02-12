"""
Step 10: Visualizations for the Closing Angle Thesis
=====================================================
"Distance Lies, Direction Doesn't"

Produces:
  1. closing_angle_gradient.png — The core chart: angle → closure/end distance
  2. distance_vs_angle_heatmap.png — The decision matrix
  3. position_closing.png — CB vs LB closing comparison
  4. example_plays.png — Side-by-side: beeline closer vs wrong-way defender
  5. qb_decision_rule.png — Pre-throw direction predicts flight outcome
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import os


def draw_field(ax, yardline_range=(0, 120), yard_numbers=True):
    """Draw a simplified football field."""
    ax.set_facecolor("#2e7d32")
    ax.set_xlim(yardline_range)
    ax.set_ylim(0, 53.3)
    ax.set_aspect("equal")

    # Yard lines
    for yd in range(0, 121, 5):
        lw = 1.5 if yd % 10 == 0 else 0.5
        ax.axvline(yd, color="white", linewidth=lw, alpha=0.4)

    # Sidelines
    ax.axhline(0, color="white", linewidth=2)
    ax.axhline(53.3, color="white", linewidth=2)

    # Hash marks
    for yd in range(10, 111):
        for h in [23.6, 29.7]:
            ax.plot([yd, yd], [h - 0.3, h + 0.3], color="white", linewidth=0.5, alpha=0.3)


def plot_closing_angle_gradient(closing_df, outdir):
    """The core chart showing the closing angle gradient."""
    nearest = closing_df[closing_df["is_nearest"]].copy()

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle("Distance Lies, Direction Doesn't\nThe Closing Angle Thesis",
                 fontsize=16, fontweight="bold", y=1.02)

    bins = [0, 15, 30, 45, 60, 75, 90, 105, 120, 135, 150, 165, 180]
    nearest["angle_bin"] = pd.cut(nearest["closing_angle"], bins=bins)
    summary = nearest.groupby("angle_bin").agg(
        end_dist=("end_dist", "mean"),
        closure=("closure", "mean"),
        arrived=("arrived", "mean"),
        n=("arrived", "count"),
    ).reset_index()
    summary["mid"] = [(b.left + b.right) / 2 for b in summary["angle_bin"]]

    # Chart 1: Closure
    ax = axes[0]
    colors = ["#2196F3" if c > 0 else "#f44336" for c in summary["closure"]]
    ax.bar(summary["mid"], summary["closure"], width=13, color=colors, edgecolor="white", linewidth=0.5)
    ax.axhline(0, color="white", linewidth=1.5, linestyle="--")
    ax.axvline(60, color="#FFD700", linewidth=2, linestyle="--", label="60° threshold")
    ax.set_xlabel("Closing Angle (°)", fontsize=11)
    ax.set_ylabel("Closure (yards)", fontsize=11)
    ax.set_title("Closure During Ball Flight", fontsize=12, fontweight="bold")
    ax.legend()
    ax.set_facecolor("#1a1a2e")
    ax.tick_params(colors="white")
    for spine in ax.spines.values():
        spine.set_color("white")

    # Chart 2: End distance
    ax = axes[1]
    ax.bar(summary["mid"], summary["end_dist"], width=13,
           color=["#4CAF50" if e < 4 else "#FF9800" if e < 7 else "#f44336" for e in summary["end_dist"]],
           edgecolor="white", linewidth=0.5)
    ax.axvline(60, color="#FFD700", linewidth=2, linestyle="--", label="60° threshold")
    ax.set_xlabel("Closing Angle (°)", fontsize=11)
    ax.set_ylabel("End Distance to Catch (yards)", fontsize=11)
    ax.set_title("Where Defender Ends Up", fontsize=12, fontweight="bold")
    ax.legend()
    ax.set_facecolor("#1a1a2e")
    ax.tick_params(colors="white")
    for spine in ax.spines.values():
        spine.set_color("white")

    # Chart 3: Arrival rate
    ax = axes[2]
    ax.bar(summary["mid"], summary["arrived"] * 100, width=13,
           color=["#4CAF50" if a > 0.2 else "#FF9800" if a > 0.1 else "#f44336" for a in summary["arrived"]],
           edgecolor="white", linewidth=0.5)
    ax.axvline(60, color="#FFD700", linewidth=2, linestyle="--", label="60° threshold")
    ax.set_xlabel("Closing Angle (°)", fontsize=11)
    ax.set_ylabel("Arrival Rate (%)", fontsize=11)
    ax.set_title("% Arriving Within 2 yds of Catch", fontsize=12, fontweight="bold")
    ax.legend()
    ax.set_facecolor("#1a1a2e")
    ax.tick_params(colors="white")
    for spine in ax.spines.values():
        spine.set_color("white")

    fig.patch.set_facecolor("#1a1a2e")
    plt.tight_layout()
    path = os.path.join(outdir, "closing_angle_gradient.png")
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor="#1a1a2e")
    plt.close()
    print(f"  Saved {path}")


def plot_distance_vs_angle_heatmap(closing_df, outdir):
    """The decision matrix heatmap."""
    nearest = closing_df[closing_df["is_nearest"]].copy()

    dist_edges = [0, 5, 8, 12, 50]
    dist_labels = ["0-5 yds", "5-8 yds", "8-12 yds", "12+ yds"]
    angle_edges = [0, 45, 90, 135, 180]
    angle_labels = ["0-45°\n(beeline)", "45-90°", "90-135°\n(lateral)", "135-180°\n(away)"]

    nearest["d_bin"] = pd.cut(nearest["start_dist"], bins=dist_edges, labels=dist_labels)
    nearest["a_bin"] = pd.cut(nearest["closing_angle"], bins=angle_edges, labels=angle_labels)

    pivot = nearest.groupby(["d_bin", "a_bin"])["end_dist"].mean().unstack()

    fig, ax = plt.subplots(figsize=(10, 6))
    im = ax.imshow(pivot.values, cmap="RdYlGn_r", aspect="auto", vmin=1, vmax=14)

    ax.set_xticks(range(len(angle_labels)))
    ax.set_xticklabels(angle_labels, fontsize=11)
    ax.set_yticks(range(len(dist_labels)))
    ax.set_yticklabels(dist_labels, fontsize=11)
    ax.set_xlabel("Closing Angle at Release", fontsize=12, fontweight="bold")
    ax.set_ylabel("Starting Distance from Catch Point", fontsize=12, fontweight="bold")

    for i in range(len(dist_labels)):
        for j in range(len(angle_labels)):
            val = pivot.values[i, j]
            if not np.isnan(val):
                n = nearest[(nearest["d_bin"] == dist_labels[i]) & (nearest["a_bin"] == angle_labels[j])].shape[0]
                text_color = "white" if val > 7 else "black"
                ax.text(j, i, f"{val:.1f}\n(n={n:,})", ha="center", va="center",
                        fontsize=10, fontweight="bold", color=text_color)

    cbar = fig.colorbar(im, ax=ax, label="End Distance to Catch (yds)")
    ax.set_title("Distance × Angle: End Distance to Catch Point\n"
                 '"A 12-yard defender on a beeline beats a 5-yard defender heading sideways"',
                 fontsize=13, fontweight="bold")

    plt.tight_layout()
    path = os.path.join(outdir, "distance_vs_angle_heatmap.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved {path}")


def plot_position_closing(closing_df, outdir):
    """CB vs LB closing comparison."""
    nearest = closing_df[closing_df["is_nearest"]].copy()

    positions = {
        "CB": nearest[nearest["player_position"] == "CB"],
        "Safety": nearest[nearest["player_position"].isin(["SS", "FS"])],
        "LB": nearest[nearest["player_position"].isin(["ILB", "OLB", "MLB"])],
    }

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle("Position Matters: Who Can Close During Ball Flight?",
                 fontsize=14, fontweight="bold")

    colors = {"CB": "#2196F3", "Safety": "#FF9800", "LB": "#f44336"}

    # Chart 1: Arrival rate
    ax = axes[0]
    pos_names = list(positions.keys())
    arrival_rates = [positions[p]["arrived"].mean() * 100 for p in pos_names]
    bars = ax.bar(pos_names, arrival_rates, color=[colors[p] for p in pos_names],
                  edgecolor="white", linewidth=1.5)
    ax.set_ylabel("Arrival Rate (%)", fontsize=11)
    ax.set_title("% Arriving at Catch Point", fontsize=12, fontweight="bold")
    for bar, rate in zip(bars, arrival_rates):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                f"{rate:.1f}%", ha="center", fontweight="bold", fontsize=12)

    # Chart 2: Same-distance comparison (5-8 yds)
    ax = axes[1]
    mid_dist = {}
    for p in pos_names:
        sub = positions[p]
        mask = (sub["start_dist"] >= 5) & (sub["start_dist"] < 8)
        if mask.sum() > 20:
            mid_dist[p] = sub.loc[mask, "arrived"].mean() * 100
    if mid_dist:
        bars = ax.bar(list(mid_dist.keys()), list(mid_dist.values()),
                      color=[colors[p] for p in mid_dist.keys()],
                      edgecolor="white", linewidth=1.5)
        ax.set_ylabel("Arrival Rate (%)", fontsize=11)
        ax.set_title("Same Distance (5-8 yds):\nWho Still Closes?", fontsize=12, fontweight="bold")
        for bar, rate in zip(bars, mid_dist.values()):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                    f"{rate:.1f}%", ha="center", fontweight="bold", fontsize=12)

    # Chart 3: % with wrong-way angle
    ax = axes[2]
    wrong_pcts = [(positions[p]["closing_angle"] >= 60).mean() * 100 for p in pos_names]
    bars = ax.bar(pos_names, wrong_pcts, color=[colors[p] for p in pos_names],
                  edgecolor="white", linewidth=1.5)
    ax.set_ylabel("% With Wrong-Way Angle (≥60°)", fontsize=11)
    ax.set_title("Who Gets Caught Going\nthe Wrong Way?", fontsize=12, fontweight="bold")
    for bar, pct in zip(bars, wrong_pcts):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                f"{pct:.1f}%", ha="center", fontweight="bold", fontsize=12)

    plt.tight_layout()
    path = os.path.join(outdir, "position_closing.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved {path}")


def plot_qb_decision_rule(closing_df, input_df, frame_refs, outdir):
    """Pre-throw direction predicts flight outcome."""
    defense = input_df[input_df["player_side"] == "Defense"]
    throw_data = defense.merge(
        frame_refs[["game_id", "play_id", "throw_frame"]], on=["game_id", "play_id"]
    )
    throw_data = throw_data[throw_data["frame_id"] == throw_data["throw_frame"]]

    throw_state = throw_data[
        ["game_id", "play_id", "nfl_id", "s", "dir"]
    ].rename(columns={"s": "throw_speed", "dir": "throw_dir"})

    nearest = closing_df[closing_df["is_nearest"]].copy()
    nearest = nearest.merge(throw_state, on=["game_id", "play_id", "nfl_id"], how="inner")

    # Direction at throw vs catch point
    dx = nearest["ball_land_x"] - nearest["start_x"]
    dy = nearest["ball_land_y"] - nearest["start_y"]
    dir_to_ball = np.degrees(np.arctan2(dy, dx)) % 360

    def angle_diff_s(a, b):
        d = a - b
        return ((d + 180) % 360 - 180).abs()

    nearest["dir_at_throw_vs_ball"] = angle_diff_s(nearest["throw_dir"], pd.Series(dir_to_ball, index=nearest.index))

    fig, ax = plt.subplots(figsize=(10, 6))

    # Scatter: x = direction at throw, y = end distance, color = speed
    sample = nearest.sample(min(5000, len(nearest)), random_state=42)
    sc = ax.scatter(
        sample["dir_at_throw_vs_ball"],
        sample["end_dist"],
        c=sample["throw_speed"],
        cmap="coolwarm",
        alpha=0.3,
        s=8,
        vmin=0,
        vmax=8,
    )

    # Trend line
    bins = np.arange(0, 181, 10)
    centers = (bins[:-1] + bins[1:]) / 2
    means = []
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (nearest["dir_at_throw_vs_ball"] >= lo) & (nearest["dir_at_throw_vs_ball"] < hi)
        means.append(nearest.loc[mask, "end_dist"].mean() if mask.sum() > 20 else np.nan)
    ax.plot(centers, means, color="#FFD700", linewidth=3, label="Average end distance")
    ax.axvline(60, color="#f44336", linewidth=2, linestyle="--", label="60° threshold")

    ax.set_xlabel("Defender Direction at Throw vs. Catch Point (degrees)", fontsize=12)
    ax.set_ylabel("End Distance from Catch Point (yards)", fontsize=12)
    ax.set_title("The QB Decision Rule:\nDefender Direction at Release Predicts Arrival",
                 fontsize=14, fontweight="bold")
    cbar = fig.colorbar(sc, ax=ax, label="Defender Speed at Throw (yd/s)")
    ax.legend(fontsize=11)

    plt.tight_layout()
    path = os.path.join(outdir, "qb_decision_rule.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved {path}")


def plot_example_plays(closing_df, input_df, output_df, frame_refs, outdir):
    """Show a beeline closer vs a wrong-way defender on the same figure."""
    nearest = closing_df[closing_df["is_nearest"]].copy()

    # Find a good beeline example: high closure, arrives
    beeline = nearest[(nearest["closing_angle"] < 30) & (nearest["arrived"]) & (nearest["start_dist"] > 5)]
    beeline = beeline.sort_values("closure", ascending=False)

    # Find a good wrong-way example: high angle, doesn't arrive, was close
    wrong_way = nearest[(nearest["closing_angle"] > 120) & (~nearest["arrived"]) & (nearest["start_dist"] < 8)]
    wrong_way = wrong_way.sort_values("end_dist", ascending=False)

    if len(beeline) == 0 or len(wrong_way) == 0:
        print("  Not enough examples for play visualization")
        return

    fig, axes = plt.subplots(1, 2, figsize=(18, 8))

    for idx, (example, ax, title) in enumerate([
        (beeline.iloc[0], axes[0], "Beeline Closer: Arrives at Catch"),
        (wrong_way.iloc[0], axes[1], "Wrong-Way Defender: Drifts Away"),
    ]):
        gid, pid, nfl_id = example["game_id"], example["play_id"], example["nfl_id"]

        draw_field(ax, yardline_range=(
            min(example["start_x"], example["ball_land_x"]) - 10,
            max(example["start_x"], example["ball_land_x"]) + 10
        ))

        # Defender path
        def_path = output_df[
            (output_df["game_id"] == gid) & (output_df["play_id"] == pid) & (output_df["nfl_id"] == nfl_id)
        ].sort_values("frame_id")

        if len(def_path) > 0:
            ax.plot(def_path["x"], def_path["y"], "r-", linewidth=2, alpha=0.8, label="Defender path")
            ax.plot(def_path.iloc[0]["x"], def_path.iloc[0]["y"], "ro", markersize=10, zorder=5)
            ax.plot(def_path.iloc[-1]["x"], def_path.iloc[-1]["y"], "rs", markersize=10, zorder=5)

        # Target receiver path
        target_ids = input_df[
            (input_df["game_id"] == gid) & (input_df["play_id"] == pid)
            & (input_df["player_role"] == "Targeted Receiver")
        ]["nfl_id"].unique()

        for tid in target_ids:
            tgt_path = output_df[
                (output_df["game_id"] == gid) & (output_df["play_id"] == pid) & (output_df["nfl_id"] == tid)
            ].sort_values("frame_id")
            if len(tgt_path) > 0:
                ax.plot(tgt_path["x"], tgt_path["y"], "b-", linewidth=2, alpha=0.8, label="Receiver path")
                ax.plot(tgt_path.iloc[0]["x"], tgt_path.iloc[0]["y"], "bo", markersize=10, zorder=5)

        # Ball landing
        ax.plot(example["ball_land_x"], example["ball_land_y"], "y*",
                markersize=20, zorder=10, label="Catch point")

        # Annotations
        ax.set_title(f"{title}\nAngle: {example['closing_angle']:.0f}°  |  "
                     f"Start: {example['start_dist']:.1f} yds  |  "
                     f"End: {example['end_dist']:.1f} yds  |  "
                     f"Closure: {example['closure']:+.1f} yds",
                     fontsize=11, fontweight="bold", color="white")
        ax.legend(loc="upper right", fontsize=9)
        ax.set_xlabel("Yards", color="white")
        ax.set_ylabel("Field Width", color="white")

    fig.suptitle("Distance Lies, Direction Doesn't\nTwo Defenders — Same Play Distance, Opposite Outcomes",
                 fontsize=14, fontweight="bold")
    plt.tight_layout()
    path = os.path.join(outdir, "example_plays.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved {path}")


if __name__ == "__main__":
    print("=" * 60)
    print("  CLOSING ANGLE VISUALIZATIONS")
    print("=" * 60)

    input_df = pd.read_pickle("data/input_processed.pkl")
    output_df = pd.read_pickle("data/output_processed.pkl")
    frame_refs = pd.read_pickle("data/frame_refs.pkl")
    closing_df = pd.read_pickle("data/closing_angle.pkl")

    outdir = "output/figures"
    os.makedirs(outdir, exist_ok=True)

    plot_closing_angle_gradient(closing_df, outdir)
    plot_distance_vs_angle_heatmap(closing_df, outdir)
    plot_position_closing(closing_df, outdir)
    plot_qb_decision_rule(closing_df, input_df, frame_refs, outdir)
    plot_example_plays(closing_df, input_df, output_df, frame_refs, outdir)

    print("\nAll closing angle visualizations complete.")
