import contextlib
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from utils import (
    dataset_matching_stats_to_df,
    dice_coeff,
    get_shades,
    intersection_over_union,
    invert_color,
    precision,
    recall,
)

###############
# PLOT CONFIG #
############### Data
DATA_PATH = Path.home() / "Desktop/Code/CELLSEG_BENCHMARK/"
############### Colormap and dark mode
COLORMAP_LIGHT = [
    "#4CC9F0",  # light blue - Stardist
    "#4361EE",  # dark blue - Cellpose
    "#7209B7",  # purple - SegRes
    "#F72585",  # pink - Swin
    # "#3A0CA3",  # dark blue - WNet3D
    "#FF4D00",  # dark orange - WNet3D
    ########### extra, not used
    # "#FF0000", # red
    # "#FF4D00", # dark orange
    "#FF7A00",
    "#F0A500",
    "#FFD700",
]
COLORMAP_DARK = [invert_color(color) for color in COLORMAP_LIGHT]
DARK_MODE = False
COLORMAP = COLORMAP_DARK if DARK_MODE else COLORMAP_LIGHT
# expanded colormap has darker and lighter shades for each original color (see get_shades in utils.py)
# See intensity parameter in get_shades to adjust the intensity of the shades
EXPANDED_COLORMAP = []
for color in COLORMAP[:4]:
    darker, lighter = get_shades(color)
    EXPANDED_COLORMAP.extend([darker, color, lighter])
EXPANDED_COLORMAP.extend(COLORMAP[4:])
################ Plot settings
DPI = 200
FONT_SIZE = 20
TITLE_FONT_SIZE = np.floor(FONT_SIZE * 1.25)
LABEL_FONT_SIZE = np.floor(FONT_SIZE * 1)
LEGEND_FONT_SIZE = np.floor(FONT_SIZE * 0.75)
BBOX_TO_ANCHOR = (1.05, 1)
LOC = "best"
################


def show_params():
    print("Plot parameters (set in plots.py) : \n- COLORMAP : ", end="")
    # print colormap with print statement colored with the colormap
    for color in COLORMAP:
        print(
            f"\033[38;2;{int(color[1:3], 16)};{int(color[3:5], 16)};{int(color[5:], 16)}m█\033[0m",
            end="",
        )
    print(
        f"\n- DPI : {DPI}\n- Data path : {DATA_PATH}\n- Font size : {FONT_SIZE}\n- Title font size : {TITLE_FONT_SIZE}\n- Label font size : {LABEL_FONT_SIZE}"
    )


def get_style_context():
    """Used to render plots with a custom palette and in dark mode if DARK_MODE is True, else in regular mode."""
    sns.set_palette(COLORMAP)
    if DARK_MODE:
        return plt.style.context("dark_background")
    return contextlib.nullcontext()


###################
# PLOT FUNCTIONS  #
###################


def plot_model_performance_semantic(
    image, gt, name, threshold_range=None, print_max=True
):
    """Plot the Dice, IoU, precision and recall for a given model and threshold range, across the specified threshold range between 0 and 1."""
    if threshold_range is None:
        threshold_range = np.arange(0, 1, 0.025)

    dice_scores = []
    iou_scores = []
    precision_scores = []
    recall_scores = []
    for threshold in threshold_range:
        pred = np.where(image > threshold, 1, 0)
        dice_scores.append(dice_coeff(gt, pred))
        iou_scores.append(intersection_over_union(gt, pred))
        precision_scores.append(precision(gt, pred))
        recall_scores.append(recall(gt, pred))
    plt.figure(figsize=(7, 7))
    plt.plot(threshold_range, dice_scores, label="Dice")
    plt.plot(threshold_range, iou_scores, label="IoU")
    plt.plot(threshold_range, precision_scores, label="Precision")
    plt.plot(threshold_range, recall_scores, label="Recall")
    # draw optimal threshold at max Dice score
    optimal_threshold = threshold_range[np.argmax(dice_scores)]
    plt.axvline(optimal_threshold, color="black", linestyle="--")
    # label line as optimal threshold at the bottom
    plt.text(
        optimal_threshold - 0.25,
        0,
        f"Max Dice @ {optimal_threshold:.2f}",
        verticalalignment="bottom",
    )
    plt.xlabel("Threshold")
    plt.ylabel("Score")
    plt.title(f"Model performance for {name}")
    plt.legend(fontsize=LEGEND_FONT_SIZE)
    plt.show()

    if print_max:
        print(
            f"Max Dice of {np.max(dice_scores):.2f} @ {threshold_range[np.argmax(dice_scores)]:.2f}"
        )
        print(
            f"Max IoU of {np.max(iou_scores):.2f} @ {threshold_range[np.argmax(iou_scores)]:.2f}"
        )

    return dice_scores, iou_scores, precision_scores, recall_scores


def plot_performance(
    taus,
    stats,
    name,
    metric="IoU",
    stats_list=(
        "precision",
        "recall",
        "accuracy",
        "f1",
        "mean_true_score",
        "mean_matched_score",
        "panoptic_quality",
    ),
):
    with get_style_context():
        sns.set_palette(COLORMAP)
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6), dpi=DPI)
        fig.suptitle(name, fontsize=TITLE_FONT_SIZE)
        stats = dataset_matching_stats_to_df(stats)
        for m in stats_list:
            sns.lineplot(
                data=stats,
                x="thresh",
                y=m,
                ax=ax1,
                label=m,
                lw=2,
                marker="o",
            )
        ax1.set(
            xlim=(0.05, 0.95),
            ylim=(-0.1, 1.1),
            xticks=np.arange(0.1, 1, 0.1),
        )
        ax1.set_xlabel(
            f"{metric}" + r" threshold $\tau$", fontsize=LABEL_FONT_SIZE
        )
        ax1.set_ylabel("Metric value", fontsize=LABEL_FONT_SIZE)
        ax1.spines["right"].set_visible(False)
        ax1.spines["top"].set_visible(False)
        ax1.tick_params(axis="both", which="major", labelsize=LEGEND_FONT_SIZE)
        ax1.grid(False)
        ax1.legend(
            fontsize=LEGEND_FONT_SIZE,
            # bbox_to_anchor=BBOX_TO_ANCHOR,
            loc="best",
        )

        for m in ("fp", "tp", "fn"):
            sns.lineplot(
                data=stats,
                x="thresh",
                y=m,
                ax=ax2,
                label=m,
                lw=2,
                marker="o",
            )
        ax2.set(xlim=(0.05, 0.95), xticks=np.arange(0.1, 1, 0.1))
        # ax2.set_ylim(0, max([stats['tp'].max(), stats['fp'].max(), stats['fn'].max()]))
        ax2.set_xlabel(
            f"{metric}" + r" threshold $\tau$", fontsize=LABEL_FONT_SIZE
        )
        ax2.set_ylabel("Number #", fontsize=LABEL_FONT_SIZE)
        ax2.spines["right"].set_visible(False)
        ax2.spines["top"].set_visible(False)
        ax2.tick_params(axis="both", which="major", labelsize=LEGEND_FONT_SIZE)
        ax2.grid(False)
        ax2.legend(
            fontsize=LEGEND_FONT_SIZE,
            # bbox_to_anchor=BBOX_TO_ANCHOR,
            loc="best",
        )

        sns.despine(
            left=False,
            right=True,
            bottom=False,
            top=True,
            trim=True,
            offset={"bottom": 40, "left": 15},
        )

        return fig


def plot_stat_comparison(
    taus, stats_list, model_names, stat="f1", metric="IoU"
):
    """Compare one stat for several models on a single plot."""
    with get_style_context():
        sns.set_palette(COLORMAP)
        fig, ax = plt.subplots(1, 1, figsize=(12, 6), dpi=DPI)
        stat_title = (stat[0].upper() + stat[1:]).replace("_", " ")
        fig.suptitle(f"{stat_title} comparison", fontsize=TITLE_FONT_SIZE)
        stats_list = [
            dataset_matching_stats_to_df(stats) for stats in stats_list
        ]
        for i, stats in enumerate(stats_list):
            sns.lineplot(
                data=stats,
                x=taus,
                y=stat,
                ax=ax,
                label=model_names[i],
                lw=2,
                marker="o",
            )
        ax.set_xlim(xmin=0.05, xmax=0.95)
        ax.set_ylim(ymin=-0, ymax=1)
        ax.tick_params(axis="both", which="major", labelsize=LEGEND_FONT_SIZE)
        ax.set_xticks(np.arange(0.1, 1, 0.1))
        ax.set_xlabel(
            f"{metric}" + r" threshold $\tau$", fontsize=LABEL_FONT_SIZE
        )
        ax.set_ylabel(stat_title, fontsize=LABEL_FONT_SIZE)
        ax.spines["right"].set_visible(False)
        ax.spines["top"].set_visible(False)
        sns.despine(
            left=False,
            right=True,
            bottom=False,
            top=True,
            trim=True,
            offset={"bottom": 40, "left": 15},
        )
        ax.grid(False)
        # legend to right (outside) of plot
        ax.legend(fontsize=FONT_SIZE, bbox_to_anchor=BBOX_TO_ANCHOR, loc=LOC)
        # return fig


def plot_stat_comparison_fold(
    fold_df, stat="f1", metric="IoU", colormap=COLORMAP
):
    """Compare one stat for several models on a single plot.
    Args:
        fold_df: DataFrame with the stats for each model
        stat: Statistic to plot
        metric: Metric used for the threshold
        colormap: Colormap to use for the plot
    """
    with get_style_context():
        sns.set_palette(colormap)
        fig, ax = plt.subplots(1, 1, figsize=(6, 6), dpi=DPI)
        # make background transparent

        stat_title = (stat[0].upper() + stat[1:]).replace("_", " ")
        # fig.suptitle(f"{stat_title} comparison", fontsize=TITLE_FONT_SIZE)
        sns.lineplot(
            data=fold_df,
            x="thresh",
            y=stat,
            ax=ax,
            hue="Model",
            lw=2,
            marker="o",
            estimator="mean",
            errorbar=("ci", 50),
        )
        ax.set_xlim(xmin=0.05, xmax=0.95)
        ax.set_ylim(ymin=-0, ymax=1)
        ax.tick_params(axis="both", which="major", labelsize=LEGEND_FONT_SIZE)
        ax.set_xticks(np.arange(0.1, 1, 0.1))
        ax.set_xlabel(
            f"{metric}" + r" threshold $\tau$", fontsize=LABEL_FONT_SIZE
        )
        ax.set_ylabel(stat_title, fontsize=LABEL_FONT_SIZE)
        ax.spines["right"].set_visible(False)
        ax.spines["top"].set_visible(False)
        sns.despine(
            left=False,
            right=True,
            bottom=False,
            top=True,
            trim=True,
            offset={"bottom": 40, "left": 15},
        )
        ax.grid(False)
        # legend to right (outside) of plot
        legend = ax.legend(
            fontsize=FONT_SIZE, bbox_to_anchor=BBOX_TO_ANCHOR, loc="upper left"
        )
        legend.get_frame().set_alpha(0)
        fig.patch.set_alpha(0)
        ax.patch.set_alpha(0)
        # return fig


def plot_losses(losses_df, loss_keys):
    """Plot the losses for the model."""
    with get_style_context():
        sns.set_palette(COLORMAP)
        fig, ax = plt.subplots(1, 1, figsize=(6, 6), dpi=DPI)
        ax2 = ax.twinx()
        axes = [ax, ax2]
        lines = []
        labels = []

        for i, key in enumerate(loss_keys):
            curr_ax = 0 if i == 0 else 1
            sns.lineplot(
                data=losses_df,
                x="Epoch",
                y=key,
                ax=axes[curr_ax],
                label=key,
                color=sns.color_palette()[i],
                legend=False,
                linewidth=3,
            )
            labels.append(key)
            if i != 1:
                lines.extend(axes[curr_ax].get_lines())

        ax.set_xlabel("Epoch", fontsize=LABEL_FONT_SIZE)
        # minor ticks on x axis
        ax.set_xticks(np.arange(0, len(losses_df), 1), minor=True)
        ax.set_ylabel("SoftNCuts loss", fontsize=LABEL_FONT_SIZE)
        ax2.set_ylabel(
            "Reconstruction loss\nWeighted sum of losses",
            fontsize=LABEL_FONT_SIZE,
        )

        ax.set_ylim(0.2, 1)
        ax.set_yticks(np.arange(0.2, 1.1, 0.1))
        ax2.set_ylim(10, 60)
        ax2.set_yticks(np.arange(10, 65, 5))

        for ax in axes:
            ax.spines["right"].set_visible(False)
            ax.spines["top"].set_visible(False)
            sns.despine(
                left=False,
                right=False,
                bottom=False,
                top=True,
                trim=True,
                offset={"bottom": 40, "left": 15},
            )
            ax.grid(False)
            ax.patch.set_alpha(0)

        legend = fig.legend(
            lines, labels, loc="lower right", bbox_to_anchor=BBOX_TO_ANCHOR
        )
        legend.get_frame().set_alpha(0)
        fig.patch.set_alpha(0)
