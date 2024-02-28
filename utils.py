from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

"""
Utilities for the figures : colormap, path to data, etc.
"""

####################
# GLOBAL VARIABLES #
####################

COLORMAP = "mako"
DPI = 200
DATA_PATH = Path.home() / "Desktop/Code/CELLSEG_BENCHMARK/"
FONT_SIZE = 15
TITLE_FONT_SIZE = int(FONT_SIZE * 1.75)
LABEL_FONT_SIZE = int(FONT_SIZE * 1.25)

print(
    f"Plot parameters : \n- Colormap : {COLORMAP}\n- DPI : {DPI}\n- Data path : {DATA_PATH}\n- Font size : {FONT_SIZE}\n- Title font size : {TITLE_FONT_SIZE}\n- Label font size : {LABEL_FONT_SIZE}"
)

###################
# UTILS FUNCTIONS #
###################


def dice_coeff(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> np.float64:
    """Compute Dice-Sorensen coefficient between two numpy arrays
    Args:
        y_true: Ground truth label
        y_pred: Prediction label
    Returns: dice coefficient.
    """
    sum_tensor = np.sum
    smooth = 1.0
    y_true_f = y_true.flatten()
    y_pred_f = y_pred.flatten()
    intersection = sum_tensor(y_true_f * y_pred_f)
    return (2.0 * intersection + smooth) / (
        sum_tensor(y_true_f) + sum_tensor(y_pred_f) + smooth
    )


def intersection_over_union(
    y_true: np.ndarray, y_pred: np.ndarray
) -> np.float64:
    """Compute Intersection over Union between two numpy arrays
    Args:
        y_true: Ground truth label
        y_pred: Prediction label
    Returns: IoU.
    """
    y_true_f = y_true.flatten()
    y_pred_f = y_pred.flatten()
    intersection = np.sum(y_true_f * y_pred_f)
    union = np.sum(y_true_f) + np.sum(y_pred_f) - intersection
    return intersection / union


def precision(y_true: np.ndarray, y_pred: np.ndarray) -> np.float64:
    """Compute precision between two numpy arrays
    Args:
        y_true: Ground truth label
        y_pred: Prediction label
    Returns: precision.
    """
    y_true_f = y_true.flatten()
    y_pred_f = y_pred.flatten()
    intersection = np.sum(y_true_f * y_pred_f)
    return intersection / np.sum(y_pred_f)


def recall(y_true: np.ndarray, y_pred: np.ndarray) -> np.float64:
    """Compute recall between two numpy arrays
    Args:
        y_true: Ground truth label
        y_pred: Prediction label
    Returns: recall.
    """
    y_true_f = y_true.flatten()
    y_pred_f = y_pred.flatten()
    intersection = np.sum(y_true_f * y_pred_f)
    return intersection / np.sum(y_true_f)


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
    plt.legend()
    plt.show()

    if print_max:
        print(
            f"Max Dice of {np.max(dice_scores):.2f} @ {threshold_range[np.argmax(dice_scores)]:.2f}"
        )
        print(
            f"Max IoU of {np.max(iou_scores):.2f} @ {threshold_range[np.argmax(iou_scores)]:.2f}"
        )

    return dice_scores, iou_scores, precision_scores, recall_scores


def models_stats_to_df(model_stats, names):
    models_dfs = {}
    for model in model_stats:
        models_dfs[names[model_stats.index(model)]] = pd.DataFrame(
            model
        ).set_index("thresh")
    return models_dfs


def dataset_matching_stats_to_df(dataset_matching_stats):
    return pd.DataFrame(dataset_matching_stats).set_index("thresh")


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
    use_palette=True,
):
    if use_palette:
        sns.set_palette(COLORMAP)
    else:
        sns.set_palette("tab10")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6), dpi=DPI)
    fig.suptitle(name, fontsize=TITLE_FONT_SIZE)
    stats = dataset_matching_stats_to_df(stats)
    for m in stats_list:
        # ax1.plot(taus, [s._asdict()[m] for s in stats], '.-', lw=2, label=m)
        sns.lineplot(
            data=stats,
            x="thresh",
            y=m,
            ax=ax1,
            label=m,
            lw=2,
            marker="o",
            #  hue=m,
            #  palette=GLOBAL_COLORMAP
        )
    ax1.set(
        xlim=(0.05, 0.95),
        ylim=(-0.1, 1.1),
        xticks=np.arange(0.1, 1, 0.1),
    )
    ax1.set_xlabel(f"{metric}" + r" threshold $\tau$", fontsize=FONT_SIZE)
    ax1.set_ylabel("Metric value", fontsize=FONT_SIZE)
    ax1.spines["right"].set_visible(False)
    ax1.spines["top"].set_visible(False)
    ax1.tick_params(axis="both", which="major", labelsize=LABEL_FONT_SIZE)
    ax1.grid()
    ax1.legend(fontsize=FONT_SIZE)

    for m in ("fp", "tp", "fn"):
        # ax2.plot(taus, [s._asdict()[m] for s in stats], '.-', lw=2, label=m)
        sns.lineplot(
            data=stats,
            x="thresh",
            y=m,
            ax=ax2,
            label=m,
            lw=2,
            marker="o",
            # hue=m,
            # palette=GLOBAL_COLORMAP
        )
    ax2.set(xlim=(0.05, 0.95), xticks=np.arange(0.1, 1, 0.1))
    # ax2.set_ylim(0, max([stats['tp'].max(), stats['fp'].max(), stats['fn'].max()]))
    ax2.set_xlabel(f"{metric}" + r" threshold $\tau$", fontsize=FONT_SIZE)
    ax2.set_ylabel("Number #", fontsize=FONT_SIZE)
    ax2.spines["right"].set_visible(False)
    ax2.spines["top"].set_visible(False)
    ax2.tick_params(axis="both", which="major", labelsize=LABEL_FONT_SIZE)
    ax2.grid()
    ax2.legend(fontsize=FONT_SIZE)

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
    taus, stats_list, model_names, stat="f1", metric="IoU", use_palette=True
):
    """Compare one stat for several models on a single plot."""
    if use_palette:
        sns.set_palette(COLORMAP)
    else:
        sns.set_palette("tab10")
    fig, ax = plt.subplots(1, 1, figsize=(16, 6), dpi=DPI)
    stat_title = (stat[0].upper() + stat[1:]).replace("_", " ")
    fig.suptitle(f"{stat_title} comparison", fontsize=TITLE_FONT_SIZE)
    stats_list = [dataset_matching_stats_to_df(stats) for stats in stats_list]
    for i, stats in enumerate(stats_list):
        # ax.plot(taus, [s._asdict()[stat] for s in stats], '.-', lw=2, label=model_names[i])
        sns.lineplot(
            data=stats,
            x=taus,
            y=stat,
            ax=ax,
            label=model_names[i],
            lw=2,
            marker="o",
            # hue=stat,
            # palette=GLOBAL_COLORMAP
        )
    ax.set_xlim(xmin=0.05, xmax=0.95)
    ax.set_ylim(ymin=-0.1, ymax=0.9)
    ax.tick_params(axis="both", which="major", labelsize=LABEL_FONT_SIZE)
    ax.set_xticks(np.arange(0.1, 1, 0.1))
    ax.set_xlabel(f"{metric}" + r" threshold $\tau$", fontsize=FONT_SIZE)
    ax.set_ylabel(stat_title, fontsize=FONT_SIZE)
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
    ax.grid()
    ax.legend(fontsize=FONT_SIZE)
