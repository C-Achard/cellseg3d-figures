from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
"""
Utilities for the figures : colormap, path to data, etc.
"""



GLOBAL_COLORMAP = "mako"
GLOBAL_DPI = 300
GLOBAL_DATA_PATH = Path.home() / "Desktop/Code/CELLSEG_BENCHMARK/"
GLOBAL_FONT_SIZE = 12

def dice_coeff(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> np.float64:
    """Compute Dice-Sorensen coefficient between two numpy arrays
    Args:
        y_true: Ground truth label
        y_pred: Prediction label
    Returns: dice coefficient
    """
    sum_tensor = np.sum
    smooth = 1.0
    y_true_f = y_true.flatten()
    y_pred_f = y_pred.flatten()
    intersection = sum_tensor(y_true_f * y_pred_f)
    score = (2.0 * intersection + smooth) / (
        sum_tensor(y_true_f) + sum_tensor(y_pred_f) + smooth
    )
    return score


def intersection_over_union(
    y_true: np.ndarray, y_pred: np.ndarray
) -> np.float64:
    """Compute Intersection over Union between two numpy arrays
    Args:
        y_true: Ground truth label
        y_pred: Prediction label
    Returns: IoU
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
    Returns: precision
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
    Returns: recall
    """
    y_true_f = y_true.flatten()
    y_pred_f = y_pred.flatten()
    intersection = np.sum(y_true_f * y_pred_f)
    return intersection / np.sum(y_true_f)

def plot_model_performance_semantic(
    image, gt, name, threshold_range=None, print_max=True
):
    """Plot the Dice, IoU, precision and recall for a given model and threshold range, across the specified threshold range between 0 and 1"""
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