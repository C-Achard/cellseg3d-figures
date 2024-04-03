import colorsys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

###################
# UTILS FUNCTIONS #
###################


def hex_to_rgb(hex_color):
    """Convert a color from hexadecimal to RGB."""
    hex_color = hex_color.lstrip("#")
    return tuple(int(hex_color[i : i + 2], 16) for i in (0, 2, 4))


def rgb_to_hex(rgb_color):
    """Convert a color from RGB to hexadecimal."""
    return f"#{''.join([f'{c:02X}' for c in rgb_color])}"


def get_shades(
    hex_color,
    saturation_intensity=0.1,
    value_intensity=0.25,
    prevent_clipping=False,
):
    """Generate a shade less saturated and a shade more saturated of a color in hexadecimal format.

    Args:
        hex_color (str): A color in hexadecimal format.
        saturation_intensity (float): Intensity of the saturation change. Default is 0.1.
        value_intensity (float): Intensity of the value change. Default is 0.35.
        prevent_clipping (bool): If True, prevents the saturation and value from being clipped (outside [0, 1] range). Default is True.
    Returns:
        less_saturated_hex_color (str): A shade less saturated of the color in hexadecimal format.
        more_saturated_hex_color (str): A shade more saturated of the color in hexadecimal format.
    """

    hex_color = hex_color.lstrip("#")
    rgb_color = tuple(int(hex_color[i : i + 2], 16) / 255.0 for i in (0, 2, 4))
    hsv_color = colorsys.rgb_to_hsv(*rgb_color)

    if (
        max(0, hsv_color[1] - saturation_intensity) == 0
        or min(1, hsv_color[1] + saturation_intensity) == 1
    ):
        print(
            f"Warning: Saturation in {hsv_color[1]} is too low or too high in hex color {hex_color}"
        )
        if prevent_clipping:
            hsv_color = (hsv_color[0], 0.5, hsv_color[2])

        if (
            max(0, hsv_color[2] - value_intensity) == 0
            or min(1, hsv_color[2] + value_intensity) == 1
        ):
            print(
                f"Warning: Value in {hsv_color[2]} is too low or too high in hex color {hex_color}"
            )
            if prevent_clipping:
                hsv_color = (hsv_color[0], hsv_color[1], 0.5)

    lower_hsv_color = (
        hsv_color[0],
        max(0, hsv_color[1] - saturation_intensity),
        max(0, hsv_color[2] - value_intensity),
    )
    lower_rgb_color = colorsys.hsv_to_rgb(*lower_hsv_color)
    lower_hex_color = "#%02x%02x%02x" % tuple(
        int(c * 255) for c in lower_rgb_color
    )

    higher_hsv_color = (
        hsv_color[0],
        min(1, hsv_color[1] + saturation_intensity),
        min(1, hsv_color[2] + value_intensity),
    )
    higher_rgb_color = colorsys.hsv_to_rgb(*higher_hsv_color)
    higher_hex_color = "#%02x%02x%02x" % tuple(
        int(c * 255) for c in higher_rgb_color
    )

    return lower_hex_color, higher_hex_color


def invert_color(hex_color):
    """Invert a color from hexadecimal to its complementary color."""
    rgb = hex_to_rgb(hex_color)
    inverted_rgb = tuple(255 - c for c in rgb)
    return rgb_to_hex(inverted_rgb)


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


def models_stats_to_df(model_stats, names):
    models_dfs = {}
    for model in model_stats:
        models_dfs[names[model_stats.index(model)]] = pd.DataFrame(
            model
        ).set_index("thresh")
    return models_dfs


def dataset_matching_stats_to_df(dataset_matching_stats):
    return pd.DataFrame(dataset_matching_stats).set_index("thresh")


def extract_losses_from_log(path_to_log):
    """Reads a log file and looks for :
    - epoch ("Epoch e of E")
    - NCuts ("NCuts loss: x")
    - Reconstruction ("Reconstruction loss: x")
    - Sum of losses ("Weighted sum of losses: x")
    - Total number of epochs (E)
    These are saved in a dict where the key is the epoch number and the value is a dict containing the losses.
    """
    file = Path.open(Path(path_to_log).resolve(), "r")
    if not file:
        raise FileNotFoundError("Log file not found")
    lines = file.readlines()
    file.close()
    losses = {}
    total_epochs = -1

    for line in lines:
        if "Epoch" in line:
            if "Epochs" in line:
                total_epochs = int(line.split(" ")[1])
                continue
            epoch = int(line.split(" ")[1])
            losses[epoch] = {}
        if "Ncuts loss:" in line:
            losses[epoch]["Ncuts"] = float(line.split(":")[1])
        if "Reconstruction loss:" in line:
            losses[epoch]["Reconstruction"] = float(line.split(":")[1])
        if "Weighted sum of losses:" in line:
            losses[epoch]["Sum"] = float(line.split(":")[1])

    return losses, total_epochs
