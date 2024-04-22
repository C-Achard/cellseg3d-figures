import colorsys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from tifffile import imread

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


def get_n_shades(
    hex_color,
    n_shades=5,
):
    """Generate n shades of a color in hexadecimal format.

    Color is obtained as hsv, then n evenly spaced values are generated for the value channel.
    """
    hsv_color = colorsys.rgb_to_hsv(*hex_to_rgb(hex_color))
    shades = []
    values = np.linspace(0, 1, n_shades)
    for value in values:
        rgb_color = colorsys.hsv_to_rgb(hsv_color[0], hsv_color[1], value)
        shades.append(rgb_to_hex(tuple(int(c * 255) for c in rgb_color)))
    return shades


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


def add_bbox_to_viewer(viewer, array, name="bbox", color="white"):
    """Adds a bounding box to a napari viewer, shaped after the input array."""
    bbox_layer = viewer.add_image(np.zeros_like(array), name=name)
    bbox_layer.bounding_box.visible = True
    bbox_layer.bounding_box.line_color = color
    bbox_layer.bounding_box.points = False
    # bbox_layer.bounding_box.blending = "additive"


def take_napari_screenshots(viewer, save_path, bbox_layer_name="bbox"):
    """Iterates over all layers in a napari viewer and saves a screenshot of each layer. Skips the bbox layer."""
    for layer in viewer.layers:
        if layer.name == bbox_layer_name:
            continue
        for l in viewer.layers:
            if l.name == bbox_layer_name:
                continue
            l.visible = False
        layer.visible = True
        print(f"Taking screenshot of {layer.name}")
        viewer.screenshot(
            str(Path(save_path) / f"{layer.name}_screenshot.png")
        )
        # wait 1s to avoid overlapping screenshots
        time.sleep(1)

        #     for layer in viewer.layers:
        # print(layer.name)
        # if layer.name == bbox_layer_name:
        #     continue
        # for layer in viewer.layers:
        #     if layer.name == bbox_layer_name:
        #         continue
        #     layer.visible = False
        # viewer.layers[layer.name].visible = True
        # viewer.screenshot(Path(save_path, f"{layer.name}_screenshot.png"))
        # # wait 1s to avoid overlapping screenshots
        # time.sleep(1)


def _get_images(path, pattern="*.tif"):
    return list(path.glob(pattern))


def _label_count(label_path):
    """Count the number of labels in a label image."""
    label = imread(label_path)
    return len(np.unique(label)) - 1


def _find_origin_volume(paths, volumes=None):
    """Find the origin volume of a given image path."""
    volumes = (
        ["c1", "c2", "c3", "c4", "c5", "visual"]
        if volumes is None
        else volumes
    )
    return [volume for volume in volumes if volume in paths][0]


def _find_unique_labels_in_volume(data_df):
    """Find unique labels per crop of origin volume"""
    volumes_unique_labels = {}
    for volumes in data_df.origin_volume.unique():
        uniques_volume = np.array([])
        for _, row in data_df[data_df.origin_volume == volumes].iterrows():
            label = imread(row["path_label"])
            uniques_volume = np.concatenate((uniques_volume, np.unique(label)))
        volumes_unique_labels[volumes] = (
            len(np.unique(uniques_volume).flatten()) - 1
        )
    return volumes_unique_labels


def create_training_dataframe(
    data_path,
    training_path,
    training_labels_path,
    test_path,
    test_labels_path,
    test_data_name="visual",
):
    """Create a dataframe with the paths to the training and test images and their corresponding labels."""
    data_path = Path(data_path)
    training_files = _get_images(data_path / training_path)
    training_labels = _get_images(data_path / training_labels_path)
    test_files = _get_images(data_path / test_path)
    test_labels = _get_images(data_path / test_labels_path)

    if not all(
        [
            len(training_files),
            len(training_labels),
            len(test_files),
            len(test_labels),
        ]
    ):
        raise ValueError(
            f"Empty paths found: {training_files}, {training_labels}, {test_files}, {test_labels}"
        )

    data_stats = pd.DataFrame(
        columns=[
            "path_image",
            "path_label",
            "origin_volume",
            "label_count",
            "training_data",
        ]
    )
    data_stats["path_image"] = [str(file) for file in training_files]
    data_stats["path_label"] = [str(file) for file in training_labels]
    data_stats["training_data"] = [True for file in training_files]
    # append entries for test data
    data_stats = pd.concat(
        [
            data_stats,
            pd.DataFrame(
                columns=data_stats.columns,
                data={
                    "path_image": [str(file) for file in test_files],
                    "path_label": [str(file) for file in test_labels],
                    "training_data": [False for file in test_files],
                },
            ),
        ]
    )
    data_stats["origin_volume"] = data_stats["path_image"].apply(
        _find_origin_volume
    )
    data_stats["name"] = data_stats["path_image"].apply(lambda x: Path(x).stem)
    data_stats["label_count"] = (
        data_stats["path_label"].apply(_label_count).astype(int)
    )
    labels_uniques = _find_unique_labels_in_volume(data_stats)
    labels_uniques = pd.DataFrame.from_dict(
        labels_uniques, orient="index", columns=["unique_labels"]
    )
    labels_uniques["percentage"] = (
        labels_uniques["unique_labels"] / labels_uniques["unique_labels"].sum()
    )
    labels_uniques["training_data"] = [
        i != test_data_name for i in labels_uniques.index
    ]

    return data_stats, labels_uniques


def _knapsack(items, max_weight):
    """Simple solver for the knapsack problem."""
    num_items = len(items)
    table = [[0 for _ in range(max_weight + 1)] for _ in range(num_items + 1)]

    # Convert the dictionary items to lists for easier indexing
    paths, weights = zip(*items.items())

    for i in range(num_items + 1):
        for w in range(max_weight + 1):
            if i == 0 or w == 0:
                table[i][w] = 0
            elif weights[i - 1] <= w:
                table[i][w] = max(
                    weights[i - 1] + table[i - 1][w - weights[i - 1]],
                    table[i - 1][w],
                )
            else:
                table[i][w] = table[i - 1][w]

    selected_paths = []
    w = max_weight
    for i in range(num_items, 0, -1):
        if table[i][w] != table[i - 1][w]:
            selected_paths.append(paths[i - 1])
            w -= weights[i - 1]

    return selected_paths


def select_volumes(data_stats, percentage, tolerance=0.5, verbose=True):
    """
    Selects the volumes that sum up to the desired percentage of the total cell count within the tolerance.

    Args:
        data_stats (pd.DataFrame): DataFrame with the data statistics.
        percentage (float): Desired percentage of the total cell count.
        tolerance (float): Tolerance for the cell count.
        verbose (bool): Print the selected volumes and cell count.

    Returns:
        list: Selected volumes.
    """
    total_cell_count = data_stats["label_count"].sum().astype(int)
    desired_cell_count = np.floor(total_cell_count * percentage / 100).astype(
        int
    )
    cell_count_tolerance = np.floor(total_cell_count * tolerance / 100).astype(
        int
    )
    if verbose:
        print(f"Total cell count: {total_cell_count}")
        print(f"Desired cell count: {desired_cell_count}")
        print(f"Cell count tolerance: {cell_count_tolerance}")

    data_stats_sorted = data_stats.sort_values("label_count", ascending=False)

    # Create a dictionary where each key is a path and each value is the corresponding label count
    path_dict = data_stats_sorted.set_index("path_image")[
        "label_count"
    ].to_dict()

    # Find the most optimal combination of volumes that sum up to the desired cell count within the tolerance
    selected_volumes = _knapsack(path_dict, desired_cell_count)
    # find selected labels directly from the dataframe using image paths
    selected_labels = data_stats[
        data_stats["path_image"].isin(selected_volumes)
    ]["path_label"].tolist()

    # If not found, scan within the tolerance
    if len(selected_volumes) == 0:
        print(
            "No proper combination found. Trying within tolerance values. Consider increasing the tolerance value if occurs frequently."
        )
        tolerance_values = np.arange(
            desired_cell_count - cell_count_tolerance,
            desired_cell_count + cell_count_tolerance,
        ).astype(int)
        for cell_count_tolerance_value in tolerance_values:
            print(f"Trying with tolerance value: {cell_count_tolerance_value}")
            selected_volumes = _knapsack(path_dict, cell_count_tolerance_value)
            selected_labels = data_stats[
                data_stats["path_image"].isin(selected_volumes)
            ]["path_label"].tolist()
            if len(selected_volumes) > 0:
                break
    if verbose:
        selected_cell_count = sum(path_dict[path] for path in selected_volumes)
        print(
            f"Selected {len(selected_volumes)} volumes with {selected_cell_count} cells."
        )

    return selected_volumes, selected_labels
