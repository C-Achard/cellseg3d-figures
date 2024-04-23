import sys
from pathlib import Path

import numpy as np
from tifffile import imread, imwrite

sys.path.append("../../..")

# DATA = Path("/data/cyril/CELLSEG_BENCHMARK/TPH2_mesospim/TRAINING")
DATA = Path.home() / "Desktop/Code/CELLSEG_BENCHMARK/TPH2_mesospim/TRAINING"
SPLITS = [10, 20, 40, 80]


def create_training_data_folders(source_folder="ALL", target_folder="SPLITS"):
    import utils

    data = DATA / source_folder
    if not data.is_dir():
        raise FileNotFoundError(f"Data folder {data} not found")
    print(
        f"Creating training data folders from {data} to {DATA/target_folder}"
    )
    data_stats, labels_uniques = utils.create_training_dataframe(
        data, ".", "labels/", "visual_crops/", "labels/visual_labels/"
    )
    training_data_stats = data_stats[data_stats.training_data]
    for s in SPLITS:
        selected_volumes, selected_labels = utils.select_volumes(
            training_data_stats,
            s,
        )
        target_folder_s = DATA / target_folder / str(s)
        target_folder_s.mkdir(parents=True, exist_ok=False)
        target_folder_lab = target_folder_s / "labels"
        target_folder_lab.mkdir(parents=True, exist_ok=False)
        for volume_path in selected_volumes:
            vol = imread(volume_path)
            vol_name = Path(volume_path).name
            if (target_folder_s / vol_name).is_file():
                print(f"File {vol_name} already exists. Skipping")
                continue
            print(f"Saving \n{vol_name} to \n{target_folder_s}")
            imwrite(target_folder_s / vol_name, vol)
        for label_path in selected_labels:
            label = imread(label_path)
            label_name = Path(label_path).name
            if (target_folder_lab / label_name).is_file():
                print(f"File {label_name} already exists. Skipping")
                continue
            print(f"Saving \n{label_name} to \n{target_folder_lab}")
            imwrite(target_folder_lab / label_name, label)
        print("_" * 30)


def create_semantic_labels(source_folder):
    """Binarizes labels and saves them to a semantic folder in the labels folder"""
    labels = DATA / source_folder
    if not labels.is_dir():
        raise FileNotFoundError(f"Labels folder {labels} not found")
    print(f"Creating semantic labels from {labels}")
    semantic_labels = labels / "semantic"
    semantic_labels.mkdir(parents=True, exist_ok=False)
    for label_path in labels.glob("*.tif"):
        label = imread(label_path)
        label_name = Path(label_path).name
        if (semantic_labels / label_name).is_file():
            print(f"File {label_name} already exists. Skipping")
            continue
        print(f"Saving \n{label_name} to \n{semantic_labels}")
        imwrite(
            semantic_labels / label_name,
            np.where(label > 0, 1, 0).astype(np.uint16),
        )


if __name__ == "__main__":
    create_training_data_folders()
    for split in SPLITS:
        create_semantic_labels(DATA / "SPLITS" / str(split) / "labels")
