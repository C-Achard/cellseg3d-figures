import sys
from pathlib import Path

from tifffile import imread, imwrite

sys.path.append("../../..")

DATA = Path.home() / "Desktop/Code/CELLSEG_BENCHMARK/TPH2_mesospim/TRAINING"
SPLITS = [10, 20, 40, 60, 80]


def create_training_data_folders(source_folder="ALL", target_folder="SPLITS"):
    import utils

    data = DATA / source_folder
    if not data.is_dir():
        raise FileNotFoundError(f"Data folder {data} not found")
    print(
        f"Creating training data folders from {data} to {DATA/target_folder}"
    )
    data_stats, labels_uniques = utils.create_training_dataframe(
        data, ".", "labels/", "visual/", "labels/visual/"
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
            print(f"Saving \n{vol_name} to \n{target_folder_s}")
            imwrite(target_folder_s / vol_name, vol)
        for label_path in selected_labels:
            label = imread(label_path)
            label_name = Path(label_path).name
            print(f"Saving \n{label_name} to \n{target_folder_lab}")
            imwrite(target_folder_lab / label_name, label)
        print("_" * 30)


if __name__ == "__main__":
    create_training_data_folders()
