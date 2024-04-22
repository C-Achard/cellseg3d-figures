# --------- REQUIRES A SEPARATE CONDA ENVIRONMENT WITH CELLPOSE INSTALLED --------- #

import pathlib as pt
from glob import glob
from os import environ

import numpy as np
import torch
from cellpose.models import CellposeModel
from tifffile import imread

CELL_MEAN_DIAM = 3.3
DATA_PATH = (
    pt.Path.home()
    / "Desktop/Code/CELLSEG_BENCHMARK/TPH2_mesospim/TRAINING/SPLITS"
)

NUM_EPOCHS = 50
TRAINING_PERCENTAGES = [10, 20, 40, 60, 80]
SEEDS = [34936339, 34936397, 34936345]


def convert_2d(images_array, images_names=None, dtype=np.float32):
    images_2d = []
    images_names_2d = [] if images_names is not None else None
    for i, image in enumerate(images_array):
        for j, slice_ in enumerate(image):
            images_2d.append(slice_.astype(dtype))
            if images_names is not None:
                images_names_2d.append(
                    f"{pt.Path(images_names[i]).stem}_{j}.tif"
                )
    return images_2d, images_names_2d


def train_cellpose(path_images, seed, save_path):
    """This code was used to train Cellpose for the Label efficiency figure"""

    np.random.seed(seed)
    torch.manual_seed(seed)
    # Enable CUDA deterministic operations
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    environ["PYTHONHASHSEED"] = "0"
    environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    # environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'

    X_paths = sorted(glob(str(path_images / "*.tif")))
    Y_paths = sorted(glob(str(path_images / "labels/*.tif")))

    X = list(map(imread, X_paths))
    Y = list(map(imread, Y_paths))
    X_2d, X_paths_2d = convert_2d(X, X_paths)
    Y_2d, Y_paths_2d = convert_2d(Y, Y_paths, dtype=np.uint16)
    print(len(X_2d))
    print(len(Y_2d))
    assert len(X_2d) == len(Y_2d)

    # ind = range(len(X_2d))
    # n_val = max(1, int(round(VAL_PERCENT * len(ind))))
    # ind_train, ind_val = ind[:-n_val], ind[-n_val:]
    # X_val, Y_val = [X_2d[i] for i in ind_val], [Y_2d[i] for i in ind_val]
    X_trn, Y_trn = X_2d, Y_2d
    print("number of images: %3d" % len(X))
    print("- training:       %3d" % len(X_trn))
    # print("- validation:     %3d" % len(X_val))

    X_trn_paths = [X_paths_2d[i] for i in range(len(X_trn))]
    # X_val_paths = [X_paths_2d[i] for i in ind_val]
    print("Train :")
    [print(p) for p in X_trn_paths]
    # print("Val :")
    # [print(p) for p in X_val_paths]
    print("*" * 20)

    print("Parameters :")
    save_path.mkdir(exist_ok=False, parents=True)
    # print(f"VAL_PERCENT : {VAL_PERCENT}")
    print(f"SAVE_NAME : {save_path}")
    print(f"Path images : {path_images}")
    print(f"cell_mean_diam : {CELL_MEAN_DIAM}")
    print("*" * 20)

    model = CellposeModel(
        gpu=True,
        pretrained_model=False,
        model_type=None,
        diam_mean=CELL_MEAN_DIAM,  # 3.3,
        # nchan=1,
    )
    save_file = f"{path_images.parts[-1]}_{seed}.cellpose"
    model.train(
        train_data=X_trn,
        train_labels=Y_trn,
        train_files=X_trn_paths,
        # test_data=X_val,
        # test_labels=Y_val,
        # test_files=X_val_paths,
        save_path=str(save_path),
        save_every=10,
        n_epochs=50,
        channels=[0, 0],
        model_name=save_file,
    )

    # check if outputs exist
    if not (save_path / save_file).is_file():
        raise ValueError(
            f"Training weights not found in {save_path}. Aborting"
        )


if __name__ == "__main__":
    for seed in SEEDS:
        for percentage in TRAINING_PERCENTAGES:
            path_images = DATA_PATH / str(percentage)
            save_path = pt.Path(f"./cellpose_{percentage}_{seed}")
            train_cellpose(path_images, seed, save_path)
