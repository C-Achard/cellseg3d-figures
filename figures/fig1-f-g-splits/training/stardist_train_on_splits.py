from __future__ import (
    absolute_import,
    division,
    print_function,
    unicode_literals,
)

import pathlib as pt
import shutil
import sys
from glob import glob
from os import environ

import numpy as np
import tensorflow as tf
from csbdeep.utils import Path, normalize
from stardist import (
    Rays_GoldenSpiral,
    calculate_extents,
    fill_label_holes,
    gputools_available,
    random_label_cmap,
)
from stardist.matching import matching, matching_dataset
from stardist.models import Config3D, StarDist3D, StarDistData3D
from tifffile import imread
from tqdm import tqdm

DATA_PATH = Path.home() / "Desktop/Code/CELLSEG_BENCHMARK/TPH2_mesospim/SPLITS"

NUM_EPOCHS = 50
TRAINING_PERCENTAGES = [10, 20, 60, 80]
SPLITS = ["1_c15", "2_c1_c4_visual", "3_c1245_visual"]
SEED = 34936339


def train_stardist(path_images, train_percentage, seed=SEED):
    # Set seeds for numpy & tensorflow
    np.random.seed(seed)
    tf.random.set_seed(seed)
    environ["PYTHONHASHSEED"] = "0"
    environ["TF_DETERMINISTIC_OPS"] = "1"
    # tf.config.threading.set_inter_op_parallelism_threads(1)
    # tf.config.threading.set_intra_op_parallelism_threads(1)

    X_paths = sorted(glob(str(path_images / "*.tif")))
    Y_paths = sorted(glob(str(path_images / "labels/*.tif")))
    # assert all(Path(x).name==Path(y).name for x,y in zip(X,Y))
    X = list(map(imread, X_paths))
    Y = list(map(imread, Y_paths))
    # Y = [Yi.astype(np.uint16) for Yi in Y]
    n_channel = 1 if X[0].ndim == 3 else X[0].shape[-1]
    axis_norm = (0, 1, 2)  # normalize channels independently
    # axis_norm = (0,1,2,3) # normalize channels jointly
    if n_channel > 1:
        print(
            "Normalizing image channels %s."
            % (
                "jointly"
                if axis_norm is None or 3 in axis_norm
                else "independently"
            )
        )
        sys.stdout.flush()

    X = [normalize(x, 1, 99.8, axis=axis_norm) for x in tqdm(X)]
    Y = [fill_label_holes(y) for y in tqdm(Y)]
    rng = np.random.RandomState(seed)
    ind = rng.permutation(len(X))
    print(ind)
    ind = range(len(X))
    print(ind)
    val_percent = 1 - train_percentage / 100
    n_val = max(1, int(round(val_percent * len(ind))))
    ind_train, ind_val = ind[:-n_val], ind[-n_val:]
    X_val, Y_val = [X[i] for i in ind_val], [Y[i] for i in ind_val]
    X_trn, Y_trn = [X[i] for i in ind_train], [Y[i] for i in ind_train]
    print("number of images: %3d" % len(X))
    print("- training:       %3d" % len(X_trn))
    print("- validation:     %3d" % len(X_val))
    print("Train files :")
    [print(X_paths[i]) for i in ind_train]
    print("Validation files :")
    [print(X_paths[i]) for i in ind_val]
    print("*" * 20)

    print(Config3D.__doc__)
    extents = calculate_extents(Y)
    anisotropy = tuple(np.max(extents) / extents)
    print("empirical anisotropy of labeled objects = %s" % str(anisotropy))
    # 96 is a good default choice (see 1_data.ipynb)
    n_rays = 96

    # Use OpenCL-based computations for data generator during training (requires 'gputools')
    # use_gpu = False and gputools_available()
    use_gpu = True

    # Predict on subsampled grid for increased efficiency and larger field of view
    # grid = tuple(1 if a > 1.5 else 2 for a in anisotropy)
    grid = (1, 1, 1)

    # Use rays on a Fibonacci lattice adjusted for measured anisotropy of the training data
    rays = Rays_GoldenSpiral(n_rays, anisotropy=anisotropy)

    conf = Config3D(
        rays=rays,
        grid=grid,
        anisotropy=anisotropy,
        use_gpu=use_gpu,
        n_channel_in=n_channel,
        # adjust for your data below (make patch size as large as possible)
        train_patch_size=(64, 64, 64),
        # train_batch_size=2,
    )
    print(conf)
    vars(conf)
    if use_gpu:
        from csbdeep.utils.tf import limit_gpu_memory

        # adjust as necessary: limit GPU memory to be used by TensorFlow to leave some to OpenCL-based computations
        # limit_gpu_memory(0.8)
        # alternatively, try this:
        limit_gpu_memory(None, allow_growth=True)

    model = StarDist3D(conf, name="stardist", basedir="models")

    median_size = calculate_extents(Y, np.median)
    fov = np.array(model._axes_tile_overlap("ZYX"))
    print(f"median object size:      {median_size}")
    print(f"network field of view :  {fov}")
    if any(median_size > fov):
        print(
            "WARNING: median object size larger than field of view of the neural network."
        )

    def random_fliprot(img, mask, axis=None):
        if axis is None:
            axis = tuple(range(mask.ndim))
        axis = tuple(axis)

        assert img.ndim >= mask.ndim
        perm = tuple(np.random.permutation(axis))
        transpose_axis = np.arange(mask.ndim)
        for a, p in zip(axis, perm):
            transpose_axis[a] = p
        transpose_axis = tuple(transpose_axis)
        img = img.transpose(transpose_axis + tuple(range(mask.ndim, img.ndim)))
        mask = mask.transpose(transpose_axis)
        for ax in axis:
            if np.random.rand() > 0.5:
                img = np.flip(img, axis=ax)
                mask = np.flip(mask, axis=ax)
        return img, mask

    def random_intensity_change(img):
        return img * np.random.uniform(0.6, 2) + np.random.uniform(-0.2, 0.2)

    def augmenter(x, y):
        """Augmentation of a single input/label image pair.
        x is an input image
        y is the corresponding ground-truth label image
        """
        # Note that we only use fliprots along axis=(1,2), i.e. the yx axis
        # as 3D microscopy acquisitions are usually not axially symmetric
        x, y = random_fliprot(x, y, axis=(1, 2))
        x = random_intensity_change(x)
        return x, y

    model.train(
        X_trn,
        Y_trn,
        validation_data=(X_val, Y_val),
        augmenter=augmenter,
        epochs=NUM_EPOCHS,
    )

    # Threshold optimization
    # model.optimize_thresholds(X_val, Y_val)
    model.optimize_thresholds(X_trn, Y_trn)

    # once done, results are saved to ./models/stardist. We want to rename the folder to include the seed and split
    model_path = Path("models") / "stardist"
    new_model_path = Path(f"models/stardist_{seed}_{path_images.parts[-1]}")
    new_model_path.mkdir(parents=True, exist_ok=True)
    try:
        # copy the model to a new folder with the seed and split. Do not rename.
        files = sorted(model_path.glob("*"))
        # skip the "logs" folder as the permissions are problematic
        files = [f for f in files if "logs" not in str(f)]
        for file in files:
            new_file = new_model_path / file.name
            try:
                shutil.copy2(file, new_file)
            except PermissionError:
                print(
                    f"Could not copy {file} to {new_file}. Permission denied."
                )
                continue
    except Exception as e:
        print(f"Could not copy {model_path} to {new_model_path}.")
        print("Training will be stopped to avoid overwriting the model.")
        raise SystemError("Could not copy files") from e


if __name__ == "__main__":
    for split in TRAINING_PERCENTAGES:
        for data in SPLITS:
            print(f"Training {split}% split with data {data}")
            print("_" * 50)
            path_images = DATA_PATH / f"{data}"
            save_path = pt.Path(f"./models/stardist_{data}_{split}")
            if save_path.exists():
                print(f"Model {save_path} already exists. Skipping.")
                continue
            train_stardist(path_images, split, seed=SEED)
            print(f"Training {split}% split with data {data} done.")
