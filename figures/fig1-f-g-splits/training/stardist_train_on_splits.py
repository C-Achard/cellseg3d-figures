from __future__ import (
    absolute_import,
    division,
    print_function,
    unicode_literals,
)

import pathlib as pt
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

DATA_PATH = (
    Path.home()
    / "Desktop/Code/CELLSEG_BENCHMARK/TPH2_mesospim/TRAINING/SPLITS"
)

NUM_EPOCHS = 50
TRAINING_PERCENTAGES = [10, 20, 40, 80]
SEEDS = [34936339, 34936397, 34936345]


def train_stardist(path_images, seed):
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
    # rng = np.random.RandomState(42)
    # ind = rng.permutation(len(X))
    # print(ind)
    # ind = range(len(X))
    # print(ind)
    # n_val = max(1, int(round(VAL_FRACTION * len(ind))))
    # ind_train, ind_val = ind[:-n_val], ind[-n_val:]
    # X_val, Y_val = [X[i] for i in ind_val]  , [Y[i] for i in ind_val]
    # X_trn, Y_trn = [X[i] for i in ind_train], [Y[i] for i in ind_train]
    X_trn, Y_trn = X, Y
    print("number of images: %3d" % len(X))
    print("- training:       %3d" % len(X_trn))
    # print('- validation:     %3d' % len(X_val))
    print("Train files :")
    [print(X_paths[i]) for i in len(X_trn)]
    print("Validation files :")
    # [print(X_paths[i]) for i in ind_val]
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
    grid = tuple(1 if a > 1.5 else 2 for a in anisotropy)

    # Use rays on a Fibonacci lattice adjusted for measured anisotropy of the training data
    rays = Rays_GoldenSpiral(n_rays, anisotropy=anisotropy)

    conf = Config3D(
        rays=rays,
        grid=grid,
        anisotropy=anisotropy,
        use_gpu=use_gpu,
        n_channel_in=n_channel,
        # adjust for your data below (make patch size as large as possible)
        train_patch_size=(8, 64, 64),
        train_batch_size=2,
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
        # validation_data=(X_val,Y_val),
        augmenter=augmenter,
        epochs=NUM_EPOCHS,
    )

    # Threshold optimization
    # model.optimize_thresholds(X_val, Y_val)
    model.optimize_thresholds(X_trn, Y_trn)

    # once done, results are saved to ./models/stardist. We want to rename the folder to include the seed and split
    model_path = Path("models") / "stardist"
    new_model_path = Path(f"models/stardist_{seed}_{path_images.parts[-1]}")
    try:
        model_path.rename(new_model_path)
    except Exception as e:
        print(f"Could not rename {model_path} to {new_model_path}.")
        print("Training will be stopped to avoid overwriting the model.")
        raise SystemError("Model already exists, aborting training.") from e


if __name__ == "__main__":
    for split in TRAINING_PERCENTAGES:
        print(f"Training {split} split.")
        for seed in SEEDS:
            print(f"Training {split} split with seed {seed}.")
            print("_" * 50)
            path_images = DATA_PATH / f"{split}"
            train_stardist(path_images, seed)
            print(f"Training {split} split with seed {seed} done.")
