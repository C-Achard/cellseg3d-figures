from __future__ import (
    absolute_import,
    division,
    print_function,
    unicode_literals,
)

import sys
from glob import glob

import numpy as np
from csbdeep.utils import Path, normalize
from stardist import (
    Rays_GoldenSpiral,
    calculate_extents,
    fill_label_holes,
    random_label_cmap,
)
from stardist.models import Config3D, StarDist3D
from tifffile import imread
from tqdm import tqdm

np.random.seed(42)
lbl_cmap = random_label_cmap()

VAL_FRACTION = 0.2
NUM_EPOCHS = 50

path_images = Path(
    "S:/Code/CELLSEG_BENCHMARK/EmbedSeg datasets/EMBEDSEG RETRAIN/PlNuc"
)
X_paths = sorted(glob(str(path_images / "images/*.tif")))
Y_paths = sorted(glob(str(path_images / "labels/*.tif")))
# assert all(Path(x).name==Path(y).name for x,y in zip(X_paths,Y_paths))

X = list(map(imread, X_paths))
Y = list(map(imread, Y_paths))
Y = [Yi.astype(np.uint16) for Yi in Y]
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

assert len(X) > 1, "not enough training data"
rng = np.random.RandomState(42)
# ind = rng.permutation(len(X))
# print(ind)
ind = range(len(X))
print(ind)

n_val = max(1, int(round(VAL_FRACTION * len(ind))))
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

extents = calculate_extents(Y)
anisotropy = tuple(np.max(extents) / extents)
print("empirical anisotropy of labeled objects = %s" % str(anisotropy))

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
    train_patch_size=(64, 64, 64),
)
print(conf)
vars(conf)

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
model.optimize_thresholds(X_val, Y_val)
