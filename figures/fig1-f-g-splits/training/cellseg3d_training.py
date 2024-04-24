"""Showcases how to train a model without napari."""

import sys
from pathlib import Path

import torch
from napari_cellseg3d import config as cfg
from napari_cellseg3d.code_models.worker_training import (
    SupervisedTrainingWorker,
    WNetTrainingWorker,
)
from napari_cellseg3d.utils import LOGGER as logger
from napari_cellseg3d.utils import get_all_matching_files

# set wandb mode globally
WANDB_MODE = "online"  # "disabled"

sys.path.append("../..")

# DATA_PATH = (
#     Path.home()
#     / "Desktop/Code/CELLSEG_BENCHMARK/TPH2_mesospim/TRAINING/SPLITS"
# )
DATA_PATH = Path("/data/cyril/CELLSEG_BENCHMARK/TPH2_mesospim/TRAINING/SPLITS")

TRAINING_PERCENTAGES = [10, 20, 40, 80]
MODELS = ["SegResNet", "SwinUNetR"]
SEEDS = [34936339, 34936397, 34936345]


def train_on_splits(models, training_splits, seeds):
    """Trains models on different training splits."""
    for model_name in models:
        for training_split in training_splits:
            for seed in seeds:
                print(
                    f"Training {model_name} on {training_split}% with seed {seed}"
                )
                remote_training_supervised(model_name, training_split, seed)


def train_wnet_on_splits(training_splits, seeds):
    """Trains WNet on different training splits."""
    for training_split in training_splits:
        for seed in seeds:
            print(f"Training WNet on {training_split}% with seed {seed}")
            remote_training_unsupervised(training_split, seed)


class LogFixture:
    """Fixture for napari-less logging, replaces napari_cellseg3d.interface.Log in model_workers.

    This allows to redirect the output of the workers to stdout instead of a specialized widget.
    """

    def __init__(self):
        """Creates a LogFixture object."""
        super(LogFixture, self).__init__()

    def print_and_log(self, text, printing=None):
        """Prints and logs text."""
        print(text)

    def warn(self, warning):
        """Logs warning."""
        logger.warning(warning)

    def error(self, e):
        """Logs error."""
        raise (e)


def prepare_data(images_path, labels_path, results_path):
    """Prepares data for training."""
    assert images_path.exists(), f"Images path does not exist: {images_path}"
    assert labels_path.exists(), f"Labels path does not exist: {labels_path}"
    if not results_path.exists():
        results_path.mkdir(parents=True)

    images = get_all_matching_files(images_path)
    labels = get_all_matching_files(labels_path)

    print(f"Images paths: {images}")
    print(f"Labels paths: {labels}")

    logger.info("Images :\n")
    for file in images:
        logger.info(Path(file).name)
    logger.info("*" * 10)
    logger.info("Labels :\n")
    for file in images:
        logger.info(Path(file).name)

    assert len(images) == len(
        labels
    ), "Number of images and labels must be the same"

    return [
        {"image": str(image_path), "label": str(label_path)}
        for image_path, label_path in zip(images, labels)
    ]


def remote_training_supervised(model_name, training_split, seed):
    """Function to train a model without napari."""
    # print(f"Results path: {RESULTS_PATH.resolve()}")
    batch_size = 10 if model_name == "SegResNet" else 5
    results_path = (
        Path("/data/cyril")
        / "CELLSEG_BENCHMARK/cellseg3d_train"
        / f"{model_name}_{training_split}_{seed}"
    )
    if results_path.exists():
        # ask user if they want to continue training
        print(
            f"Results path {results_path} already exists. Continue training? (y/n)"
        )
        answer = input()
        if answer.lower() != "y":
            print("Training aborted.")
            return

    data_path = DATA_PATH / str(training_split)
    data_dict = prepare_data(
        data_path, data_path / "labels/semantic", results_path
    )

    wandb_config = cfg.WandBConfig(
        mode=WANDB_MODE,
        save_model_artifact=True,
    )

    deterministic_config = cfg.DeterministicConfig(
        seed=seed,
    )

    # create validation data dict
    val_data = DATA_PATH / "validation"
    val_data_dict = prepare_data(val_data, val_data / "labels", results_path)

    worker_config = cfg.SupervisedTrainingWorkerConfig(
        device="cuda:3",
        max_epochs=50,
        learning_rate=0.001,  # 1e-3
        validation_interval=2,
        batch_size=batch_size,  # 10 for SegResNet
        deterministic_config=deterministic_config,
        scheduler_factor=0.5,
        scheduler_patience=10,  # use default scheduler
        weights_info=cfg.WeightsInfo(),  # no pretrained weights
        results_path_folder=str(results_path),
        sampling=False,
        do_augmentation=True,
        train_data_dict=data_dict,
        eval_data_dict=val_data_dict,
        # supervised specific
        model_info=cfg.ModelInfo(
            name=model_name,
            model_input_size=(64, 64, 64),
        ),
        loss_function="Generalized Dice",
        training_percent=99,
    )

    worker = SupervisedTrainingWorker(worker_config)
    worker.wandb_config = wandb_config
    ######### SET LOG
    log = LogFixture()
    worker.log_signal.connect(log.print_and_log)
    worker.warn_signal.connect(log.warn)
    worker.error_signal.connect(log.error)

    results = []
    for result in worker.train():
        results.append(result)
    print("Training finished")


def remote_training_unsupervised(training_split, seed):
    """Function to train a model without napari."""
    # print(f"Results path: {RESULTS_PATH.resolve()}")
    batch_size = 5
    results_path = (
        Path("/data/cyril")
        / "CELLSEG_BENCHMARK/cellseg3d_train"
        / f"WNet_{training_split}_{seed}"
    )

    data_path = DATA_PATH / str(training_split)
    data_dict = prepare_data(
        data_path, data_path / "labels" / "semantic", results_path
    )

    data_dict = [{"image": d["image"]} for d in data_dict]

    wandb_config = cfg.WandBConfig(
        mode="online",
        save_model_artifact=True,
    )

    deterministic_config = cfg.DeterministicConfig(
        seed=seed,
    )

    worker_config = cfg.WNetTrainingWorkerConfig(
        device="cuda:3",
        max_epochs=int(6000 / training_split),  # 50,
        # model params
        in_channels=1,
        out_channels=1,
        num_classes=2,
        dropout=0.65,
        learning_rate=2e-5,
        use_clipping=False,
        clipping=1.0,
        weight_decay=0.01,  # 1e-5
        # NCuts loss params
        intensity_sigma=1.0,
        spatial_sigma=4.0,
        radius=2,
        # reconstruction loss params
        reconstruction_loss="MSE",  # or "BCE"
        # summed losses weights
        n_cuts_weight=0.5,
        rec_loss_weight=0.5 / 1.0,  # 0.5 / 100
        validation_interval=2,
        batch_size=batch_size,
        deterministic_config=deterministic_config,
        scheduler_factor=0.5,
        scheduler_patience=10,  # use default scheduler
        weights_info=cfg.WeightsInfo(),  # no pretrained weights
        results_path_folder=str(results_path),
        sampling=False,
        do_augmentation=True,
        train_data_dict=data_dict,
        # eval_data_dict=val_data_dict,
    )

    worker = WNetTrainingWorker(worker_config)
    worker.wandb_config = wandb_config
    ######### SET LOG
    log = LogFixture()
    worker.log_signal.connect(log.print_and_log)
    worker.warn_signal.connect(log.warn)
    worker.error_signal.connect(log.error)

    results = []
    for result in worker.train():
        results.append(result)
    print("Training finished")
    del results
    del worker
    torch.cuda.empty_cache()


if __name__ == "__main__":
    # results = remote_training_supervised()
    # train_on_splits(MODELS, TRAINING_PERCENTAGES, SEEDS)
    train_wnet_on_splits(TRAINING_PERCENTAGES, SEEDS)
