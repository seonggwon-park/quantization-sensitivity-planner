from typing import Sequence

import torch
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import transforms
from torchvision.datasets import CIFAR10

from config import ExperimentConfig


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class BinaryCIFAR10(Dataset):
    """
    CIFAR-10에서 선택한 두 class만 남기고,
    label을 0 / 1 binary label로 remap하는 Dataset.
    """

    def __init__(
        self,
        root: str,
        train: bool,
        class_ids: Sequence[int],
        transform,
        source_indices: Sequence[int] | None = None,
        download: bool = False,
    ):
        if len(class_ids) != 2:
            raise ValueError("BinaryCIFAR10 requires exactly two class IDs.")

        self.base_dataset = CIFAR10(
            root=root,
            train=train,
            transform=transform,
            download=download,
        )

        self.class_ids = tuple(class_ids)
        self.label_map = {
            self.class_ids[0]: 0,
            self.class_ids[1]: 1,
        }

        if source_indices is None:
            self.source_indices = [
                index
                for index, target in enumerate(self.base_dataset.targets)
                if target in self.class_ids
            ]
        else:
            self.source_indices = list(source_indices)

        for index in self.source_indices:
            original_label = self.base_dataset.targets[index]

            if original_label not in self.label_map:
                raise ValueError(
                    f"Index {index} has label {original_label}, "
                    "but it is outside selected binary classes."
                )

    def __len__(self) -> int:
        return len(self.source_indices)

    def __getitem__(self, index: int):
        original_index = self.source_indices[index]

        image, original_label = self.base_dataset[original_index]
        binary_label = self.label_map[int(original_label)]

        return image, binary_label


def build_transforms(image_size: int):
    """
    ResNet-18 pretrained weights were trained with ImageNet-style normalization.
    """

    train_transform = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )

    evaluation_transform = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )

    return train_transform, evaluation_transform


def split_train_validation_indices(
    root: str,
    class_ids: Sequence[int],
    validation_fraction: float,
    seed: int,
):
    """
    Select only binary-class samples, then split them into train and validation.
    """

    raw_train_dataset = CIFAR10(
        root=root,
        train=True,
        download=True,
    )

    eligible_indices = [
        index
        for index, target in enumerate(raw_train_dataset.targets)
        if target in class_ids
    ]

    generator = torch.Generator().manual_seed(seed)
    permutation = torch.randperm(
        len(eligible_indices),
        generator=generator,
    ).tolist()

    validation_size = int(len(eligible_indices) * validation_fraction)

    validation_positions = permutation[:validation_size]
    train_positions = permutation[validation_size:]

    validation_indices = [
        eligible_indices[position]
        for position in validation_positions
    ]

    train_indices = [
        eligible_indices[position]
        for position in train_positions
    ]

    return train_indices, validation_indices


def build_dataloaders(config: ExperimentConfig):
    """
    Create train / validation / test dataloaders.
    """

    train_transform, evaluation_transform = build_transforms(
        config.image_size
    )

    train_indices, validation_indices = split_train_validation_indices(
        root=str(config.data_dir),
        class_ids=config.class_ids,
        validation_fraction=config.validation_fraction,
        seed=config.seed,
    )

    train_dataset = BinaryCIFAR10(
        root=str(config.data_dir),
        train=True,
        class_ids=config.class_ids,
        transform=train_transform,
        source_indices=train_indices,
        download=False,
    )

    validation_dataset = BinaryCIFAR10(
        root=str(config.data_dir),
        train=True,
        class_ids=config.class_ids,
        transform=evaluation_transform,
        source_indices=validation_indices,
        download=False,
    )

    test_dataset = BinaryCIFAR10(
        root=str(config.data_dir),
        train=False,
        class_ids=config.class_ids,
        transform=evaluation_transform,
        source_indices=None,
        download=True,
    )

    loader_kwargs = {
        "batch_size": config.batch_size,
        "num_workers": config.num_workers,
        "pin_memory": torch.cuda.is_available(),
    }

    if config.num_workers > 0:
        loader_kwargs["persistent_workers"] = True

    train_loader = DataLoader(
        train_dataset,
        shuffle=True,
        **loader_kwargs,
    )

    validation_loader = DataLoader(
        validation_dataset,
        shuffle=False,
        **loader_kwargs,
    )

    test_loader = DataLoader(
        test_dataset,
        shuffle=False,
        **loader_kwargs,
    )

    return {
        "train": train_loader,
        "validation": validation_loader,
        "test": test_loader,
    }


def make_fixed_subset_loader(
    original_loader: DataLoader,
    max_samples: int | None,
    seed: int,
) -> DataLoader:
    """
    For fast debugging, evaluate on a fixed random subset.

    Use max_samples=None for the full test set.
    """

    if max_samples is None:
        return original_loader

    dataset = original_loader.dataset

    if max_samples >= len(dataset):
        return original_loader

    generator = torch.Generator().manual_seed(seed)

    indices = torch.randperm(
        len(dataset),
        generator=generator,
    )[:max_samples].tolist()

    subset = Subset(dataset, indices)

    loader_kwargs = {
        "batch_size": original_loader.batch_size,
        "num_workers": original_loader.num_workers,
        "pin_memory": torch.cuda.is_available(),
    }

    if original_loader.num_workers > 0:
        loader_kwargs["persistent_workers"] = True

    return DataLoader(
        subset,
        shuffle=False,
        **loader_kwargs,
    )