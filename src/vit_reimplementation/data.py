"""CIFAR-10 dataset splitting and DataLoader construction."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import datasets, transforms


def create_cifar10_splits(
    root: str | Path = "./data",
    validation_size: int = 5_000,
    seed: int = 0,
    train_transform: Any | None = None,
    eval_transform: Any | None = None,
    download: bool = True,
) -> tuple[Dataset, Dataset, Dataset]:
    """Create reproducible train, validation, and official test datasets."""
    if train_transform is None:
        train_transform = transforms.ToTensor()
    if eval_transform is None:
        eval_transform = transforms.ToTensor()

    full_train_dataset = datasets.CIFAR10(
        root=str(root),
        train=True,
        download=download,
        transform=train_transform,
    )

    # A second view of the same official training split allows validation to
    # use deterministic transforms while training later uses augmentation.
    full_validation_dataset = datasets.CIFAR10(
        root=str(root),
        train=True,
        download=False,
        transform=eval_transform,
    )

    test_dataset = datasets.CIFAR10(
        root=str(root),
        train=False,
        download=download,
        transform=eval_transform,
    )

    num_examples = len(full_train_dataset)
    if not 0 < validation_size < num_examples:
        raise ValueError(
            "validation_size must be between 1 and the number of training examples"
        )

    generator = torch.Generator().manual_seed(seed)
    shuffled_indices = torch.randperm(
        num_examples,
        generator=generator,
    )

    num_train = num_examples - validation_size
    train_indices = shuffled_indices[:num_train]
    validation_indices = shuffled_indices[num_train:]

    train_dataset = Subset(
        full_train_dataset,
        train_indices.tolist(),
    )
    validation_dataset = Subset(
        full_validation_dataset,
        validation_indices.tolist(),
    )

    return train_dataset, validation_dataset, test_dataset


def create_data_loaders(
    train_dataset: Dataset,
    validation_dataset: Dataset,
    test_dataset: Dataset,
    train_batch_size: int = 128,
    eval_batch_size: int = 256,
    num_workers: int = 0,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """Create conventional train, validation, and test DataLoaders."""
    train_loader = DataLoader(
        train_dataset,
        batch_size=train_batch_size,
        shuffle=True,
        num_workers=num_workers,
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=eval_batch_size,
        shuffle=False,
        num_workers=num_workers,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=eval_batch_size,
        shuffle=False,
        num_workers=num_workers,
    )

    return train_loader, validation_loader, test_loader
