import argparse

import torch
import torch.nn.functional as functional

from config import ExperimentConfig
from data import build_dataloaders
from metrics import evaluate_classifier
from model import (
    build_binary_resnet18,
    load_binary_resnet18_checkpoint,
    save_checkpoint,
)
from utils import get_device, set_seed


def train_one_epoch(
    model,
    dataloader,
    optimizer,
    device,
):
    """
    One training epoch.
    """

    model.train()

    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    for images, labels in dataloader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        logits = model(images)

        loss = functional.cross_entropy(
            logits,
            labels,
        )

        loss.backward()
        optimizer.step()

        predictions = logits.argmax(dim=1)

        total_loss += loss.item() * labels.size(0)
        total_correct += (
            predictions == labels
        ).sum().item()

        total_samples += labels.size(0)

    return {
        "loss": total_loss / total_samples,
        "accuracy": total_correct / total_samples,
    }


def parse_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--epochs",
        type=int,
        default=10,
    )

    parser.add_argument(
        "--lr",
        type=float,
        default=1e-4,
    )

    return parser.parse_args()


def main():
    args = parse_arguments()

    config = ExperimentConfig()
    config.epochs = args.epochs
    config.learning_rate = args.lr

    config.create_directories()

    set_seed(config.seed)
    device = get_device()

    print(f"Using device: {device}")
    print(
        f"Binary task: "
        f"{config.class_names[0]} vs "
        f"{config.class_names[1]}"
    )

    dataloaders = build_dataloaders(config)

    print(
        "Dataset sizes:",
        {
            split_name: len(loader.dataset)
            for split_name, loader in dataloaders.items()
        },
    )

    model = build_binary_resnet18(
        pretrained=True
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=config.epochs,
    )

    best_validation_accuracy = -1.0

    for epoch in range(1, config.epochs + 1):
        train_metrics = train_one_epoch(
            model=model,
            dataloader=dataloaders["train"],
            optimizer=optimizer,
            device=device,
        )

        validation_metrics = evaluate_classifier(
            model=model,
            dataloader=dataloaders["validation"],
            device=device,
        )

        scheduler.step()

        print(
            f"[Epoch {epoch:02d}/{config.epochs}] "
            f"train_acc={train_metrics['accuracy']:.4f} "
            f"train_loss={train_metrics['loss']:.4f} | "
            f"val_acc={validation_metrics['accuracy']:.4f} "
            f"val_loss={validation_metrics['loss']:.4f}"
        )

        if (
            validation_metrics["accuracy"]
            > best_validation_accuracy
        ):
            best_validation_accuracy = validation_metrics[
                "accuracy"
            ]

            save_checkpoint(
                model=model,
                checkpoint_path=config.checkpoint_path,
                metadata={
                    "class_ids": config.class_ids,
                    "class_names": config.class_names,
                    "best_validation_accuracy": (
                        best_validation_accuracy
                    ),
                },
            )

            print(
                f"Saved best checkpoint: "
                f"{config.checkpoint_path}"
            )

    best_model, checkpoint = (
        load_binary_resnet18_checkpoint(
            checkpoint_path=config.checkpoint_path,
            device=device,
        )
    )

    test_metrics = evaluate_classifier(
        model=best_model,
        dataloader=dataloaders["test"],
        device=device,
    )

    print("\nFinal test metrics")
    print(test_metrics)

    print("\nCheckpoint metadata")
    print(
        {
            key: value
            for key, value in checkpoint.items()
            if key != "model_state_dict"
        }
    )


if __name__ == "__main__":
    main()