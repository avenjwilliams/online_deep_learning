import argparse
from datetime import datetime
from pathlib import Path

import torch
import torch.nn as nn

import numpy as np
import torch
import torch.utils.tensorboard as tb

from .models import load_model, save_model
from .datasets.classification_dataset import load_data

class ClassificationLoss(nn.Module):
    def forward(self, logits: torch.Tensor, target: torch.LongTensor) -> torch.Tensor:
        """
        Multi-class classification loss
        Hint: simple one-liner

        Args:
            logits: tensor (b, c) logits, where c is the number of classes
            target: tensor (b,) labels

        Returns:
            tensor, scalar loss
        """

        return torch.nn.functional.cross_entropy(logits, target)

def train(
    exp_dir: str = "logs",
    model_name: str = "classifier",
    num_epoch: int = 10,
    lr: float = 1e-3,
    batch_size: int = 64,
    seed: int = 2024,
    channels_l0: int = 32,
    n_blocks: int = 4,
    **kwargs,
):
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available() and torch.backends.mps.is_built():
        device = torch.device("mps")
    else:
        print("CUDA not available, using CPU")
        device = torch.device("cpu")

    # set random seed so each run is deterministic
    torch.manual_seed(seed)
    np.random.seed(seed)

    # directory with timestamp to save tensorboard logs and model checkpoints
    log_dir = Path(exp_dir) / f"{model_name}_{datetime.now().strftime('%m%d_%H%M%S')}"
    logger = tb.SummaryWriter(log_dir)

    # note: the grader uses default kwargs, you'll have to bake them in for the final submission
    model = load_model(model_name, channels_l0=channels_l0, n_blocks=n_blocks, **kwargs)
    model = model.to(device)
    model.train()

    train_data = load_data("classification_data/train", shuffle=True, batch_size=batch_size, num_workers=2)
    val_data = load_data("classification_data/val", shuffle=False)

    # create loss function and optimizer
    loss_func = ClassificationLoss()
    
    optim = torch.optim.Adam(model.parameters(), lr = lr)

    global_step = 0
    metrics = {"train_acc": [], "val_acc": []}

    # training loop
    for epoch in range(num_epoch):
        # clear metrics at beginning of epoch
        for key in metrics:
            metrics[key].clear()

        model.train()

        for img, label in train_data:
            img, label = img.to(device), label.to(device)
            pred = model(img)
            loss_val = loss_func(pred, label)
            logger.add_scalar("train_loss", loss_val, global_step)

            # adjust
            optim.zero_grad()
            loss_val.backward()
            optim.step()

            # log
            batch_acc = (pred.argmax(dim = 1) == label).float().mean().item()
            metrics['train_acc'].append(batch_acc)

            global_step += 1

        # disable gradient computation and switch to evaluation mode
        with torch.inference_mode():
            model.eval()

            for img, label in val_data:
                img, label = img.to(device), label.to(device)
                pred = model(img)

                batch_acc = (pred.argmax(dim = 1) == label).float().mean().item()
                metrics['val_acc'].append(batch_acc)

        # log average train and val accuracy to tensorboard
        epoch_train_acc = torch.as_tensor(metrics["train_acc"]).mean()
        logger.add_scalar("train_accuracy", epoch_train_acc, global_step)

        epoch_val_acc = torch.as_tensor(metrics["val_acc"]).mean()
        logger.add_scalar("val_accuracy", epoch_val_acc, global_step)

        # print on first, last, every 10th epoch
        if epoch == 0 or epoch == num_epoch - 1 or (epoch + 1) % 10 == 0:
            print(
                f"Epoch {epoch + 1:2d} / {num_epoch:2d}: "
                f"train_acc={epoch_train_acc:.4f} "
                f"val_acc={epoch_val_acc:.4f}"
            )

    # save and overwrite the model in the root directory for grading
    save_model(model)

    # save a copy of model weights in the log directory
    torch.save(model.state_dict(), log_dir / f"{model_name}.th")
    print(f"Model saved to {log_dir / f'{model_name}.th'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--exp_dir", type=str, default="logs")
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--num_epoch", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=2024)

    # optional: additional model hyperparamters
    # parser.add_argument("--num_layers", type=int, default=3)

    # pass all arguments to train
    train(**vars(parser.parse_args()))
