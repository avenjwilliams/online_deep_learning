import argparse
from datetime import datetime
from pathlib import Path

import torch
import torch.nn as nn

import numpy as np
import torch
import torch.utils.tensorboard as tb

from .models import load_model, save_model
from .datasets.road_dataset import load_data

class Loss_Function(nn.Module):
    def forward(self, pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        abs_err = (pred - target).abs()
        m = mask.bool().unsqueeze(-1).expand_as(abs_err)
        return abs_err[m].mean()
        
F_idx = 0
L_idx = 1
def longitudinal_MAE(pred, target, mask):
    e = (pred[..., F_idx] - target[..., F_idx]).abs()
    return e[mask].mean().item()
    
def latitudinal_MAE(pred, target, mask):
    e = (pred[..., L_idx] - target[..., L_idx]).abs()
    return e[mask].mean().item()
    

def train(
    exp_dir: str = "logs",
    model_name: str = "mlp_planner",
    num_epoch: int = 10,
    lr: float = 1e-3,
    batch_size: int = 64,
    seed: int = 2024,
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
    model = load_model(model_name, **kwargs)
    model = model.to(device)
    model.train()

    train_data = load_data("drive_data/train", shuffle=True, batch_size=batch_size, num_workers=2)
    val_data = load_data("drive_data/val", shuffle=False)

    # create loss function and optimizer
    loss_func = Loss_Function()
    
    optim = torch.optim.Adam(model.parameters(), lr = lr)

    global_step = 0
    metrics = {"train_longitudinal_acc": [], "train_latitudinal_acc": [], "val_longitudinal_acc": [], "val_latitudinal_acc": []}

    # training loop
    for epoch in range(num_epoch):
        # clear metrics at beginning of epoch
        for key in metrics:
            metrics[key].clear()

        model.train()

        for (track_left, track_right), (waypoints, waypoints_mask) in train_data:

            track_left, track_right = track_left.to(device), track_right.to(device)
            waypoints, waypoints_mask = waypoints.to(device), waypoints_mask.to(device)

            pred = model(track_left, track_right)

            loss_val = loss_func(pred, waypoints, waypoints_mask)

            # adjust
            optim.zero_grad()
            loss_val.backward()
            optim.step()

            # log
            metrics['train_longitudinal_acc'].append(longitudinal_MAE(pred, waypoints, waypoints_mask))
            metrics['train_latitudinal_acc'].append(latitudinal_MAE(pred, waypoints, waypoints_mask))

            global_step += 1

        # disable gradient computation and switch to evaluation mode
        with torch.inference_mode():
            model.eval()

            for track_left, track_right, waypoints, waypoints_mask in val_data:
                track_left, track_right = track_left.to(device), track_right.to(device)
                waypoints, waypoints_mask = waypoints.to(device), waypoints_mask.to(device)

                pred = model(track_left, track_right)

                # log
                metrics['val_longitudinal_acc'].append(longitudinal_MAE(pred, waypoints, waypoints_mask))
                metrics['val_latitudinal_acc'].append(latitudinal_MAE(pred, waypoints, waypoints_mask))

        # log average train and val accuracy to tensorboard
        epoch_train_longitudinal_acc = torch.as_tensor(metrics["train_longitudinal_acc"]).mean()
        logger.add_scalar("train_longitudinal_accuracy", epoch_train_longitudinal_acc, global_step)

        epoch_train_latitudinal_acc = torch.as_tensor(metrics["train_latitudinal_acc"]).mean()
        logger.add_scalar("train_latitudinal_accuracy", epoch_train_latitudinal_acc, global_step)

        # log average train and val accuracy to tensorboard
        epoch_val_longitudinal_acc = torch.as_tensor(metrics["val_longitudinal_acc"]).mean()
        logger.add_scalar("val_longitudinal_accuracy", epoch_val_longitudinal_acc, global_step)

        epoch_val_latitudinal_acc = torch.as_tensor(metrics["val_latitudinal_acc"]).mean()
        logger.add_scalar("val_latitudinal_accuracy", epoch_val_latitudinal_acc, global_step)



        # print on first, last, every 10th epoch
        if epoch == 0 or epoch == num_epoch - 1 or (epoch + 1) % 10 == 0:
            print(
                f"Epoch {epoch + 1:2d} / {num_epoch:2d}: "
                f"train_longitudinal_acc={epoch_train_longitudinal_acc:.4f} "
                f"train_latitudinal_acc={epoch_train_latitudinal_acc:.4f} "
                f"val_longitudinal_acc={epoch_val_longitudinal_acc:.4f} "
                f"val_latitudinal_acc={epoch_val_latitudinal_acc:.4f}"
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