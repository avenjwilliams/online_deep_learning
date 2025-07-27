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
from .metrics import ConfusionMatrix  # or wherever it's defined

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
    
class RegressionLoss(nn.Module):
    def forward(self, logits: torch.Tensor, target: torch.FloatTensor) -> torch.Tensor:
        """
        Depth regression loss
        Hint: simple one-liner

        Args:
            logits: tensor (b, h, w) raw depth predictions
            target: tensor (b, h, w) ground truth depth values

        Returns:
            tensor, scalar loss
        """
        return torch.nn.functional.l1_loss(logits, target)
    
class total_detection_loss(nn.Module):
    def __init__(self, classification_loss: nn.Module, regression_loss: nn.Module):
        super().__init__()
        self.classification_loss = classification_loss
        self.regression_loss = regression_loss

    def forward(self, logits: torch.Tensor, target: torch.LongTensor, depth_logits: torch.Tensor, depth_target: torch.FloatTensor) -> torch.Tensor:
        """
        Args:
            logits: tensor (b, h, w) raw class predictions
            target: tensor (b, h, w) ground truth class labels
            depth_logits: tensor (b, h, w) raw depth predictions
            depth_target: tensor (b, h, w) ground truth depth values

        Returns:
            tensor, scalar loss
        """
        return self.classification_loss(logits, target) + self.regression_loss(depth_logits, depth_target)

def train(
    exp_dir: str = "logs",
    model_name: str = "classifier",
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
    model = load_model(model_name, **kwargs).to(device)
    model.train()

    train_data = load_data("drive_data/train", shuffle=True, batch_size=batch_size, num_workers=2)
    val_data = load_data("drive_data/val", shuffle=False)

    loss_func = total_detection_loss(ClassificationLoss(), RegressionLoss())
    optim = torch.optim.Adam(model.parameters(), lr=lr)
    global_step = 0

    for epoch in range(num_epoch):
        model.train()
        for img, label in train_data:
            img = img.to(device)
            label = {k: v.to(device) for k, v in label.items()}

            logits, depth_pred = model(img)
            loss_val = loss_func(logits, label['track'], depth_pred, label['depth'])

            optim.zero_grad()
            loss_val.backward()
            optim.step()

            logger.add_scalar("train_loss", loss_val.item(), global_step)
            global_step += 1

        with torch.inference_mode():
            model.eval()
            cm = ConfusionMatrix(num_classes=3)
            overall_mae_values = []
            lane_mae_values = []
            for img, label in val_data:
                img = img.to(device)
                label = {k: v.to(device) for k, v in label.items()}
                
                logits, depth_pred = model(img)

                # IoU
                preds = logits.argmax(dim=1)
                cm.add(preds, label['track'])

                # Overall Depth MAE
                overall_mae = torch.abs(depth_pred - label["depth"]).mean().item()
                overall_mae_values.append(overall_mae)

                # Lane Depth MAE
                lane_mask = (label['track'] == 1) | (label['track'] == 2)
                if lane_mask.any():
                    lane_mae = torch.abs(depth_pred - label["depth"])[lane_mask].mean().item()
                    lane_mae_values.append(lane_mae)

        # Compute and log metrics
        metrics_iou = cm.compute()
        mean_iou = metrics_iou["iou"]
        logger.add_scalar("val_mean_iou", mean_iou, global_step)
        print(f"Epoch {epoch+1}: Mean IoU = {mean_iou:.4f}")

        if lane_mae_values:
            avg_lane_mae = sum(lane_mae_values) / len(lane_mae_values)
            logger.add_scalar("val_lane_depth_mae", avg_lane_mae, global_step)
            print(f"Epoch {epoch+1}: Lane-only MAE = {avg_lane_mae:.4f}")

        if overall_mae_values:
            avg_overall_mae = sum(overall_mae_values) / len(overall_mae_values)
            logger.add_scalar("val_depth_mae", avg_overall_mae, global_step)
            print(f"Epoch {epoch+1}: Overall Depth MAE = {avg_overall_mae:.4f}")


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
