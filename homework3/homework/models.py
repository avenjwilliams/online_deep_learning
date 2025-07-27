from pathlib import Path

import torch
import torch.nn as nn

HOMEWORK_DIR = Path(__file__).resolve().parent
INPUT_MEAN = [0.2788, 0.2657, 0.2629]
INPUT_STD = [0.2064, 0.1944, 0.2252]


class Classifier(nn.Module):
    class Block(nn.Module):
      def __init__(self, in_channels = 3, out_channels = 32, stride = 1):
          super().__init__()

          # Main path: 2 conv layers
          self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, stride=stride)
          self.bn1 = nn.BatchNorm2d(out_channels)
          self.relu = nn.ReLU(inplace=True)
          self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
          self.bn2 = nn.BatchNorm2d(out_channels)

          # Residual (skip) connection
          if in_channels != out_channels:
              self.skip = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride)
          else:
              self.skip = nn.Identity()

      def forward(self, x):
          identity = self.skip(x)

          out = self.relu(self.bn1(self.conv1(x)))
          out = self.bn2(self.conv2(out))
          out += identity
          out = self.relu(out)

          return out
        
    def __init__(
        self,
        in_channels: int = 3,
        num_classes: int = 6,
        channels_l0: int = 32,
        n_blocks: int = 4
    ):
        """
        A convolutional network for image classification.

        Args:
            in_channels: int, number of input channels
            num_classes: int
        """
        super().__init__()

        self.register_buffer("input_mean", torch.as_tensor(INPUT_MEAN))
        self.register_buffer("input_std", torch.as_tensor(INPUT_STD))

        cnn_layers = []
        # Initial downsampling
        cnn_layers.append(nn.Conv2d(in_channels, channels_l0, kernel_size=3, stride=2, padding=1))
        cnn_layers.append(nn.ReLU())

        # Residual blocks with stride=1
        for _ in range(n_blocks):
            cnn_layers.append(self.Block(channels_l0, channels_l0, stride=1))

        # Head
        cnn_layers.append(nn.AdaptiveAvgPool2d((1, 1)))
        cnn_layers.append(nn.Flatten())
        cnn_layers.append(nn.Linear(channels_l0, num_classes))  # → (B, 6)
        self.network = torch.nn.Sequential(*cnn_layers)


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: tensor (b, 3, h, w) image

        Returns:
            tensor (b, num_classes) logits
        """
        # optional: normalizes the input
        z = (x - self.input_mean[None, :, None, None]) / self.input_std[None, :, None, None]
        logits = self.network(z)

        return logits

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """
        Used for inference, returns class labels
        This is what the AccuracyMetric uses as input (this is what the grader will use!).
        You should not have to modify this function.

        Args:
            x (torch.FloatTensor): image with shape (b, 3, h, w) and vals in [0, 1]

        Returns:
            pred (torch.LongTensor): class labels {0, 1, ..., 5} with shape (b, h, w)
        """
        return self(x).argmax(dim=1)


class Detector(torch.nn.Module):
    class Down_Block(nn.Module):
        def __init__(self, in_channels: int, out_channels: int):
            super().__init__()
            self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1)
            self.bn = nn.BatchNorm2d(out_channels)
            self.relu = nn.ReLU(inplace=True)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.relu(self.bn(self.conv(x)))
        
    class Up_Block(nn.Module):
        def __init__(self, in_channels: int, out_channels: int):
            super().__init__()
            self.conv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=3, stride = 2, padding=1, output_padding=1)
            self.bn = nn.BatchNorm2d(out_channels)
            self.relu = nn.ReLU(inplace=True)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.relu(self.bn(self.conv(x)))


    def __init__(
        self,
        in_channels: int = 3,
        num_classes: int = 3,
        first_channels: int = 16
    ):
        """
        A single model that performs segmentation and depth regression

        Args:
            in_channels: int, number of input channels
            num_classes: int
        """
        super().__init__()

        self.register_buffer("input_mean", torch.as_tensor(INPUT_MEAN))
        self.register_buffer("input_std", torch.as_tensor(INPUT_STD))

        self.segmentation_head = nn.Conv2d(first_channels, num_classes, kernel_size=1)
        self.depth_head = nn.Conv2d(first_channels, 1, kernel_size=1)
        self.depth_activation = nn.Sigmoid()

        Up_Conv = self.Up_Block
        Down_Conv = self.Down_Block
        self.network = torch.nn.Sequential(
            # Downsample
            Down_Conv(in_channels, first_channels),
            Down_Conv(first_channels, first_channels * 2),

            # Upsample
            Up_Conv(first_channels * 2, first_channels),
            Up_Conv(first_channels, first_channels)
        )
        

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Used in training, takes an image and returns raw logits and raw depth.
        This is what the loss functions use as input.

        Args:
            x (torch.FloatTensor): image with shape (b, 3, h, w) and vals in [0, 1]

        Returns:
            tuple of (torch.FloatTensor, torch.FloatTensor):
                - logits (b, num_classes, h, w)
                - depth (b, h, w)
        """
        # optional: normalizes the input
        z = (x - self.input_mean[None, :, None, None]) / self.input_std[None, :, None, None]

        output = self.network(z)
        logits = self.segmentation_head(output)
        raw_depth = self.depth_activation(self.depth_head(output)).squeeze(1)

        return logits, raw_depth

    def predict(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Used for inference, takes an image and returns class labels and normalized depth.
        This is what the metrics use as input (this is what the grader will use!).

        Args:
            x (torch.FloatTensor): image with shape (b, 3, h, w) and vals in [0, 1]

        Returns:
            tuple of (torch.LongTensor, torch.FloatTensor):
                - pred: class labels {0, 1, 2} with shape (b, h, w)
                - depth: normalized depth [0, 1] with shape (b, h, w)
        """
        logits, raw_depth = self(x)
        pred = logits.argmax(dim=1)

        # Optional additional post-processing for depth only if needed
        depth = raw_depth

        return pred, depth


MODEL_FACTORY = {
    "classifier": Classifier,
    "detector": Detector,
}


def load_model(
    model_name: str,
    with_weights: bool = False,
    **model_kwargs,
) -> torch.nn.Module:
    """
    Called by the grader to load a pre-trained model by name
    """
    m = MODEL_FACTORY[model_name](**model_kwargs)

    if with_weights:
        model_path = HOMEWORK_DIR / f"{model_name}.th"
        assert model_path.exists(), f"{model_path.name} not found"

        try:
            m.load_state_dict(torch.load(model_path, map_location="cpu"))
        except RuntimeError as e:
            raise AssertionError(
                f"Failed to load {model_path.name}, make sure the default model arguments are set correctly"
            ) from e

    # limit model sizes since they will be zipped and submitted
    model_size_mb = calculate_model_size_mb(m)

    if model_size_mb > 20:
        raise AssertionError(f"{model_name} is too large: {model_size_mb:.2f} MB")

    return m


def save_model(model: torch.nn.Module) -> str:
    """
    Use this function to save your model in train.py
    """
    model_name = None

    for n, m in MODEL_FACTORY.items():
        if type(model) is m:
            model_name = n

    if model_name is None:
        raise ValueError(f"Model type '{str(type(model))}' not supported")

    output_path = HOMEWORK_DIR / f"{model_name}.th"
    torch.save(model.state_dict(), output_path)

    return output_path


def calculate_model_size_mb(model: torch.nn.Module) -> float:
    """
    Args:
        model: torch.nn.Module

    Returns:
        float, size in megabytes
    """
    return sum(p.numel() for p in model.parameters()) * 4 / 1024 / 1024


def debug_model(batch_size: int = 1):
    """
    Test your model implementation

    Feel free to add additional checks to this function -
    this function is NOT used for grading
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sample_batch = torch.rand(batch_size, 3, 64, 64).to(device)

    print(f"Input shape: {sample_batch.shape}")

    model = load_model("classifier", in_channels=3, num_classes=6).to(device)
    output = model(sample_batch)

    # should output logits (b, num_classes)
    print(f"Output shape: {output.shape}")


if __name__ == "__main__":
    debug_model()
