from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

IMAGE_WIDTH = 256
IMAGE_HEIGHT = 240
BATCH_SIZE = 16
DATA_DIR = Path("/home/groups/comp3710/ADNI/AD_NC")
CLASS_TO_LABEL = {"AD": 0, "NC": 1}


class BrainSliceDataset(Dataset):
    def __init__(
        self, split, image_size=(IMAGE_WIDTH, IMAGE_HEIGHT), data_dir=DATA_DIR
    ):
        self.image_size = image_size
        self.data_dir = Path(data_dir)
        split_dir = self.data_dir / split
        self.samples = sorted(
            (path, label)
            for class_name, label in CLASS_TO_LABEL.items()
            for path in (split_dir / class_name).glob("*.jpeg")
        )
        if not self.samples:
            raise FileNotFoundError(f"No JPEG images found under {split_dir}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        image_path, label = self.samples[index]
        image = Image.open(image_path).convert("L")
        if image.size != self.image_size:
            image = image.resize(self.image_size, Image.BILINEAR)
        image_array = np.asarray(image, dtype=np.float32) / 255.0
        image_tensor = torch.from_numpy(image_array).unsqueeze(0)
        return image_tensor, label


def load_dataset(split, batch_size=BATCH_SIZE, data_dir=DATA_DIR):
    if split not in ("train", "test"):
        raise ValueError("split must be 'train' or 'test'")

    dataset = BrainSliceDataset(
        split, image_size=(IMAGE_WIDTH, IMAGE_HEIGHT), data_dir=data_dir
    )
    return DataLoader(dataset, batch_size=batch_size, shuffle=(split == "train"))
