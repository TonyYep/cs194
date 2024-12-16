from types import SimpleNamespace

from torch.utils.data import DataLoader
from torchvision import datasets, transforms

import config

cfg = SimpleNamespace(**vars(config))


def get_data():
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Resize((224, 224)),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    # 加载训练集和验证集
    train_dataset = datasets.ImageFolder(root=cfg.train_images, transform=transform)
    val_dataset = datasets.ImageFolder(root=cfg.val_images, transform=transform)

    # 创建DataLoader
    train_loader = DataLoader(train_dataset, batch_size=cfg.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=cfg.batch_size, shuffle=True)
    return train_loader, val_loader
