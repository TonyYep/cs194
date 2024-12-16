# utils.py
from types import SimpleNamespace

import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import CocoDetection

import config

cfg = SimpleNamespace(**vars(config))


def get_transform():
    def transforms_fn(img, target):
        # 对img进行处理
        img = transforms.ToTensor()(img)
        _, orig_height, orig_width = img.shape
        img = transforms.Resize((224, 224))(img)
        img = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])(img)
        ratio_w = 224 / orig_width
        ratio_h = 224 / orig_height
        for obj in target:
            x, y, w, h = obj["bbox"]
            x = x * ratio_w
            y = y * ratio_h
            w = w * ratio_w
            h = h * ratio_h
            obj["bbox"] = [x, y, w, h]
        target = process_target(target)
        return img, target

    return transforms_fn


def get_data():
    train_dataset = CocoDetection(
        root=cfg.train_images, annFile=cfg.train_annotations, transforms=get_transform()
    )

    val_dataset = CocoDetection(
        root=cfg.val_images, annFile=cfg.val_annotations, transforms=get_transform()
    )

    train_loader = DataLoader(train_dataset, batch_size=cfg.batch_size, shuffle=True)

    val_loader = DataLoader(val_dataset, batch_size=cfg.batch_size, shuffle=True)
    return train_loader, val_loader


def process_target(target, max_target=6):
    num_target = len(target)
    aligned_target = []
    if num_target > max_target:
        sorted_target = sorted(target, key=lambda x: x["area"], reverse=True)[
            :max_target
        ]
    else:
        sorted_target = target.copy()
    if len(sorted_target) < max_target:
        pad_size = max_target - len(sorted_target)
        for _ in range(pad_size):
            sorted_target.append(
                {
                    "category_id": 0,  # 填充类别为 0（背景）
                    "bbox": [0, 0, 0, 0],  # 填充边界框为 [0, 0, 0, 0]
                    "area": 0,  # 填充面积为 0
                }
            )
    processed = []
    for obj in sorted_target:
        class_id = obj["category_id"]
        bbox = obj["bbox"]  # 假设是 [x_min, y_min, x_max, y_max]
        bbox = [b / 224 for b in bbox]
        class_id = int(class_id)
        processed.append([class_id] + bbox)

    processed_tensor = torch.tensor(processed, dtype=torch.float32)

    return processed_tensor
