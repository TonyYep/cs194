from types import SimpleNamespace

import torch
import torch.nn as nn
from torchvision import models

import config

cfg = SimpleNamespace(**vars(config))


class ResNet18MultiObjectCNN(nn.Module):
    def __init__(self, num_classes, max_objects=6):
        super(ResNet18MultiObjectCNN, self).__init__()

        resnet18 = models.resnet18(pretrained=True)
        self.resnet18 = nn.Sequential(*list(resnet18.children())[:-2])
        self.fc1 = nn.Linear(512 * 7 * 7, 1024)
        self.fc2 = nn.Linear(1024, num_classes * max_objects)
        self.fc3 = nn.Linear(1024, 4 * max_objects)
        self.max_objects = max_objects

    def forward(self, x):
        x = self.resnet18(x)
        x = x.view(x.size(0), -1)

        x = torch.relu(self.fc1(x))

        class_out = self.fc2(x)
        bbox_out = self.fc3(x)

        class_out = class_out.view(-1, self.max_objects, cfg.num_classes)
        bbox_out = bbox_out.view(
            -1, self.max_objects, 4
        )  # [batch_size, max_objects, 4]

        return class_out, bbox_out


class MultiObjectLoss(nn.Module):
    def __init__(self):
        super(MultiObjectLoss, self).__init__()
        self.class_loss = nn.CrossEntropyLoss()
        self.bbox_loss = nn.SmoothL1Loss()

    def forward(self, class_pred, bbox_pred, class_true, bbox_true):
        class_loss = 0
        bbox_loss = 0.0
        batch_size = class_true.size(0)
        for i in range(class_true.size(0)):
            mask = class_true[i] != 0
            if mask.sum() == 0:  # If no valid objects in the sample, skip it
                continue
            i_class_pred = class_pred[i, mask]
            i_bbox_pred = bbox_pred[i, mask]
            class_loss += self.class_loss(
                i_class_pred, class_true[i, mask].to(torch.long)
            )
            # # print(class_true)
            # # print(self.bbox_loss(i_bbox_pred, bbox_true[i, mask]))
            # if torch.isnan(class_loss):
            #     print(i_class_pred)
            #     print(class_true[i, mask])
            bbox_loss += self.bbox_loss(i_bbox_pred, bbox_true[i, mask])

        return (class_loss + bbox_loss) / batch_size


def compute_iou(box1, box2):
    inter_xmin = torch.max(box1[0], box2[0])
    inter_ymin = torch.max(box1[1], box2[1])
    inter_xmax = torch.min(box1[2], box2[2])
    inter_ymax = torch.min(box1[3], box2[3])

    inter_width = torch.clamp(inter_xmax - inter_xmin, min=0)
    inter_height = torch.clamp(inter_ymax - inter_ymin, min=0)
    inter_area = inter_width * inter_height

    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])

    union_area = box1_area + box2_area - inter_area
    iou = inter_area / union_area if union_area > 0 else 0.0
    return iou
