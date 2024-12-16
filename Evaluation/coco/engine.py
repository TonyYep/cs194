# engine.py

import logging
from types import SimpleNamespace

import torch
import torch.optim as optim

import config
from models.backbone import MultiObjectLoss, ResNet18MultiObjectCNN
from utils import get_data

cfg = SimpleNamespace(**vars(config))


def train_one_epoch(model, optimizer, dataloader, loss_func):
    model.train()
    total_losses = 0
    total_len = 0
    total_pre = 0
    total_rec = 0
    total_iou = 0
    times = 0
    for images, targets in dataloader:
        optimizer.zero_grad()
        images = images.to(cfg.device)
        targets = targets.to(cfg.device)

        class_pred, bbox_pred = model(images)
        loss = loss_func(class_pred, bbox_pred, targets[:, :, 0], targets[:, :, 1:])
        loss.backward()
        optimizer.step()

        total_losses = total_losses + loss.item()
        total_len = total_len + len(images)
        precision, recall, average_iou = evaluate_result(
            class_pred, bbox_pred, targets[:, :, 0], targets[:, :, 1:]
        )
        total_pre = total_pre + precision
        total_rec = total_rec + recall
        total_iou = average_iou + total_iou
        times = times + 1
    ave_losses = total_losses / total_len
    total_pre = total_pre / times
    total_rec = total_rec / times
    total_iou = total_iou / times

    return ave_losses, total_pre, total_rec, total_iou


# def evaluate_result(class_pred, bbox_pred, class_true, bbox_true, iou_threshold=0.5):
#     true_positive = 0
#     false_positive = 0
#     false_negative = 0
#     iou_list = []

#     batch_size = class_pred.size(0)
#     class_pred = torch.argmax(class_pred, dim=-1)
#     for b in range(batch_size):
#         pred_classes = class_pred[b]
#         true_classes = class_true[b]
#         pred_bboxes = bbox_pred[b]
#         true_bboxes = bbox_true[b]

#         for i in range(len(pred_classes)):
#             pred_class = pred_classes[i]
#             pred_bbox = pred_bboxes[i]

#             # 过滤掉背景预测
#             if pred_class == 0:
#                 continue

#             best_iou = 0
#             best_true_class = 0
#             best_true_bbox = None

#             for j in range(len(true_classes)):
#                 true_class = true_classes[j]
#                 true_bbox = true_bboxes[j]

#                 # 只计算类别相同的框之间的IOU
#                 if true_class == pred_class:
#                     iou = compute_iou(pred_bbox, true_bbox)
#                     if iou > best_iou:
#                         best_iou = iou
#                         best_true_class = true_class
#                         best_true_bbox = true_bbox

#             if best_iou > iou_threshold:
#                 true_positive += 1
#                 iou_list.append(best_iou)
#             else:
#                 false_positive += 1

#         for k in range(len(true_classes)):
#             true_class = true_classes[k]
#             if true_class == 0:
#                 continue

#             matched = False
#             for m in range(len(pred_classes)):
#                 pred_class = pred_classes[m]
#                 pred_bbox = pred_bboxes[m]

#                 if pred_class == true_class:
#                     iou = compute_iou(pred_bbox, true_bboxes[k])
#                     if iou > iou_threshold:
#                         matched = True
#                         break

#             if not matched:
#                 false_negative += 1
#     precision = (
#         true_positive / (true_positive + false_positive)
#         if (true_positive + false_positive) > 0
#         else 0
#     )
#     recall = (
#         true_positive / (true_positive + false_negative)
#         if (true_positive + false_negative) > 0
#         else 0
#     )

#     average_iou = sum(iou_list) / len(iou_list) if len(iou_list) > 0 else 0

#     return precision, recall, average_iou


def evaluate_one_epoch(model, val_loader, loss_func):
    model.eval()
    total_losses = 0
    total_len = 0
    total_pre = 0
    total_rec = 0
    total_iou = 0
    times = 0
    with torch.no_grad():
        for images, targets in val_loader:
            images = images.to(cfg.device)
            targets = targets.to(cfg.device)
            class_pred, bbox_pred = model(images)
            loss = loss_func(class_pred, bbox_pred, targets[:, :, 0], targets[:, :, 1:])
            total_losses = total_losses + loss.item()
            total_len = total_len + len(images)
            precision, recall, average_iou = evaluate_result(
                class_pred, bbox_pred, targets[:, :, 0], targets[:, :, 1:]
            )
            total_pre = total_pre + precision
            total_rec = total_rec + recall
            total_iou = average_iou + total_iou
            times = times + 1
    ave_losses = total_losses / total_len
    total_pre = total_pre / times
    total_rec = total_rec / times
    total_iou = total_iou / times

    return ave_losses, total_pre, total_rec, total_iou


def train():
    num_classes = cfg.num_classes
    model = ResNet18MultiObjectCNN(num_classes)
    model.to(cfg.device)

    train_loader, val_loader = get_data()

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.AdamW(params, lr=0.005, weight_decay=1e-3, eps=1e-8)

    lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.9)

    num_epochs = 1000
    loss_func = MultiObjectLoss()
    for epoch in range(num_epochs):
        loss, precision, recall, iou = train_one_epoch(
            model, optimizer, train_loader, loss_func
        )
        lr_scheduler.step()
        logging.info(
            f"Train Epoch:{epoch}: loss:{loss:.4f}, precision:{precision:.4f}, recall:{recall:.4f}, iou:{iou:.4f}"
        )
        loss, precision, recall, iou = evaluate_one_epoch(model, val_loader, loss_func)
        logging.info(
            f"Evaluate Epoch:{epoch}: loss:{loss:.4f}, precision:{precision:.4f}, recall:{recall:.4f}, iou:{iou:.4f}"
        )
    print("Training complete.")


def compute_iou_matrix(bbox_pred, bbox_true):
    """
    向量化计算IOU:
    bbox_pred: [B, N, 4]
    bbox_true: [B, M, 4]
    返回: iou: [B, N, M]
    """
    # 分解坐标
    pred_xmin = bbox_pred[..., 0].unsqueeze(2)  # [B,N,1]
    pred_ymin = bbox_pred[..., 1].unsqueeze(2)
    pred_xmax = bbox_pred[..., 2].unsqueeze(2)
    pred_ymax = bbox_pred[..., 3].unsqueeze(2)

    gt_xmin = bbox_true[..., 0].unsqueeze(1)  # [B,1,M]
    gt_ymin = bbox_true[..., 1].unsqueeze(1)
    gt_xmax = bbox_true[..., 2].unsqueeze(1)
    gt_ymax = bbox_true[..., 3].unsqueeze(1)

    inter_xmin = torch.max(pred_xmin, gt_xmin)  # [B,N,M]
    inter_ymin = torch.max(pred_ymin, gt_ymin)
    inter_xmax = torch.min(pred_xmax, gt_xmax)
    inter_ymax = torch.min(pred_ymax, gt_ymax)

    inter_w = torch.clamp(inter_xmax - inter_xmin, min=0)
    inter_h = torch.clamp(inter_ymax - inter_ymin, min=0)
    inter_area = inter_w * inter_h

    area_pred = (pred_xmax - pred_xmin) * (pred_ymax - pred_ymin)  # [B,N,1]
    area_gt = (gt_xmax - gt_xmin) * (gt_ymax - gt_ymin)  # [B,1,M]

    union_area = area_pred + area_gt - inter_area
    iou = inter_area / torch.clamp(union_area, min=1e-6)
    return iou


def evaluate_result(class_pred, bbox_pred, class_true, bbox_true, iou_threshold=0.5):
    """
    向量化计算precision, recall, average_iou

    参数:
    class_pred: [B, N, C]
    bbox_pred: [B, N, 4]
    class_true: [B, M]
    bbox_true: [B, M, 4]
    iou_threshold: float

    返回:
    precision, recall, average_iou
    """
    # 获取预测类别
    # class_pred: [B,N,C] => pred_classes: [B,N]
    pred_classes = torch.argmax(class_pred, dim=-1)

    # 背景类假设为0
    # 有效预测框掩码 [B,N]
    pred_mask = pred_classes != 0

    # 计算IOU矩阵 [B,N,M]
    iou = compute_iou_matrix(bbox_pred, bbox_true)

    # 类别匹配掩码: [B,N,M]
    # 如果pred_class与true_class一致则为True
    class_match = pred_classes.unsqueeze(-1) == class_true.unsqueeze(1)

    # 对IOU只保留类别匹配的部分
    iou_class_masked = iou * class_match

    # 只考虑有效预测框（非背景）
    iou_class_masked = iou_class_masked * pred_mask.unsqueeze(-1)

    # 对每个预测框找到最佳匹配的GT框 (最大IOU)
    best_ious, best_gt_idx = iou_class_masked.max(dim=-1)  # [B,N]

    # TP：满足IOU阈值的预测框数
    tp_mask = (best_ious > iou_threshold) & pred_mask
    tp_count = tp_mask.sum().item()

    # FP：预测为非背景但没有匹配上高IOU的GT框
    fp_mask = pred_mask & (~tp_mask)
    fp_count = fp_mask.sum().item()

    # FN：真实框中未被匹配的框
    # 对于每个GT框，看是否有预测框匹配 (iou>iou_threshold且类相同)
    matched_gt_mask = (iou_class_masked > iou_threshold).any(dim=1)  # [B,M]
    # 去除背景GT
    gt_mask = class_true != 0
    fn_mask = gt_mask & (~matched_gt_mask)
    fn_count = fn_mask.sum().item()

    # 计算precision, recall
    precision = tp_count / (tp_count + fp_count) if (tp_count + fp_count) > 0 else 0.0
    recall = tp_count / (tp_count + fn_count) if (tp_count + fn_count) > 0 else 0.0

    # 计算average_iou (仅对TP的预测框)
    tp_ious = best_ious[tp_mask]
    average_iou = tp_ious.mean().item() if tp_ious.numel() > 0 else 0.0

    return precision, recall, average_iou
