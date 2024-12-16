# engine.py

import logging
from types import SimpleNamespace

import torch
import torch.optim as optim
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

import config
from models import ResNet18
from utils import get_data

cfg = SimpleNamespace(**vars(config))


def train_one_epoch(model, optimizer, dataloader, loss_func):
    model.train()
    total_losses = 0
    total_len = 0
    total_out = []
    total_label = []
    for images, targets in dataloader:
        optimizer.zero_grad()
        images = images.to(cfg.device)
        targets = targets.to(cfg.device)

        out = model(images)
        loss = loss_func(out, targets)
        loss.backward()
        optimizer.step()

        total_losses = total_losses + loss.item()
        total_len = total_len + len(images)
        total_out.append(out)
        total_label.append(targets)
    total_out = torch.cat(total_out, dim=0)
    total_label = torch.cat(total_label, dim=0)
    accuracy, precision, recall, f1 = evaluate_result(total_out, total_label)
    ave_losses = total_losses / total_len
    return ave_losses, accuracy, precision, recall, f1


def evaluate_result(out, label):
    _, predicted = torch.max(out, 1)  # 每行选择最大值的索引（类别）

    # 转换为numpy数组，确保在计算时数据类型一致
    predicted = predicted.cpu().numpy()
    label = label.detach().cpu().numpy()
    accuracy = accuracy_score(predicted, label)
    precision = precision_score(
        label, predicted, average="weighted"
    )  # 平均方法：加权平均
    recall = recall_score(label, predicted, average="weighted")
    f1 = f1_score(label, predicted, average="weighted")
    return accuracy, precision, recall, f1


def evaluate_one_epoch(model, val_loader, loss_func):
    model.eval()
    total_losses = 0
    total_len = 0
    total_out = []
    total_label = []
    with torch.no_grad():
        for images, targets in val_loader:
            images = images.to(cfg.device)
            targets = targets.to(cfg.device)

            out = model(images)
            loss = loss_func(out, targets)

            total_losses = total_losses + loss.item()
            total_len = total_len + len(images)
            total_out.append(out)
            total_label.append(targets)
    ave_losses = total_losses / total_len
    total_out = torch.cat(total_out, dim=0)
    total_label = torch.cat(total_label, dim=0)
    accuracy, precision, recall, f1 = evaluate_result(total_out, total_label)

    return ave_losses, accuracy, precision, recall, f1


def train():
    num_classes = cfg.num_classes
    model = ResNet18(num_classes)
    model.to(cfg.device)

    train_loader, val_loader = get_data()

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.AdamW(params, lr=0.005, weight_decay=1e-3, eps=1e-8)

    lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.9)

    num_epochs = 1000
    loss_func = torch.nn.CrossEntropyLoss()
    for epoch in range(num_epochs):
        loss, accuracy, precision, recall, f1 = train_one_epoch(
            model, optimizer, train_loader, loss_func
        )
        lr_scheduler.step()
        logging.info(
            f"Train Epoch:{epoch}: loss:{loss:.4f}, accuracy:{accuracy:.4f}, precision:{precision:.4f}, recall:{recall:.4f}, f1:{f1:.4f}"
        )
        loss, accuracy, precision, recall, f1 = evaluate_one_epoch(
            model, val_loader, loss_func
        )
        logging.info(
            f"Evaluate Epoch:{epoch}: loss:{loss:.4f}, accuracy:{accuracy:.4f}, precision:{precision:.4f}, recall:{recall:.4f}, f1:{f1:.4f}"
        )
    print("Training complete.")
