import logging

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

logging.basicConfig(level=logging.INFO, format="%(message)s")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# images: np.ndarray with shape (N, C, H, W)
# labels: np.ndarray with shape (N,)


class NumpyDataset(Dataset):
    def __init__(self, images, labels, transform=None):
        self.images = images
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]

        image = np.transpose(image, (1, 2, 0))
        image = image.astype(np.uint8)  # 确保图像为 uint8 类型
        image = transforms.ToPILImage()(image)

        if self.transform:
            image = self.transform(image)

        return image, label


class MixupDataset(Dataset):
    def __init__(self, dataset, num_classes, alpha=1.0, mixup_prob=0.1):
        self.dataset = dataset
        self.num_classes = num_classes
        self.alpha = alpha
        self.mixup_prob = mixup_prob

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        x1, y1 = self.dataset[idx]

        if (
            np.random.rand() < self.mixup_prob
        ):  # TODO: mixup strategy should be adjusted, this mixup is based on channel not patches.
            idx2 = np.random.randint(0, len(self.dataset))
            x2, y2 = self.dataset[idx2]

            if self.alpha > 0:
                lam = np.random.beta(self.alpha, self.alpha)
            else:
                lam = 1.0
            x = lam * x1 + (1 - lam) * x2

            y1_onehot = torch.zeros(self.num_classes)
            y1_onehot[y1] = 1.0
            y2_onehot = torch.zeros(self.num_classes)
            y2_onehot[y2] = 1.0
            y = lam * y1_onehot + (1 - lam) * y2_onehot
        else:
            x = x1
            y = torch.zeros(self.num_classes)
            y[y1] = 1.0
        return x, y


# class SimpleCNN(nn.Module):
#     def __init__(self, num_classes=10):
#         super(SimpleCNN, self).__init__()
#         self.features = nn.Sequential(
#             nn.Conv2d(3, 32, kernel_size=3, padding=1),  # 输入通道数为 3（RGB）
#             nn.BatchNorm2d(32),
#             nn.ReLU(inplace=True),
#             nn.MaxPool2d(kernel_size=2, stride=2),  # 输出大小：32 x H/2 x W/2
#             nn.Conv2d(32, 64, kernel_size=3, padding=1),
#             nn.BatchNorm2d(64),
#             nn.ReLU(inplace=True),
#             nn.MaxPool2d(kernel_size=2, stride=2),  # 输出大小：64 x H/4 x W/4
#             nn.Conv2d(64, 128, kernel_size=3, padding=1),
#             nn.BatchNorm2d(128),
#             nn.ReLU(inplace=True),
#             nn.AdaptiveAvgPool2d((1, 1)),  # 输出大小：128 x 1 x 1
#         )
#         self.classifier = nn.Sequential(
#             nn.Dropout(0.5),
#             nn.Linear(128, num_classes),
#         )

#     def forward(self, x):
#         x = self.features(x)
#         x = x.view(x.size(0), -1)  # 展平
#         x = self.classifier(x)
#         return x


def compute_metrics(outputs, labels):
    prob = nn.functional.softmax(outputs, dim=1)
    preds = torch.argmax(prob, dim=1)

    labels_np = labels.cpu().numpy()
    preds_np = preds.cpu().numpy()
    probs_np = prob.cpu().numpy()

    accuracy = accuracy_score(labels_np, preds_np)
    f1 = f1_score(labels_np, preds_np, average="weighted")

    # AUC for multi-class classification
    try:
        auc = roc_auc_score(labels_np, probs_np, multi_class="ovr")
    except ValueError:
        auc = float("nan")
    return accuracy, f1, auc


def evaluate_model(model, dataloader):
    model.eval()
    all_outputs = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            all_outputs.append(outputs)
            all_labels.append(labels)

    all_outputs = torch.cat(all_outputs)
    all_labels = torch.cat(all_labels)
    accuracy, f1, auc = compute_metrics(all_outputs, all_labels)

    return accuracy, f1, auc


def cross_entropy_loss_with_soft_labels(logits, soft_labels):
    log_probs = nn.functional.log_softmax(logits, dim=1)
    loss = -torch.sum(soft_labels * log_probs, dim=1).mean()
    return loss


def main():
    images, labels = [], []

    transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0, 0, 0],
                std=[1, 1, 1],
            ),
        ]
    )

    train_images, test_images, train_labels, test_labels = train_test_split(
        images, labels, test_size=0.2, random_state=2024, stratify=labels
    )

    train_dataset = NumpyDataset(train_images, train_labels, transform=transform)
    test_dataset = NumpyDataset(test_images, test_labels, transform=transform)

    num_classes = len(np.unique(labels))

    mixup_alpha = 1.0
    mixup_prob = 0.1
    train_dataset = MixupDataset(
        train_dataset, num_classes=num_classes, alpha=mixup_alpha, mixup_prob=mixup_prob
    )

    batch_size = 32
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    model = []  # TODO: Introduce model
    model = model.to(device)

    optimizer = optim.AdamW(model.parameters(), lr=1e-3, eps=1e-8, weight_decay=1e-3)

    num_epochs = 1000
    best_accuracy = 0.0

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0

        for inputs, targets in train_loader:
            inputs = inputs.to(device)
            targets = targets.to(device)

            outputs = model(inputs)
            loss = cross_entropy_loss_with_soft_labels(outputs, targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)

        epoch_loss = running_loss / len(train_dataset)
        logging.info(f"Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}")
        accuracy, f1, auc, sensitivity, specificity = evaluate_model(model, test_loader)
        logging.info(
            f"Validation Accuracy: {accuracy*100:.2f}%, F1 Score: {f1:.4f}, AUC: {auc:.4f}"
        )
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            torch.save(model.state_dict(), "best_model.pth")

    logging.info("Training complete.")
    logging.info(f"Best Validation Accuracy: {best_accuracy*100:.2f}%")

    model.load_state_dict(torch.load("best_model.pth"))
    accuracy, f1, auc, sensitivity, specificity = evaluate_model(model, test_loader)
    logging.info("Test Set Performance:")
    logging.info(f"Accuracy: {accuracy*100:.2f}%")
    logging.info(f"F1 Score: {f1:.4f}")
    logging.info(f"AUC: {auc:.4f}")
    if sensitivity is not None and specificity is not None:
        logging.info(f"Sensitivity: {sensitivity*100:.2f}%")
        logging.info(f"Specificity: {specificity*100:.2f}%")


if __name__ == "__main__":
    main()
