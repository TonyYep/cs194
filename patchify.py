import torch
import torch.nn.functional as F


def patchify(im: torch.Tensor):
    c, h, w = im.shape  # Shape of image should be channel, height, width

    patch_h, patch_w = h // 2, w // 2
    patches = []
    for i in range(0, h - patch_h, patch_h // 2):
        for j in range(0, w, patch_w):
            patch = im[
                :, i * patch_h : (i + 1) * patch_h, j * patch_w : (j + 1) * patch_w
            ]
            patches.append(patch)
    return patches


def merge_image(image0, image1, coor0, coor1, shape0, shape1, label0, label1):
    """
    Assume that we preserve the image0's background and cut image1's most important part to mix into image0
    Assume that labels are one-hot format
    """
    assert len(image0.shape) == 3, "No batch dimension"

    blend_image = image0.clone()
    image1_patch = image1[
        :, coor1[0] : coor1[0] + shape1[0], coor1[1] : coor1[1] + shape1[1]
    ]

    image1_patch = F.interpolate(image1_patch, size=shape0, align_corners=True)
    blend_image[:, coor0[0] : coor0[0] + shape0[0], coor0[1] : coor0[1] + shape0[1]] = (
        image1_patch
    )
    coef = shape0[0] * shape0[1] / (image0.shape[1] * image0.shape[2])
    label = (1 - coef) * label0 + coef * label1
    return blend_image, label


def label2onehot(labels, num_classes=None):
    """
    Convert batch of label indices to one-hot encoded targets.
    Args:
        labels: tensor of labels, shape [batch_size, 1]
        num_classes: total number of classes. If None, it defaults to the maximum label + 1.
    Returns:
        one_hot_labels: one-hot encoded labels, shape [batch_size, num_classes]
    """
    labels = (
        labels.squeeze()
    )  # Remove any extra dimensions to fit F.one_hot requirements
    if num_classes is None:
        num_classes = labels.max().item() + 1  # Assume labels are 0-indexed
    one_hot_labels = F.one_hot(labels, num_classes=num_classes)
    return one_hot_labels
