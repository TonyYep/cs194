import os
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
from patchify import patchify

# Define transformations for tiny-imagenet-200
transform = transforms.Compose([
    transforms.Resize((64, 64)),  # tiny-imagenet images are 64x64
    transforms.ToTensor()
])

class ImageFolderWithPaths(datasets.ImageFolder):
    """Custom dataset that includes image file paths. 
    Extends torchvision.datasets.ImageFolder
    """

    def __getitem__(self, index):
        original_tuple = super(ImageFolderWithPaths, self).__getitem__(index)
        path = self.imgs[index][0]
        return (*original_tuple, path)

# Load tiny-imagenet-200 dataset
def load_tiny_imagenet(data_dir, batch_size=64):
    train_dir = os.path.join(data_dir, 'train')
    val_dir = os.path.join(data_dir, 'val')

    train_dataset = ImageFolderWithPaths(root=train_dir, transform=transform)

    val_dataset = ImageFolderWithPaths(root=val_dir, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, len(train_dataset.classes)

# Apply patchify and save patches to files
def save_patches(data_loader, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    for images, labels, filenames in data_loader:
        
        for image, filename in zip(images, filenames):
            # Extract the original filename (without extension for folder name)
            image_name = os.path.splitext(filename)[0]

            # Create a directory for the image's patches
            image_dir = os.path.join(output_dir, image_name)
            os.makedirs(image_dir, exist_ok=True)

            # Generate patches
            patches = patchify(image)

            # Save each patch as an image file
            for i, patch in enumerate(patches):
                patch_image = transforms.ToPILImage()(patch)
                patch_path = os.path.join(image_dir, f"patch_{i}.JPEG")
                patch_image.save(patch_path)

# Path to tiny-imagenet-200 dataset (modify as needed)
data_dir = "tiny-imagenet-200"
output_dir = "patches"  # Directory to save patches

# Load dataset
train_loader, val_loader, num_classes = load_tiny_imagenet(data_dir)

# Save patches for the training set
save_patches(train_loader, output_dir)

print(f"Patches saved to {output_dir}")
