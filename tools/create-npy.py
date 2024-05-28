import os
import numpy as np
import zipfile
from PIL import Image
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from tqdm import tqdm

def preprocess_and_save_images(root, save_path, dataset_type, transform=None):
    os.makedirs(save_path, exist_ok=True)  # This line creates the directory if it does not exist

    # Define the dataset using ImageFolder
    dataset = datasets.ImageFolder(
        root=root,
        transform=transform
    )

    # Use DataLoader to handle batching and optional shuffling
    loader = DataLoader(dataset, batch_size=1, shuffle=False)

    images = []
    labels = []

    # Process images
    for image, label in tqdm(loader, desc=f"Processing {dataset_type}"):
        # Convert image from batched tensor to numpy array
        img_array = image.numpy()  # Remove batch dimension and convert to numpy
        images.append(img_array)
        labels.append(label.item())  # Get scalar value

    # Convert lists to NumPy arrays
    images_np = np.array(images)
    labels_np = np.array(labels)

    # Save to .npy files and zip them
    images_file_path = os.path.join(save_path, f'{dataset_type}_images.npy')
    labels_file_path = os.path.join(save_path, f'{dataset_type}_labels.npy')

    np.save(images_file_path, images_np)
    np.save(labels_file_path, labels_np)

# Specify the root directory of your dataset
dataset_root = '/home/long/data/tiny-imagenet-200/'

# Specify where to save the .npy files
npy_save_path = 'tiny-imagenet-200/'

# Process each dataset partition
for dataset_type in ['train', 'val']:  # Add 'test' only if you have a test set
    preprocess_and_save_images(dataset_root, npy_save_path, dataset_type, transforms.Compose([
    np.array,
]))
