import numpy as np
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image

class NpyDataLoader(Dataset):
    def __init__(self, image_file, label_file, transform=None):
        """
        Custom dataset that loads images and labels from .npy files.
        
        Args:
            image_file (str): Path to the .npy file that contains images.
            label_file (str): Path to the .npy file that contains labels.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        # Load data from .npy files
        self.images = np.load(image_file)
        self.labels = np.load(label_file)
        print(f"Loaded {len(self.images)} images.")
        self.transform = transform

    def __len__(self):
        """Return the total number of images."""
        return len(self.images)

    def __getitem__(self, idx):
        """
        Fetches the image and label at the given index and applies the transformation.
        
        Args:
            idx (int): Index of the data to fetch.
        
        Returns:
            tuple: (image, label) where image is the transformed image, and label is the corresponding label.
        """
        # Load image and label
        image = self.images[idx].squeeze()
        label = self.labels[idx]

        # Convert numpy array to PIL Image to apply transformation
        image = Image.fromarray(image)
        if self.transform:
            image = self.transform(image)

        return image, label