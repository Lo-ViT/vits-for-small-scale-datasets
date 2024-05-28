import os
from PIL import Image
from torchvision import datasets, transforms
from torch.utils.data import Dataset
from tqdm import tqdm
class InMemoryImageFolder(Dataset):
    def __init__(self, root, transform=None):
        # Initialize variables
        self.transform = transform
        self.images = []
        self.labels = []

        # Define allowed image extensions
        valid_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.gif')

        # Load all the images and labels into memory
        for class_id, class_name in tqdm(enumerate(sorted(os.listdir(root)))):
            class_dir = os.path.join(root, class_name, "images")
            if os.path.isdir(class_dir):
                for image_name in os.listdir(class_dir):
                    if image_name.lower().endswith(valid_extensions):
                        image_path = os.path.join(class_dir, image_name)
                        try:
                            # Load the image
                            with Image.open(image_path) as img:
                                img = img.convert('RGB')  # Ensure image is RGB

                            # Apply transformation
                            if self.transform is not None:
                                img = self.transform(img)

                            # Append to list
                            self.images.append(img)
                            self.labels.append(class_id)
                        except Exception as e:
                            print(f"Failed to load image {image_path}: {e}")
        print(f"Loaded {len(self.images)} images from {root}")
        
    def __getitem__(self, index):
        # Return the preloaded image and label
        return self.images[index], self.labels[index]

    def __len__(self):
        # Return the total number of images
        return len(self.images)
