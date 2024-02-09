import torch
import os
from PIL import Image
from torch.utils.data import ConcatDataset, DataLoader, Dataset, random_split, WeightedRandomSampler, BatchSampler
from torchvision.transforms import transforms


class Airogs_Dataset(Dataset):
    def __init__(self, data, label, class_name, data_length, data_dir, train_image_path, is_transform, train_transforms, val_transforms, is_training, image_size):
        super().__init__()
        self.data = data
        self.label = label
        self.class_name = class_name
        self.train_image_path = train_image_path
        self.data_length = data_length
        self.data_dir = data_dir
        self.is_transform = is_transform
        self.train_transforms = train_transforms
        self.val_transforms = val_transforms
        self.is_training = is_training
        self.image_size = image_size

    def __len__(self):
        return self.data_length

    def __getitem__(self, index):
        # Load image using self.df['image'][index], assuming 'image' is the column containing image paths
        image = None
        image_name = self.data[index]
        # Check if the image name ends with specific strings
        if ("color") in self.data[index]:
            # Attempt to open the image with .jpg extension
            image_path = os.path.join(
                self.data_dir,  "AIROGS_2024", "color_aug_images", image_name)
            # Replacing backslashes with forward slashes
            image_path = image_path.replace("\\", "/")
            image = Image.open(image_path).convert('RGB')  # Adjust as needed

        elif "geo" in image_name:
            # Attempt to open the image with .jpg extension
            image_path = os.path.join(
                self.data_dir, "AIROGS_2024",  "geo_aug_images", image_name)
            # Replacing backslashes with forward slashes
            image_path = image_path.replace("\\", "/")
            image = Image.open(image_path).convert('RGB')  # Adjust as needed
        else:
            try:
                # Attempt to open the image with .jpg extension
                image_path = os.path.join(
                    self.train_image_path, image_name + ".jpg")
                # Replacing backslashes with forward slashes
                image_path = image_path.replace("\\", "/")
                image = Image.open(image_path).convert(
                    'RGB')  # Adjust as needed

            except FileNotFoundError:
                try:
                    # If the file with .jpg extension is not found, try to open the image with .png extension
                    image_path = os.path.join(
                        self.train_image_path, image_name + ".png")
                    # Replacing backslashes with forward slashes
                    image_path = image_path.replace("\\", "/")
                    image = Image.open(image_path).convert(
                        'RGB')  # Adjust as needed

                except FileNotFoundError:
                    try:
                        # If the file with .jpg extension is not found, try to open the image with .png extension
                        image_path = os.path.join(
                            self.train_image_path, image_name + ".jpeg")
                        # Replacing backslashes with forward slashes
                        image_path = image_path.replace("\\", "/")
                        image = Image.open(image_path).convert(
                            'RGB')  # Adjust as needed

                    except FileNotFoundError:
                        # Handle the case where both .jpg and .png files are not found
                        print(f"Error: File not found for index {index}")
                        # You might want to return a placeholder image or raise an exception as needed

        # Apply transformations if specified
        if image is not None:
            if self.is_transform:
                if self.is_training:
                    image = self.train_transforms(image)
                else:
                    image = self.val_transforms(image)
            else:
                self.transforms = transforms.Compose([
                    transforms.Resize((self.image_size, self.image_size)),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        (0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
                ])
                image = self.transform(image)

            label = self.label[index]
            label = self.class_name[label].index()
            # Create a one-hot encoded tensor
            if len(self.class_name) > 2:
                one_hot_encoded = torch.zeros(
                    len(self.class_name), dtype=torch.float32)
                one_hot_encoded[label] = torch.ones(1, dtype=torch.float32)
            else:
                one_hot_encoded = torch.tensor(label, dtype=torch.float32)

            return image, one_hot_encoded
        else:
            return None
