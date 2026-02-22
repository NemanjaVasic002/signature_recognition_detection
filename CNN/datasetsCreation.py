import os
import torch
import pandas as pd
from torchvision import transforms
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import cv2

class SignatureDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None):
        """
        Args:
            csv_file (str): Path to CSV file with filenames and labels.
            root_dir (str): Directory with all the images.
            transform (callable, optional): Optional transform to be applied.
        """
        self.data = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_name = self.data.iloc[idx, 0] + '.jpg'  # Append extension if needed
        #print(f"Loading: {img_name}")              #if you want to see names of the images loaded into dataset
        label_text = self.data.iloc[idx, 1]
        label = 0 if label_text == 'original' else 1  # Map label text to number


        img_path = os.path.join(self.root_dir, img_name)
        image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

        if image is None:
            raise FileNotFoundError(f"Image not found: {img_path}")

        if self.transform:
            image = self.transform(image)
        label = torch.tensor(label, dtype=torch.float32)  # float, not long

        return image, label

transform = transforms.Compose([
    transforms.ToTensor()
])


#to resize all img in folder

def resizeSignatures(input_folder, output_folder, target_height=512, targetWidth=512):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for filename in os.listdir(input_folder):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            img = cv2.imread(os.path.join(input_folder, filename))
            if img is not None:
                new_width = targetWidth
                # Use INTER_AREA for downsizing (best for sharp details)
                resized = cv2.resize(img, (new_width, target_height), interpolation=cv2.INTER_AREA)
                cv2.imwrite(os.path.join(output_folder, filename), resized)
                print(f"Resized {filename} to {new_width}x{target_height}")
            else:
                print(f"Failed to read {filename}")



