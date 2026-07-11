import random
import os
import shutil
import zipfile
import kagglehub
import gdown
import pandas as pd
from PIL import Image
import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split


def dataset_download(dataset_name, data_path="./Data/"):
    if not dataset_name in ["Aptos", "IDRiD", "DDR", "Messidor-2"]:
        print("Invalid Dataset name")
        return
    
    print(f"Downloading {dataset_name} Dataset")
    
    # Aptos Dataset
    if dataset_name == "Aptos":        
        path = kagglehub.dataset_download("mariaherrerot/aptos2019")
        
    # IDRiD Dataset
    elif dataset_name == "IDRiD":        
        file_id = "https://drive.google.com/file/d/1QY2jrzOLf787qH1PrRnohTycAor-UibD/view?usp=sharing"
        zip_path = os.path.join(data_path, "IDRiD_Grading.zip")
        
        # Download
        gdown.download(file_id, output=zip_path,quiet=False)
        
        # Unzip
        print(f"Extracting file {zip_path} ...")
        
        path = os.path.join(data_path, "IDRiD_RAW/")
        with zipfile.ZipFile(zip_path, 'r') as ref:
            ref.extractall(path)
        
        path = os.path.join(path, "B.Disease Grading/")
                
    # DDR Dataset
    if dataset_name == "DDR":
        path = kagglehub.dataset_download("mariaherrerot/ddrdataset")
        
    # Messidor-2 Dataset
    if dataset_name == "Messidor-2":
        path = kagglehub.dataset_download("mariaherrerot/messidor2preprocess")
        
    return path

def dataset_prepare(dataset_name, download_path, save_path="./Data/", data_path="./Data/"):
    dataset_path = os.path.join(save_path, dataset_name)
    images_path = os.path.join(dataset_path, "Images/") 
    
    # Ensure the target image directory exists before copying
    os.makedirs(images_path, exist_ok=True)
    
    print("Preparing Dataset Files...")
    
    # Aptos dataset
    if dataset_name == "Aptos":
        # Merge Images
        shutil.copytree(os.path.join(download_path, "train_images/train_images/"), images_path, dirs_exist_ok=True)
        shutil.copytree(os.path.join(download_path, "test_images/test_images/"), images_path, dirs_exist_ok=True)
        shutil.copytree(os.path.join(download_path, "val_images/val_images/"), images_path, dirs_exist_ok=True)

        # Merge Labels
        df_train = pd.read_csv(os.path.join(download_path, 'train_1.csv'))
        df_test = pd.read_csv(os.path.join(download_path, 'test.csv'))
        df_val = pd.read_csv(os.path.join(download_path, 'valid.csv'))

        # Concatenate all labels
        df = pd.concat([df_train, df_test, df_val], ignore_index=True) 

        # add img extension
        df["file_name"] = df["id_code"].astype(str) + ".png" 

        # Save labels
        df.to_csv(os.path.join(dataset_path, "labels.csv"), index=False)
        
    # IDRiD dataset
    elif dataset_name == "IDRiD":
        # Merge Images
        shutil.copytree(os.path.join(download_path, "1. Original Images/a. Training Set"), images_path, dirs_exist_ok=True)
        shutil.copytree(os.path.join(download_path, "1. Original Images/b. Testing Set"), images_path, dirs_exist_ok=True)
        
        # Merge Labels
        df_train = pd.read_csv(os.path.join(download_path, "2. Groundtruths/a. IDRiD_Disease Grading_Training Labels.csv"))
        df_test = pd.read_csv(os.path.join(download_path, "2. Groundtruths/b. IDRiD_Disease Grading_Testing Labels.csv"))
        
        # Concat training and testing
        df = pd.concat([df_train, df_test], ignore_index=True)
        df = df.rename(columns={'Image name': 'id_code'})
        df = df.rename(columns={'Retinopathy grade': 'diagnosis'})
        
        # add img extension
        df["file_name"] = df["id_code"].astype(str) + ".jpg" 
        
        # Save to csv
        df.to_csv(os.path.join(dataset_path, "labels.csv"), index=False)

    # DDR dataset
    elif dataset_name == "DDR":
        # Copy images
        shutil.copytree(os.path.join(download_path, "DR_grading/DR_grading/"), images_path, dirs_exist_ok=True)

        # Merge Labels
        df = pd.read_csv(os.path.join(download_path, 'DR_grading.csv'))

        # add img extension
        df["file_name"] = df["id_code"].astype(str) + "" 

        # save to csv
        df.to_csv(os.path.join(dataset_path, "labels.csv"), index=False)
    
    # Messidor-2 dataset
    elif dataset_name == "Messidor-2":
        # Merge Images
        shutil.copytree(os.path.join(download_path, "messidor-2/messidor-2/preprocess/"), images_path, dirs_exist_ok=True)

        # Merge Labels
        df = pd.read_csv(os.path.join(download_path, 'messidor_data.csv'))

        # add img extension
        df["file_name"] = df["id_code"].astype(str) + "" 

        # save to csv
        df.to_csv(os.path.join(dataset_path, "labels.csv"), index=False)
        
    return dataset_path

class DRDataset(Dataset):
    def __init__(self, folder_path, file_names, labels, transform=None):
        self.folder_path = folder_path
        # Ensure file_names and labels support integer-location indexing
        if isinstance(file_names, pd.Series):
            self.file_names = file_names.reset_index(drop=True)
        else:
            self.file_names = file_names

        if isinstance(labels, pd.Series):
            self.labels = labels.reset_index(drop=True)
        else:
            self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.file_names)

    def __getitem__(self, idx):
        # Use integer-location indexing for pandas Series, fall back to list/array indexing
        if isinstance(self.file_names, pd.Series):
            file_name = self.file_names.iloc[idx]
        else:
            file_name = self.file_names[idx]

        img_path = os.path.join(self.folder_path, file_name)
        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        if isinstance(self.labels, pd.Series):
            label = self.labels.iloc[idx]
        else:
            label = self.labels[idx]

        return image, label
    
def data_load(dataset_path, img_size, labels_type="Grading"):
    images_path = os.path.join(dataset_path, "Images")
    df = pd.read_csv(os.path.join(dataset_path, 'labels.csv'))
   
    # Load Labels
    y = torch.tensor(df["diagnosis"])

    # Apply Labels Type
    y = torch.where(y == 0, 0, 1) if labels_type == "Binary" else y

    # Define transformations
    ops = [transforms.Resize((img_size, img_size)), transforms.ToTensor()]

    # Apply Zero Centred Normalization
    ops.append(transforms.Normalize(mean=[0.5] * 3, std=[0.5] * 3))

    transform = transforms.Compose(ops)

    # Initialize Dataset 
    dataset = DRDataset(images_path, df["file_name"], y, transform=transform)
    
    return dataset


def plot_img_byClass(samples_dict, samples_per_class=5, figsize=(12, 8)):
    classes = list(samples_dict.keys())
    num_classes = len(classes)

    fig, axes = plt.subplots(nrows=samples_per_class,
                            ncols=num_classes,
                            figsize=figsize)

    # Standardize axes to 2D array
    if num_classes == 1: axes = np.expand_dims(axes, axis=1)
    if samples_per_class == 1: axes = np.expand_dims(axes, axis=0)

    for col, class_name in enumerate(classes):
        images = samples_dict[class_name]
        axes[0, col].set_title(class_name, fontweight='bold')

        for row in range(samples_per_class):
            ax = axes[row, col]

            if row < len(images):
                img = images[row]

                # Convert (C, H, W) -> (H, W, C) for Matplotlib
                if img.ndim == 3:
                    img = img.transpose((1, 2, 0))

                # Clip values if normalized (to prevent matplotlib warnings)
                img = np.clip(img, 0, 1)

                ax.imshow(img, cmap='gray' if img.ndim == 2 else None)

            ax.axis('off')

    plt.tight_layout()
    plt.show()


def get_random_samples_perClass(dataset, keys=None, samples_per_class=5, seed=2026):
    labels = dataset.labels
    
    rng = np.random.default_rng(seed)

    # Ensure labels is a numpy array for easy indexing
    y_np = labels.numpy() if hasattr(labels, 'numpy') else np.array(labels)

    unique_classes = np.unique(y_np)
    class_names = unique_classes if keys is None else keys
    samples_dict = {}

    for cls, name in zip(unique_classes, class_names):
        # 1. Find all indices where the label matches this class
        indices = np.where(y_np == cls)[0]

        # 2. Randomly pick 'n' indices
        n_to_pick = min(len(indices), samples_per_class)
        chosen_indices = rng.choice(indices, size=n_to_pick, replace=False)

        # 3. Fetch images from the dataset one by one
        class_samples = []
        for idx in chosen_indices:
            img, _ = dataset[idx] # Get image from dataset
            class_samples.append(img.numpy()) # Convert tensor to numpy for plotting

        samples_dict[str(name)] = class_samples

    return samples_dict


def data_split(dataset, test_split_ratio=0.2, val_split=False, val_split_ratio=0.1, seed=2026):
    X_train_files, X_test_files, y_train, y_test = train_test_split(
        dataset.file_names, dataset.labels,
        test_size=test_split_ratio,
        random_state=seed,
        stratify= dataset.labels)
    
    if val_split:
        X_train_files, X_val_files, y_train, y_val = train_test_split(
            X_train_files, y_train,
            test_size=val_split_ratio,
            random_state=seed,
            stratify= y_train)
        
    return (X_train_files, X_test_files, y_train, y_test) if not val_split else\
        (X_train_files, X_val_files, X_test_files, y_train, y_val, y_test)
        

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)
        
def create_dataloader(dataset, batch_size=16, shuffle=True, seed=2026, num_workers=None, pin_memory=False):
    g = torch.Generator()
    g.manual_seed(seed)

    # Conservative defaults to avoid shared-memory allocation issues in constrained environments
    if num_workers is None:
        num_workers = min(2, os.cpu_count() or 1)

    dataloader = DataLoader(dataset=dataset,
                            batch_size=batch_size,
                            shuffle=shuffle, generator=g,
                            num_workers=num_workers, pin_memory=pin_memory,
                            worker_init_fn=seed_worker)

    return dataloader