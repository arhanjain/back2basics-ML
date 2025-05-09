import cv2
import torch
import numpy as np
import pandas as pd

SPLITS = pd.read_csv("./data/celeb_faces/list_eval_partition.csv")
TRAIN_LIST = SPLITS[SPLITS["partition"] == 0]["image_id"].values
VAL_LIST = SPLITS[SPLITS["partition"] == 1]["image_id"].values

# Resize transform
# Normalize transform

class CelebrityDataset(torch.utils.data.Dataset):
    def __init__(self, file_list, transform=None):
        self.file_list = file_list
        self.transform = transform

    def __len__(self):
        return len(self.file_list)
    
    def __getitem__(self, idx):
        image_path = f"./data/celeb_faces/img_align_celeba/{self.file_list[idx]}"
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        image = np.array(image)
        image = image.astype(np.float32) / 255.0

        if self.transform is not None:
            image = self.transform(image)

        return image


        