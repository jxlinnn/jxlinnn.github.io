import torch
import torch.utils.data 
from torchvision import datasets
from torchvision.transforms import v2
from typing import Dict, Any, List
import matplotlib.pyplot as pyplot
import pandas as pd
import numpy as np
from PIL import Image

# checks files in directory and returns empty and non-empty image file paths
def load_img_paths(dir):
    empty = []
    img_paths = []
    for fname in os.listdir(dir):
        pth = os.path.join(dir, fname)
        try:
            img = Image.open(pth)
            img_paths.append(pth)
        except:
            print(pth)
            empty.append(pth)
            continue
    return img_paths, empty

# create dictionary that matches image data with corresponding label
def load_img_targets(csv_path: str, img_paths: List) -> Dict[str, Any]:
  image_targets = {}
  df_target = pd.read_csv(csv_path).set_index('image_id')
  for path in img_paths:
    _ = path.split('/')[-1].split('.')[0]
    img_id = int(_)
    image_targets[path] = df_target.loc[img_id].to_list()
  return image_targets

# create train and test data
class CustomImageDataset(torch.utils.data.Dataset):
  def __init__(self, img_paths: List, img_targets: Dict[str, Any]=None):
    self.img_paths = img_paths
    self.transform = v2.Compose([v2.Resize((240,240)), v2.ToTensor()])
    if img_targets is not None: 
        self.img_targets = img_targets
        self.ds_type = "train"
    else: self.ds_type = "test"

  def __len__(self):
    return len(self.img_paths)

  def __getitem__(self, idx):
    img_path = self.img_paths[idx]
    image =  Image.open(img_path)
    image = self.transform(image)
    image = image[:3]
    label = torch.tensor(int(img_path.split('/')[-1].split('.')[0]))
    if self.ds_type == "train":
        outputs = torch.tensor(self.img_targets[img_path])
        return image, label, outputs
    return image, label
  
# data augmentation by incorporating transformations
class TransformedDataset(torch.utils.data.Dataset):
  def __init__(self, img_paths: List, img_targets: Dict[str, Any]=None):
    self.img_paths = img_paths
    self.transform = v2.Compose([
        v2.RandomHorizontalFlip(p=0.5),
        v2.Resize((240,240)),
        v2.ToTensor()])
    if img_targets is not None: 
        self.img_targets = img_targets
        self.ds_type = "train"
    else: self.ds_type = "test"

  def __len__(self):
    return len(self.img_paths)

  def __getitem__(self, idx):
    img_path = self.img_paths[idx]
    image =  Image.open(img_path)
    image = self.transform(image)
    image = image[:3]
    label = torch.tensor(int(img_path.split('/')[-1].split('.')[0]))
    if self.ds_type == "train":
        outputs = torch.tensor(self.img_targets[img_path])
        return image, label, outputs
    return image, label


