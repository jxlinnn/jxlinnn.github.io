import os
import numpy as np
import pandas as pd
import torch
from PIL import Image
import dataloading as dl
import network
from torch.utils.data import DataLoader

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

train_paths, e1 = load_img_paths('./training_data/training_data')
test_paths, e2 = load_img_paths('./test_data/test_data')

def collate_fn(batch):
   batch = list(filter(lambda x: x is not None, batch))
   return torch.utils.data.dataloader.default_collate(batch)

image_targets = dl.load_img_targets('./training_norm.csv', train_paths)
training_data = dl.CustomImageDataset(train_paths, image_targets)
train_dataloader = DataLoader(training_data, batch_size=1, shuffle=True, collate_fn=collate_fn)

train_features, train_labels, train_targets = next(iter(train_dataloader))
print(f"Feature batch shape: {train_features.size()}")
print(f"Labels batch shape: {train_labels.size()}")
print(f"Targets batch shape: {train_targets.size()}")

transformed_data = dl.TranformedDataset(train_paths, image_targets)
trans_dataloader = DataLoader(transformed_data, batch_size=1, shuffle=True, collate_fn=collate_fn)

train_features, train_labels, train_targets = next(iter(trans_dataloader))
print(f"Feature batch shape: {train_features.size()}")
print(f"Labels batch shape: {train_labels.size()}")
print(f"Targets batch shape: {train_targets.size()}")

torch.manual_seed(42)
image_size = train_features[0].size()
if torch.cuda.is_available():
    device = 'cuda'
else:
    device = 'cpu'

n = 40
model = network.train_network(device, train_dataloader, n_epochs=n, image_size=image_size)

checkpt = CNNRegressionModel(image_size)
checkpt.load_state_dict(torch.load('cnn_reg_5.pth', map_location=device))

test_data = dl.CustomImageDataset(test_paths)
test_dataloader = DataLoader(test_data, batch_size=1, collate_fn=collate_fn)

dl_iter = iter(test_dataloader)
predictions = []

for i in range(test_dataloader.__len__()):
  test_input, test_label = next(dl_iter)
  pred = checkpt(test_input.to('cpu'))
  pred = pred.cpu().detach().numpy()[0]
  predictions.append({'image_id': int(test_label), 'angle': pred[0], 'speed': pred[1]})

pred_df = pd.DataFrame(predictions)
pred_df.head()

pred_df.to_csv('/content/results.csv', index=False)

