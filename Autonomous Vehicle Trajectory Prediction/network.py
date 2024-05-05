import torch
from torch import nn
from typing import Tuple
from torch.utils.tensorboard import SummaryWriter
import dataloading as dl


class CNNRegressionModel(nn.Module):
  def __init__(self, image_size):
    super(CNNRegressionModel, self).__init__()
    self.counter = 0
    self.image_size = tuple(image_size)
    self.conv1 = nn.Conv2d(in_channels=self.image_size[0], out_channels=24, kernel_size=3, stride=1, padding=1)
    self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
    
    self.conv2 = nn.Conv2d(in_channels=24, out_channels=48, kernel_size=3, stride=1, padding=1)
    self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
    
    self.conv3 = nn.Conv2d(in_channels=48, out_channels=96, kernel_size=3, stride=1, padding=1)
    self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

    
    self.flat = nn.Flatten()
    
    self.fc4 = nn.Linear(in_features=86400, out_features=96)
    self.drop4 = nn.Dropout(0.5)
    self.output = nn.Linear(in_features=96, out_features=2)

  def forward(self, x):
    x = self.conv1(x)
    print(f'conv1 {x.size()}')
    x = nn.functional.relu(x)
    print(f'relu1 {x.size()}')
    x = self.pool1(x)
    print(f'pool1 {x.size()}')
    
    x = nn.functional.relu(self.conv2(x))
    print(f'conv2 {x.size()}')
    x = self.pool2(x)
    print(f'pool2 {x.size()}')
    
    x = nn.functional.relu(self.conv3(x))
    print(f'conv3 {x.size()}')
    x = self.pool3(x)
    
    x = self.flat(x)
    print(f'flat {x.size()}')
    
    x = nn.functional.relu(self.fc4(x))
    print(f'fc4 {x.size()}')
    x = self.drop4(x)
    x = self.output(x)
    return x   

def train_network(device, train_dataloader,  n_epochs: int = 3, image_size: Tuple[int, int, int] = (3, 100, 100)):
  """
  This trains the network for a set number of epochs.
  """
  # Define the model, loss function, and optimizer
  model = CNNRegressionModel(image_size=image_size)
  model.to(device)
  print(model)
  criterion = nn.MSELoss()
  optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

  # Train the model
  writer = SummaryWriter()
  for epoch in range(n_epochs):
      for i, (inputs, label, targets) in enumerate(train_dataloader):
          # Zero the gradients
          optimizer.zero_grad()
    
          # Forward pass
          outputs = model(inputs.to(device))
          loss = criterion(outputs, targets.to(device))
    
          # Backward pass and optimization
          loss.backward()
          optimizer.step()
    
          writer.add_scalar('Train Loss', loss.item(), i)
    
          # Print training statistics
          if (i + 1) % 10 == 0:
              print(f'Epoch [{epoch + 1}/{n_epochs}], Step [{i + 1}/{len(train_dataloader)}], Loss: {loss.item():.4f}')
          if loss > 0.5:
              if self.counter < 3:
                self.counter += 1
                continue
            torch.save(model.state_dict(), f'./cnn_reg_{epoch}.pth')
                
  writer.close()

  return model
