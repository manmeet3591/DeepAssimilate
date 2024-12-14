import torch
from torch.utils.data import DataLoader
from deepassimilate import SRCNN, ncDataset, train

# Generate random data for demonstration
x_train = torch.rand(100, 1, 32, 32)
y_train = torch.rand(100, 1, 32, 32)
z_train = torch.rand(100, 1, 32, 32)

train_dataset = ncDataset(x_train, y_train, z_train)
train_dataloader = DataLoader(train_dataset, batch_size=10, shuffle=True)

device = "cuda" if torch.cuda.is_available() else "cpu"
model = SRCNN().to(device)
criterion = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters())

# Train for 1 epoch
train(model, train_dataloader, train_dataloader, criterion, optimizer, device)
