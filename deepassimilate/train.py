from .loss import masked_mse_loss

def train(model, train_dataloader, val_dataloader, criterion, optimizer, device):
    model.train()
    train_loss = 0.0
    for batch in train_dataloader:
        lr, hr, station = batch
        lr, hr, station = lr.to(device), hr.to(device), station.to(device)
        optimizer.zero_grad()

        sr = model(lr)
        loss1 = criterion(sr, hr)
        loss2 = masked_mse_loss(sr, station)
        total_loss = loss1 + loss2
        total_loss.backward()
        optimizer.step()
        train_loss += total_loss.item()

    train_loss /= len(train_dataloader)
    return train_loss
