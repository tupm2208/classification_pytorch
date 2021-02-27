import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from model import MainModel
from utils import check_accuracy, load_checkpoint, save_checkpoint, make_prediction
import config
from dataloader import DataFolder
import argparse
import os
from torch.utils.tensorboard import SummaryWriter



def train_fn(loader, model, optimizer, loss_fn, scaler, device, epoch, log_dir="logs"):

    #writing for tensorboard visualization
    writer = SummaryWriter(log_dir)
    running_losses = 0.0

    tk0 = tqdm(loader)
    for batch_idx, (data, targets) in enumerate(tk0):
        # Get data to cuda if possible
        data = data.to(device=device)
        targets = targets.to(device=device)

        # forward
        with torch.cuda.amp.autocast():
            scores = model(data)
            loss = loss_fn(scores.float(), targets)

        # backward
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        tk0.set_postfix(loss=loss.item())
        
        running_losses += loss.item()

        if batch_idx % 30 == 29:
            writer.add_scalar('training loss',
                            running_losses / 30,
                            epoch * len(loader) + batch_idx)
            running_losses = 0.0    



def main(train_dir, val_dir, checkpoint_dir, batch_size, checkpoint_name, num_epochs=10, num_workers=1, pin_memory=True, log_dir="logs"):
    
    # declare datasets
    train_ds = DataFolder(root_dir=train_dir, transform=config.train_transforms)
    val_ds = DataFolder(root_dir=val_dir, transform=config.val_transforms)
    train_loader = DataLoader(train_ds, batch_size=batch_size, num_workers=num_workers,pin_memory=pin_memory, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, num_workers=num_workers,pin_memory=pin_memory,shuffle=True)

    #init model
    model = MainModel(train_ds.__num_class__(), 'efficientnet-b3')

    # configure parameter
    loss_fn = nn.CrossEntropyLoss()
    model = model.to(config.DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE)
    scaler = torch.cuda.amp.GradScaler()

    if checkpoint_name:
        ckp_path = os.path.join(checkpoint_dir, checkpoint_name)
        load_checkpoint(torch.load(ckp_path), model, optimizer)

    # make_prediction(model, config.val_transforms, 'test/', config.DEVICE)
    # check_accuracy(train_loader, model, config.DEVICE)

    #training
    for epoch in range(num_epochs):
        train_fn(train_loader, model, optimizer, loss_fn, scaler, config.DEVICE, epoch, log_dir=log_dir)
        check_accuracy(val_loader, model, config.DEVICE)
        checkpoint = {'state_dict': model.state_dict(), 'optimizer': optimizer.state_dict()}
        save_checkpoint(checkpoint, os.path.join(checkpoint_dir, f"checkpoint_{epoch}.pth.tar"))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_dir", default="../datasets/train", help="training directory")
    parser.add_argument("--val_dir", default="../datasets/val", help="validation directory")
    parser.add_argument("--checkpoint_dir", default="../checkpoints", help="checkpoint directory")
    parser.add_argument("--batch_size", default=2, type=int)
    parser.add_argument("--num_epochs", default=10, type=int)
    parser.add_argument("--num_workers", default=2, type=int)
    parser.add_argument("--checkpoint_name", default=None, type=str)
    parser.add_argument("--log_dir", default="../logs", type=str)
    args = vars(parser.parse_args())
    main(**args)