import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from model import MainModel
from utils import check_accuracy, load_checkpoint, save_checkpoint, make_prediction
from transforms import transform
from dataloader import DataFolder
import argparse
import os
from torch.utils.tensorboard import SummaryWriter

device = "cuda" if torch.cuda.is_available() else "cpu"


def train_fn(loader, model, optimizer, loss_fn, scaler, device, epoch, log_dir="logs"):

    #writing for tensorboard visualization
    writer = SummaryWriter(log_dir)
    running_losses = 0.0

    tk0 = tqdm(loader)
    running_accuracy = 0.0
    for batch_idx, (data, targets) in enumerate(tk0):
        # Get data to cuda if possible
        data = data.to(device=device)
        targets = targets.to(device=device)

        # forward
        with torch.cuda.amp.autocast():
            scores = model(data)
            loss = loss_fn(scores.float(), targets)

            max_score = torch.argmax(scores, dim=1)
            predictions = max_score
            num_correct = (predictions == targets).sum().to('cpu').numpy()
        
        running_accuracy += num_correct

        # backward
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        tk0.set_postfix(loss=loss.item(), accuracy=num_correct/loader.batch_size)
        
        running_losses += loss.item()

        if batch_idx % 30 == 29:
            writer.add_scalar('training loss',
                            running_losses / 30,
                            epoch * len(loader) + batch_idx)
            
            writer.add_scalar('training accuracy',
                            running_accuracy / (30*loader.batch_size),
                            epoch * len(loader) + batch_idx)
            running_losses = 0.0
            running_accuracy = 0.0 


def main(
    train_dir,val_dir, checkpoint_dir,
    batch_size, image_size=512, num_epochs=10, checkpoint_name=None, 
    num_workers=1, pin_memory=True, log_dir="logs", model_name=None):
    
    # declare datasets
    # train_ds = DataFolder(root_dir=train_dir, transform=transform(image_size, is_training=True))
    # val_ds = DataFolder(root_dir=val_dir, transform=transform(image_size, is_training=False))
    # train_loader = DataLoader(train_ds, batch_size=batch_size, num_workers=num_workers,pin_memory=pin_memory, shuffle=True)
    # val_loader = DataLoader(val_ds, batch_size=batch_size, num_workers=num_workers,pin_memory=pin_memory,shuffle=True)

    #init model
    model = MainModel(128, model_name)

    # configure parameter
    loss_fn = nn.CrossEntropyLoss()
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    scaler = torch.cuda.amp.GradScaler()
    checkpoint = {'state_dict': model.state_dict(), 'optimizer': optimizer.state_dict()}
    save_checkpoint(checkpoint, os.path.join(checkpoint_dir, f"checkpoint_initialilze.pth.tar"))
    return

    if checkpoint_name:
        ckp_path = os.path.join(checkpoint_dir, checkpoint_name)
        load_checkpoint(torch.load(ckp_path), model, optimizer)

    # check_accuracy(train_loader, model, device)

    #training
    for epoch in range(num_epochs):
        train_fn(train_loader, model, optimizer, loss_fn, scaler, device, epoch, log_dir=log_dir)
        check_accuracy(val_loader, model, device)
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
    parser.add_argument("--model_name", default="../logs", type=str)
    parser.add_argument("--image_size", default=512, type=int)
    args = vars(parser.parse_args())
    main(**args)