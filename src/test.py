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
from sklearn.metrics import classification_report

parser = argparse.ArgumentParser()
parser.add_argument("--test_dir", default="../datasets/train", help="test directory")
parser.add_argument("--batch_size", default=2, type=int)
parser.add_argument("--num_workers", default=2, type=int)
parser.add_argument("--checkpoint_path", default=None, type=str)


def main(test_dir, checkpoint_path, batch_size, num_workers=1, pin_memory=True):
    
    # declare datasets
    test_ds = DataFolder(root_dir=test_dir, transform=config.val_transforms, is_test=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, num_workers=num_workers,pin_memory=pin_memory,shuffle=True)
    
    device = config.DEVICE
    #init model
    model = MainModel(test_ds.__num_class__(), 'efficientnet-b3')
    model = model.to(device)

    # load checkpoint
    load_checkpoint(torch.load(checkpoint_path), model)

    model.eval()

    iterator = tqdm(test_loader)

    num_correct = 0
    num_samples = 0

    preds = []
    groundtruths = []
    print(test_ds.class_names)
    

    with torch.no_grad():
        for x, y, image_paths in iterator:
            
            #convert to device
            x = x.to(device=device)
            y = y.to(device=device)

            # inference
            scores = torch.sigmoid(model(x))

            # get prediction
            max_score = torch.argmax(scores, dim=1)

            # add to global comparing value
            preds += max_score.to("cpu").numpy().tolist()
            groundtruths += y.to("cpu").numpy().tolist()

            #calculate score
            predictions = max_score.float()
            num_correct += (predictions == y).sum()
            num_samples += predictions.shape[0]
            iterator.set_postfix(accuracy=f'{float(num_correct) / float(num_samples) * 100:.2f}')
            # break
    print(classification_report(groundtruths, preds, zero_division=0, target_names=test_ds.class_names))
 

if __name__ == "__main__":
    
    args = vars(parser.parse_args())
    main(**args)