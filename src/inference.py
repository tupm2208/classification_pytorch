import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from model import MainModel
from utils import  load_checkpoint
from transforms import transform
import argparse
import os


device = "cuda" if torch.cuda.is_available() else "cpu"

def inference(test_dir, checkpoint_path, device):
    transform = transform(is_training=False)
    
    #init model
    model = MainModel(128, 'efficientnet-b3')
    model = model.to(device)

    # load checkpoint
    load_checkpoint(torch.load(checkpoint_path), model)

    model.eval()

    with torch.no_grad():
        for x, y in iterator:
            
            #convert to device
            x = x.to(device=device)

            # inference
            scores = torch.sigmoid(model(x))

            # get prediction
            max_score = torch.argmax(scores, dim=1)
            preds = max_score.to("cpu").numpy().tolist()
            
 

if __name__ == "__main__":
    
    pass