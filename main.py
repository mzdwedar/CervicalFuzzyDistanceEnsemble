import argparse
import numpy as np
from utils.generate_csv import generate_csv
from utils.train import train

parser = argparse.ArgumentParser()
parser.add_argument('--num_epochs', type=int, default = 70,
                    help='Number of Epochs for training')
parser.add_argument('--path', type=str, default = './',
                    help='Path where the image data is stored')
parser.add_argument('--batch_size', type=int, default = 16,
                    help='Batch Size for Mini Batch Training')
parser.add_argument('--kfold', type=int, default = 5,
                    help='Number of folds for training')
parser.add_argument('--lr', type=float, default = 1e-4,
                    help='Learning rate for training')
args = parser.parse_args()

train_df, val_df = generate_csv(args.path)

train(train_df, val_df,
    "InceptionV3" , "MobileNetV2" ,"InceptionResNetV2",
    NUM_EPOCHS = args.num_epochs , train_batch=args.batch_size ,
    validation_batch = args.batch_size, lr=args.lr)
