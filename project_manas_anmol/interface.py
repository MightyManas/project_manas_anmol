# interface.py

from model import NarutoModel as TheModel
from train import train_model as the_trainer
from predict import predict_images as the_predictor
from dataset import NarutoDataset as TheDataset
from dataset import naruto_loader as the_dataloader

from config import batchsize as the_batch_size
from config import epochs as total_epochs
