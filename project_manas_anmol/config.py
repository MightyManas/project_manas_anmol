# config.py

# training params
batchsize = 32
epochs = 6
learning_rate = 1e-3

# image params
resize_x = 224
resize_y = 224
input_channels = 3

# dataset
num_classes = 13

# paths
train_dir = "data/train"
val_dir = "data/test"

# device
device = "cuda"  # or "cpu"
