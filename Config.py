import tensorflow as tf

IMG_WIDTH = 256
IMG_HEIGHT = 256
EPOCHS = 10
BATCH_SIZE = 1
DATA_ROOT_DIR = "./Datasets/"
MODEL_FOLDER = "summer2winter_yosemite/"
SAVED_MODEL_DIR = "./Saved_Model/"
OUTPUT_DIR = "./Outputs/"
SAVE_MODEL = True
LOAD_MODEL = False
LEARNING_RATE = 2e-4
CYCLE_LOSS_WEIGHT = 10.0
ID_LOSS_WEIGHT = 0.0

