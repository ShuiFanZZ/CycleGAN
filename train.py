import os.path
from os import mkdir

import keras.optimizers

import Config

import tensorflow as tf
from CycleGAN import CycleGAN
from dataset import dataset


def main():
    if not os.path.isdir(Config.OUTPUT_DIR):
        mkdir(Config.OUTPUT_DIR)
    if not os.path.isdir(Config.SAVED_MODEL_DIR):
        mkdir(Config.SAVED_MODEL_DIR)

    train_data = dataset(train=True)
    test_data = dataset(train=False)
    model = CycleGAN(train_data, test_data)
    if Config.LOAD_MODEL:
        model.load_model()

    model.train()
    loss_gen, loss_dsc = model.test()

    tf.print("G_loss:", loss_gen, ",", "D_loss:", loss_dsc)

    if Config.SAVE_MODEL:
        model.save_model()


if __name__ == "__main__":
    main()
    