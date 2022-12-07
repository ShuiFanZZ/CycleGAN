import sys

import keras.optimizers
import tqdm

import Config

import tensorflow as tf
from Discriminator import Discriminator
from Generator import Generator
from dataset import dataset


class CycleGAN:

    def __init__(self, train_data=None, test_data=None):
        self.train_data = train_data
        self.test_data = test_data

        self.Dsc_X = Discriminator().model()
        self.Dsc_Y = Discriminator().model()
        self.Gen_X = Generator().model()
        self.Gen_Y = Generator().model()

        self.D_opt = keras.optimizers.Adam(learning_rate=Config.LEARNING_RATE, beta_1=0.5)
        self.G_opt = keras.optimizers.Adam(learning_rate=Config.LEARNING_RATE, beta_1=0.5)

        self.mse = tf.losses.MeanSquaredError()
        self.l1 = tf.losses.MeanAbsoluteError()

    def D_loss(self, real, fake):
        return self.mse(tf.ones_like(real), real) + self.mse(tf.zeros_like(fake), fake)

    def G_loss(self, fake):
        return self.mse(tf.ones_like(fake), fake)

    def I_loss(self, a, b):
        return self.l1(a, b)

    def train_step(self, X, Y):
        # Train Generator
        with tf.GradientTape() as t:
            X_Y = self.Gen_Y(X, training=True)
            Y_X = self.Gen_X(Y, training=True)
            X_Y_X = self.Gen_X(X_Y, training=True)
            Y_X_Y = self.Gen_Y(Y_X, training=True)
            X_X = self.Gen_X(X, training=True)
            Y_Y = self.Gen_Y(Y, training=True)

            X_fake = self.Dsc_X(Y_X, training=True)
            Y_fake = self.Dsc_Y(X_Y, training=True)

            loss_gen = self.G_loss(X_fake) + self.G_loss(Y_fake) + \
                       (self.I_loss(X, X_Y_X) + self.I_loss(Y, Y_X_Y)) * Config.CYCLE_LOSS_WEIGHT + \
                       (self.I_loss(X, X_X) + self.I_loss(Y, Y_Y)) * Config.ID_LOSS_WEIGHT

        G_grad = t.gradient(loss_gen, self.Gen_X.trainable_variables + self.Gen_Y.trainable_variables)
        self.G_opt.apply_gradients(zip(G_grad, self.Gen_X.trainable_variables + self.Gen_Y.trainable_variables))

        # Train Discriminator
        with tf.GradientTape() as t:
            X_real = self.Dsc_X(X, training=False)
            Y_real = self.Dsc_Y(Y, training=False)
            X_fake = self.Dsc_X(Y_X, training=False)
            Y_fake = self.Dsc_Y(X_Y, training=False)

            loss_dsc = self.D_loss(X_real, X_fake) + self.D_loss(Y_real, Y_fake)

        D_grad = t.gradient(loss_dsc, self.Dsc_X.trainable_variables + self.Dsc_Y.trainable_variables)
        self.D_opt.apply_gradients(zip(D_grad, self.Dsc_X.trainable_variables + self.Dsc_Y.trainable_variables))

        return loss_gen, loss_dsc

    def train(self):
        ep_checkpoint = tf.Variable(-1)
        checkpoint = tf.train.Checkpoint(Dsc_X=self.Dsc_X,
                                         Dsc_Y=self.Dsc_Y,
                                         Gen_X=self.Gen_X,
                                         Gen_Y=self.Gen_Y,
                                         D_opt=self.D_opt,
                                         G_opt=self.G_opt,
                                         ep_checkpoint=ep_checkpoint)
        save_path = Config.OUTPUT_DIR + Config.MODEL_FOLDER + "checkpoint/"
        manager = tf.train.CheckpointManager(checkpoint, save_path, max_to_keep=5)
        if manager.latest_checkpoint is not None:
            checkpoint.restore(manager.latest_checkpoint)
            print("Loaded Checkpoint from " + save_path)

        summary_writer = tf.summary.create_file_writer(Config.OUTPUT_DIR + Config.MODEL_FOLDER + "summary/", max_queue=Config.EPOCHS)

        # Training Loop
        train_data_len = len(self.train_data)
        with summary_writer.as_default():
            for ep in tqdm.trange(Config.EPOCHS, desc="Epoch Loop"):
                if ep <= ep_checkpoint:
                    continue

                ep_checkpoint.assign_add(1)
                loss_gen_total, loss_dsc_total = 0, 0

                for X, Y in tqdm.tqdm(self.train_data, desc="Inner Loop"):
                    loss_gen, loss_dsc = self.train_step(X, Y)
                    loss_gen_total += loss_gen
                    loss_dsc_total += loss_dsc

                tf.print("Generator Loss: ", loss_gen_total/train_data_len, output_stream=sys.stdout)
                tf.print("Discriminator Loss: ", loss_dsc_total/train_data_len, output_stream=sys.stdout)
                tf.summary.scalar(name="G_loss", data=loss_gen_total/train_data_len, step=ep)
                tf.summary.scalar(name="D_loss", data=loss_dsc_total / train_data_len, step=ep)
                manager.save()

    def test_step(self, X, Y):
        X_Y = self.Gen_Y(X, training=False)
        Y_X = self.Gen_X(Y, training=False)
        X_Y_X = self.Gen_X(X_Y, training=False)
        Y_X_Y = self.Gen_Y(Y_X, training=False)
        X_X = self.Gen_X(X, training=False)
        Y_Y = self.Gen_Y(Y, training=False)

        X_fake = self.Dsc_X(Y_X, training=False)
        Y_fake = self.Dsc_Y(X_Y, training=False)

        loss_gen = self.G_loss(X_fake) + self.G_loss(Y_fake) + \
                   (self.I_loss(X, X_Y_X) + self.I_loss(Y, Y_X_Y)) * Config.CYCLE_LOSS_WEIGHT + \
                   (self.I_loss(X, X_X) + self.I_loss(Y, Y_Y)) * Config.ID_LOSS_WEIGHT

        X_real = self.Dsc_X(X, training=True)
        Y_real = self.Dsc_Y(Y, training=True)
        X_fake = self.Dsc_X(Y_X, training=True)
        Y_fake = self.Dsc_Y(X_Y, training=True)

        loss_dsc = self.D_loss(X_real, X_fake) + self.D_loss(Y_real, Y_fake)

        return loss_gen, loss_dsc

    def test(self):
        loss_gen_total = 0
        loss_dsc_total = 0
        for X, Y in tqdm.tqdm(self.test_data, desc="Evaluating Test Images"):
            loss_gen, loss_dsc = self.test_step(X, Y)
            loss_dsc_total += loss_dsc
            loss_gen_total += loss_gen

        return loss_gen_total / len(self.test_data), loss_dsc_total / len(self.test_data)

    def save_model(self):
        tf.saved_model.save(self.Gen_X, Config.SAVED_MODEL_DIR + Config.MODEL_FOLDER + "Gen_X")
        tf.saved_model.save(self.Gen_Y, Config.SAVED_MODEL_DIR + Config.MODEL_FOLDER + "Gen_Y")
        tf.saved_model.save(self.Dsc_X, Config.SAVED_MODEL_DIR + Config.MODEL_FOLDER + "Dsc_X")
        tf.saved_model.save(self.Dsc_Y, Config.SAVED_MODEL_DIR + Config.MODEL_FOLDER + "Dsc_Y")

    def load_model(self):
        self.Gen_X = tf.saved_model.load(Config.SAVED_MODEL_DIR + Config.MODEL_FOLDER + "Gen_X")
        self.Gen_Y = tf.saved_model.load(Config.SAVED_MODEL_DIR + Config.MODEL_FOLDER + "Gen_Y")
        self.Dsc_X = tf.saved_model.load(Config.SAVED_MODEL_DIR + Config.MODEL_FOLDER + "Dsc_X")
        self.Dsc_Y = tf.saved_model.load(Config.SAVED_MODEL_DIR + Config.MODEL_FOLDER + "Dsc_Y")

    def generate_img(self, input_path, output_path, reverse=False):
        if reverse:
            generator = self.Gen_X
        else:
            generator = self.Gen_Y

        img = tf.keras.utils.load_img(input_path,
                                      grayscale=False,
                                      color_mode='rgb',
                                      target_size=(Config.IMG_HEIGHT, Config.IMG_WIDTH),
                                      interpolation='nearest',
                                      keep_aspect_ratio=True)
        img = tf.keras.utils.img_to_array(img)
        img = tf.expand_dims(img, axis=0)
        img = tf.keras.layers.Rescaling(1. / 127.5, offset=-1)(img)
        fake_img = generator(img)
        fake_img = tf.keras.layers.Rescaling(127.5, offset=127.5)(fake_img)
        tf.keras.utils.save_img(output_path, tf.squeeze(fake_img))


def main():
    ds_train = dataset(train=True)
    ds_test = dataset(train=False)
    model = CycleGAN(ds_train, ds_test)
    model.train()


if __name__ == "__main__":
    main()
