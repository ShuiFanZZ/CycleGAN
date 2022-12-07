import tensorflow as tf
from tensorflow import keras
import keras.layers as layers
import tensorflow_addons as tfa


class Discriminator(keras.Model):
    def __init__(self, features=[64, 128, 256, 512]):
        super(Discriminator, self).__init__()
        self.features = features

    def call(self, inputs, training=True, mask=None):
        # No instance norm for the first layer C64 as mentioned in the paper
        x = self.conv(inputs, self.features[0], stride=2, norm=False)
        x = self.conv(x, self.features[1], stride=2)
        x = self.conv(x, self.features[2], stride=2)
        x = self.conv(x, self.features[3], stride=1)
        return layers.Conv2D(1, 4, strides=1, padding='same')(x)

    def conv(self, x, dimension, stride, norm=True):
        h = layers.Conv2D(dimension, kernel_size=4, strides=stride, padding='same', use_bias=True)(x)
        if norm:
            h = tfa.layers.InstanceNormalization()(h)
        h = tf.nn.leaky_relu(h, alpha=0.2)
        return h

    def model(self, shape=(256, 256, 3)):
        x = keras.Input(shape=shape)
        return keras.Model(inputs=x, outputs=self.call(x))


def test():
    model = Discriminator().model()
    model.summary()


if __name__ == "__main__":
    test()
