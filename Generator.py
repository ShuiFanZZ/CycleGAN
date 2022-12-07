import tensorflow as tf
from tensorflow import keras
import keras.layers as layers
import tensorflow_addons as tfa


class Generator(keras.Model):
    def __init__(self, dim=64, channel=3):
        super(Generator, self).__init__()
        self.dim = dim
        self.channel = channel

    def call(self, inputs, training=True, mask=None):
        h = inputs

        h = tf.pad(h, paddings=[[0,0], [3,3], [3,3], [0,0]], mode='REFLECT')
        h = layers.Conv2D(self.dim, 7, padding='valid', use_bias=False)(h)
        h = tfa.layers.InstanceNormalization()(h)
        h = tf.nn.relu(h)

        out_dim = self.dim
        for _ in range(2):
            out_dim *= 2
            h = self.conv(h, out_dim)

        for _ in range(9):
            h = self.res_block(h, out_dim)

        for _ in range(2):
            out_dim /= 2
            h = self.conv_tp(h, out_dim)

        h = tf.pad(h, [[0, 0], [3, 3], [3, 3], [0, 0]], mode='REFLECT')
        h = keras.layers.Conv2D(self.channel, 7, padding='valid')(h)
        h = tf.tanh(h)

        return h


    def conv(self, x, out_dim):
        h = layers.Conv2D(out_dim, 3, strides=2, padding='same', use_bias=False)(x)
        h = tfa.layers.InstanceNormalization()(h)
        return tf.nn.relu(h)

    def conv_tp(self, x, out_dim):
        h = layers.Conv2DTranspose(out_dim, 3, strides=2, padding='same', use_bias=False)(x)
        h = tfa.layers.InstanceNormalization()(h)
        return tf.nn.relu(h)

    def res_block(self, x, dimension):
        h = x

        h = tf.pad(h, paddings=[[0,0], [1,1], [1,1], [0,0]], mode='REFLECT')
        h = layers.Conv2D(dimension, 3, padding='valid', use_bias=False)(h)
        h = tfa.layers.InstanceNormalization()(h)
        h = tf.nn.relu(h)

        h = tf.pad(h, paddings=[[0, 0], [1, 1], [1, 1], [0, 0]], mode='REFLECT')
        h = layers.Conv2D(dimension, 3, padding='valid', use_bias=False)(h)
        h = tfa.layers.InstanceNormalization()(h)

        return layers.add([x,h])

    def model(self,shape=(256,256,3)):
        x = keras.Input(shape=shape)
        return keras.Model(inputs=x, outputs=self.call(x))


def test():
    model = Generator().model()
    model.summary()


if __name__ == "__main__":
    test()