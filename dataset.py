import Config
import tensorflow as tf


class dataset:
    def __init__(self, train=True):
        self.train = train
        if self.train:
            self.A = self.load_data("trainA/")
            self.B = self.load_data("trainB/")
        else:
            self.A = self.load_data("testA/", batch_size=1)
            self.B = self.load_data("testB/", batch_size=1)

        self.countA = 0
        self.countB = 0
        for batch in self.A:
            if batch.shape[0] == Config.BATCH_SIZE:
                self.countA += 1
        for batch in self.B:
            if batch.shape[0] == Config.BATCH_SIZE:
                self.countB += 1

        self.count = min(self.countA, self.countB)
        self.idx = 0
        self.iterA = None
        self.iterB = None

    def __len__(self):
        return self.count

    def __iter__(self):
        self.idx = 0
        self.iterA = self.A.__iter__()
        self.iterB = self.B.__iter__()
        return self

    def __next__(self):
        self.idx += 1
        if self.idx <= self.count:
            return self.iterA.__next__(), self.iterB.__next__()
        else:
            raise StopIteration

    def load_data(self, name, batch_size=Config.BATCH_SIZE):
        directory = Config.DATA_ROOT_DIR + Config.MODEL_FOLDER + name

        ds = tf.keras.utils.image_dataset_from_directory(
            directory,
            labels=None,
            color_mode='rgb',
            batch_size=batch_size,
            image_size=(Config.IMG_HEIGHT, Config.IMG_WIDTH),
            shuffle=True,
            seed=None,
            validation_split=None,
            subset=None,
            interpolation='bilinear',
            follow_links=False,
            crop_to_aspect_ratio=True
        )

        # rescale the image from [0, 255] to [-1., 1.]
        norm_layer = tf.keras.layers.Rescaling(1. / 127.5, offset=-1)
        normalized_ds = ds.map(lambda x: norm_layer(x))

        return normalized_ds


def test():
    ds = dataset(train=False)
    print(ds.countA)
    print(ds.countB)
    c = 0
    for batch in ds:
        print(str(batch[0].shape) + " " + str(batch[1].shape))
        c += 1
    print(c)
    c = 0
    for batch in ds:
        print(str(batch[0].shape) + " " + str(batch[1].shape))
        c += 1
    print(c)


if __name__ == "__main__":
    test()
