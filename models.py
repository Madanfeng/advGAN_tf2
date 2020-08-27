import tensorflow as tf
from tensorflow.keras import Model
from config import cfg

INPUT_SHAPE = [cfg.IMAGE_SIZE, cfg.IMAGE_SIZE, cfg.NUM_CHANNELS]


class BatchNormalization(tf.keras.layers.BatchNormalization):
    """
    "Frozen state" and "inference mode" are two separate concepts.
    `layer.trainable = False` is to freeze the layer, so the layer will use
    stored moving `var` and `mean` in the "inference mode", and both `gama`
    and `beta` will not be updated !
    """
    def call(self, x, training=False):
        if not training:
            training = tf.constant(False)
        training = tf.logical_and(training, self.trainable)
        return super().call(x, training)


def convolutional(input_layer, filters_shape, downsample=False, activate=True, bn=True):
    if downsample:
        input_layer = tf.keras.layers.ZeroPadding2D(((1, 0), (1, 0)))(input_layer)
        padding = 'valid'
        strides = 2
    else:
        strides = 1
        padding = 'same'

    conv = tf.keras.layers.Conv2D(filters=filters_shape[-1], kernel_size=filters_shape[0], strides=strides, padding=padding,
                                  use_bias=not bn, kernel_regularizer=tf.keras.regularizers.l2(0.0005),
                                  kernel_initializer=tf.random_normal_initializer(stddev=0.01))(input_layer)

    if bn: conv = BatchNormalization()(conv)
    if activate == True: conv = tf.nn.leaky_relu(conv, alpha=0.1)

    return conv


def residual_block(input_layer, input_channel, filter_num1, filter_num2):
    short_cut = input_layer
    conv = convolutional(input_layer, filters_shape=(1, 1, input_channel, filter_num1))
    conv = convolutional(conv       , filters_shape=(3, 3, filter_num1,   filter_num2))

    residual_output = short_cut + conv
    return residual_output


def upsample(input_layer, filters_shape, activate=True, bn=True):
    conv = tf.keras.layers.Conv2DTranspose(filters=filters_shape[-1], kernel_size=filters_shape[0], strides=2, padding='same',
                                           kernel_initializer=tf.random_normal_initializer(0., 0.01),
                                           bias_initializer=tf.constant_initializer(0.),
                                           use_bias=False)(input_layer)
    if bn: conv = BatchNormalization()(conv)

    if activate == True: conv = tf.nn.leaky_relu(conv, alpha=0.1)
    return conv


# target model
class target_model():
    # load_model from trained .hdf5
    def __init__(self, model_path=cfg.MODEL_PATH):
        # self.image_size = cfg.IMAGE_SIZE
        # self.num_channels = cfg.NUM_CHANNELS
        # self.num_labels = cfg.NUM_CLASS
        self.model = tf.keras.models.load_model(model_path, compile=False)
        self.model_logits = Model(inputs=self.model.input, outputs=self.model.layers[-2].output)
        # self.model.summary()

    def predict_logits(self, imgs):
        logits = self.model_logits(imgs)
        probs = tf.nn.sigmoid(logits)
        return logits, probs

    def predict_softmax(self, imgs):
        return self.model(imgs)


# Generator
def Generator():

    # (bs, 320, 320, 3)
    inp = tf.keras.layers.Input(shape=INPUT_SHAPE, name='input_image')
    # downsample
    x = convolutional(inp, (3, 3, 3, 32), downsample=True, bn=False)  # (bs, 160, 160, 32)
    x = convolutional(x, (3, 3, 32, 64), downsample=True)    # (bs, 80, 80, 64)
    x = convolutional(x, (3, 3, 64, 128), downsample=True)   # (bs, 40, 40, 128)
    x = convolutional(x, (3, 3, 128, 256), downsample=True)  # (bs, 20, 20, 256)
    x = convolutional(x, (3, 3, 256, 512), downsample=True)  # (bs, 10, 10, 512)

    # residual_block
    for _ in range(5):
        x = residual_block(x, 512, 256, 512)

    # upsample
    x = upsample(x, (3, 3, 512, 256))  # (bs, 20, 20, 256)
    x = upsample(x, (3, 3, 256, 128))  # (bs, 40, 40, 128)
    x = upsample(x, (3, 3, 128, 64))   # (bs, 80, 80, 64)
    x = upsample(x, (3, 3, 64, 32))    # (bs, 160, 160, 32)
    x = upsample(x, (3, 3, 32, 3), bn=False)     # (bs, 320, 320, 3)

    return tf.keras.Model(inputs=inp, outputs=x)


# Discriminator
def Discriminator():

    inp = tf.keras.layers.Input(shape=INPUT_SHAPE, name='input_image')
    # (bs, 320, 320, 3)
    x = convolutional(inp, (3, 3, 3, 32), downsample=True, bn=False)  # (bs, 160, 160, 32)
    x = convolutional(x, (3, 3, 32, 64), downsample=True)   # (bs, 80, 80, 64)
    x = convolutional(x, (3, 3, 64, 128), downsample=True)  # (bs, 40, 40, 128)
    x = convolutional(x, (3, 3, 128, 256), downsample=True)  # (bs, 20, 20, 256)
    x = convolutional(x, (3, 3, 256, 512), downsample=True)  # (bs, 10, 10, 512)

    for _ in range(2):
        x = residual_block(x, 512, 256, 512)

    x = convolutional(x, (3, 3, 512, 256))  # (bs, 10, 10, 256)
    x = convolutional(x, (3, 3, 256, 128))  # (bs, 10, 10, 128)
    x = convolutional(x, (3, 3, 128, 64), bn=False)  # (bs, 10, 10, 64)

    x = tf.keras.layers.Flatten()(x)  # (bs, 10*10*64)
    x = tf.keras.layers.Dense(units=1024, activation="relu")(x)  # (bs, 1024)
    x = tf.keras.layers.Dense(units=512, activation="relu")(x)   # (bs, 512)
    logits = tf.keras.layers.Dense(units=1)(x)  # (bs, 1)
    # probs = tf.nn.sigmoid(logits)

    return tf.keras.Model(inputs=inp, outputs=logits)
