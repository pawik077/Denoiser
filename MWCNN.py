import tensorflow as tf
from helpers import *

class Conv_block(tf.keras.layers.Layer):
    def __init__(self, filters=200, kernel_size=3, **kwargs):
        super().__init__(**kwargs)
        self.filters = filters
        self.kernel_size = kernel_size

        self.conv1 = tf.keras.layers.Conv2D(filters=self.filters, kernel_size=self.kernel_size, padding='same')
        self.conv2 = tf.keras.layers.Conv2D(filters=self.filters, kernel_size=self.kernel_size, padding='same')
        self.conv3 = tf.keras.layers.Conv2D(filters=self.filters, kernel_size=self.kernel_size, padding='same')
        self.conv4 = tf.keras.layers.Conv2D(filters=self.filters, kernel_size=self.kernel_size, padding='same')

        self.bn1 = tf.keras.layers.BatchNormalization()
        self.bn2 = tf.keras.layers.BatchNormalization()
        self.bn3 = tf.keras.layers.BatchNormalization()
        self.bn4 = tf.keras.layers.BatchNormalization()
    
    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'num_filters': self.filters,
            'kernel_size': self.kernel_size,
        })
        return config
    
    def call(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = tf.nn.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = tf.nn.relu(x)
        x = self.conv3(x)
        x = self.bn3(x)
        x = tf.nn.relu(x)
        x = self.conv4(x)
        x = self.bn4(x)
        x = tf.nn.relu(x)
        return x
    
class DWT_downsampling(tf.keras.layers.Layer):
    """
    DWT (Discrete Wavelet Transform) function implementation according to
    "Multi-level Wavelet Convolutional Neural Networks"
    by Pengju Liu, Hongzhi Zhang, Wei Lian, Wangmeng Zuo
    https://arxiv.org/abs/1907.03128
    blatantly stolen from
    https://github.com/AureliiiieP/Keras-WaveletTransform/blob/master/models/DWT.py
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def call(self, x):
        df = tf.keras.backend.image_data_format()
        if df == 'channels_last':
            x1 = x[:, 0::2, 0::2, :] # x(2i-1, 2j-1)
            x2 = x[:, 1::2, 0::2, :] # x(2i, 2j-1)
            x3 = x[:, 0::2, 1::2, :] # x(2i-1, 2j)
            x4 = x[:, 1::2, 1::2, :] # x(2i, 2j)
        elif df == 'channels_first':
            x1 = x[:, :, 0::2, 0::2] # x(2i-1, 2j-1)
            x2 = x[:, :, 1::2, 0::2] # x(2i, 2j-1)
            x3 = x[:, :, 0::2, 1::2] # x(2i-1, 2j)
            x4 = x[:, :, 1::2, 1::2] # x(2i, 2j)

        x_LL = x1 + x2 + x3 + x4
        x_LH = -x1 + x2 - x3 + x4
        x_HL = -x1 - x2 + x3 + x4
        x_HH = x1 - x2 - x3 + x4

        if df == 'channels_last':
            return tf.concat([x_LL, x_LH, x_HL, x_HH], axis=-1)
        elif df == 'channels_first':
            return tf.concat([x_LL, x_LH, x_HL, x_HH], axis=1)
        
class IWT_upsampling(tf.keras.layers.Layer):
    """
    IWT (Inverse Discrete Wavelet Transform) function implementation according to
    "Multi-level Wavelet Convolutional Neural Networks"
    by Pengju Liu, Hongzhi Zhang, Wei Lian, Wangmeng Zuo
    https://arxiv.org/abs/1907.03128
    blatantly stolen from
    https://github.com/AureliiiieP/Keras-WaveletTransform/blob/master/models/DWT.py
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def call(self, x):
        df = tf.keras.backend.image_data_format()
        if df == 'channels_last':
            x_LL = x[:, :, :, 0:(x.shape[3] // 4)]
            x_LH = x[:, :, :, (x.shape[3] // 4):(x.shape[3] // 4 * 2)]
            x_HL = x[:, :, :, (x.shape[3] // 4 * 2):(x.shape[3] // 4 * 3)]
            x_HH = x[:, :, :, (x.shape[3] // 4 * 3):]
        elif df == 'channels_first':
            # it may blow up, better use channels_last instead, implemented for completeness
            x_LL = x[:, 0:(x.shape[1] // 4), :, :]
            x_LH = x[:, (x.shape[1] // 4):(x.shape[1] // 4 * 2), :, :]
            x_HL = x[:, (x.shape[1] // 4 * 2):(x.shape[1] // 4 * 3), :, :]
            x_HH = x[:, (x.shape[1] // 4 * 3):, :, :]
        
        x1 = (x_LL - x_LH - x_HL + x_HH) / 4
        x2 = (x_LL - x_LH + x_HL - x_HH) / 4
        x3 = (x_LL + x_LH - x_HL - x_HH) / 4
        x4 = (x_LL + x_LH + x_HL + x_HH) / 4

        if df == 'channels_last':
            y1 = tf.stack([x1,x3], axis=2)
            y2 = tf.stack([x2,x4], axis=2)
        elif df == 'channels_first':
            y1 = tf.stack([x1,x3], axis=3)
            y2 = tf.stack([x2,x4], axis=3)

        shape = tf.shape(x)
        if df == 'channels_last':
            return tf.reshape(tf.concat([y1,y2], axis=-1), tf.stack([shape[0], shape[1] * 2, shape[2] * 2, shape[3] // 4]))
        elif df == 'channels_first':
            return tf.reshape(tf.concat([y1,y2], axis=1), tf.stack([shape[0], shape[1] // 4, shape[2] * 2, shape[3] * 2]))

def MWCNN_model():
    input = tf.keras.layers.Input(shape=(None, None, 3), name='input')
    
    conv1 = Conv_block(filters=64)(input)
    dwt1 = DWT_downsampling()(conv1)
    
    conv2 = Conv_block(filters=128)(dwt1)
    dwt2 = DWT_downsampling()(conv2)
    
    conv3 = Conv_block(filters=256)(dwt2)
    dwt3 = DWT_downsampling()(conv3)
    
    conv4 = Conv_block(filters=512)(dwt3)
    dwt4 = DWT_downsampling()(conv4)
    
    conv5 = Conv_block(filters=512)(dwt4)
    conv5 = tf.keras.layers.BatchNormalization()(conv5)
    conv5 = Conv_block(filters=512)(conv5)
    conv5 = tf.keras.layers.Conv2D(filters=2048, kernel_size=3, strides=1, padding='same')(conv5)

    up1 = IWT_upsampling()(conv5)
    up1 = Conv_block(filters=512)(tf.keras.layers.Add()([up1, conv4]))
    up1 = tf.keras.layers.Conv2D(filters=1024, kernel_size=3, strides=1, padding='same')(up1)
    
    up2 = IWT_upsampling()(up1)
    up2 = Conv_block(filters=256)(tf.keras.layers.Add()([up2, conv3]))
    up2 = tf.keras.layers.Conv2D(filters=512, kernel_size=3, strides=1, padding='same')(up2)

    up3 = IWT_upsampling()(up2)
    up3 = Conv_block(filters=128)(tf.keras.layers.Add()([up3, conv2]))
    up3 = tf.keras.layers.Conv2D(filters=256, kernel_size=3, strides=1, padding='same')(up3)

    up4 = IWT_upsampling()(up3)
    up4 = Conv_block(filters=64)(tf.keras.layers.Add()([up4, conv1]))
    up4 = tf.keras.layers.Conv2D(filters=128, kernel_size=3, strides=1, padding='same')(up4)

    out = tf.keras.layers.Conv2D(filters=3, kernel_size=(1,1), padding='same')(up4)

    return tf.keras.Model(inputs=[input], outputs=[out])

if __name__=='__main__':
    model = MWCNN_model()
    model.summary()