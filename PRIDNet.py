import tensorflow as tf
from helpers import *

class Convolution_block(tf.keras.layers.Layer):
    def __init__(self, filters=64, kernel_size=(3,3), **kwargs):
        super().__init__(*kwargs)
        self.filters = filters
        self.kernel_size = kernel_size

        self.conv1 = tf.keras.layers.Conv2D(filters=self.filters, kernel_size=self.kernel_size, strides=1, padding='same')
        self.conv2 = tf.keras.layers.Conv2D(filters=self.filters, kernel_size=self.kernel_size, strides=1, padding='same')
        self.conv3 = tf.keras.layers.Conv2D(filters=self.filters, kernel_size=self.kernel_size, strides=1, padding='same')
        self.conv4 = tf.keras.layers.Conv2D(filters=self.filters, kernel_size=self.kernel_size, strides=1, padding='same')

    def call(self, x):
        x = self.conv1(x)
        x = tf.nn.relu(x)
        x = self.conv2(x)
        x = tf.nn.relu(x)
        x = self.conv3(x)
        x = tf.nn.relu(x)
        x = self.conv4(x)
        x = tf.nn.relu(x)
        return x
    
class Channel_attention(tf.keras.layers.Layer):
    def __init__(self, C=64, **kwargs):
        super().__init__(*kwargs)
        self.C = C
        self.gap = tf.keras.layers.GlobalAveragePooling2D()
        self.dense_middle = tf.keras.layers.Dense(units=2, activation='relu')
        self.dense_end = tf.keras.layers.Dense(units=C, activation='sigmoid')

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'C': self.C,
        })
        return config
    
    def call(self, x):
        v = self.gap(x)
        fc1 = self.dense_middle(v)
        mu = self.dense_end(fc1)
        U_out = tf.multiply(x, mu)
        return U_out

class Avg_pool_Unet_Upsample_msfe(tf.keras.layers.Layer):
    def __init__(self, avg_pool_size, upsampling_rate, **kwargs):
        super().__init__(**kwargs)

        # Avg pool init
        self.avg_pool = tf.keras.layers.AveragePooling2D(pool_size=avg_pool_size, padding='same')

        # Unet init
        self.deconv = []
        filter = 512

        for _ in range(4):
            self.deconv.append(tf.keras.layers.Conv2DTranspose(filters=filter/2, kernel_size=(3,3), strides=2, padding='same'))
            filter /= 2

        self.conv_32_down = []
        for _ in range(4):
            self.conv_32_down.append(tf.keras.layers.Conv2D(filters=64, kernel_size=(3,3), activation='relu', padding='same', kernel_regularizer=tf.keras.regularizers.l2(0.001)))

        self.conv_64_down = []
        for _ in range(4):
            self.conv_64_down.append(tf.keras.layers.Conv2D(filters=128, kernel_size=(3,3), activation='relu', padding='same', kernel_regularizer=tf.keras.regularizers.l2(0.001)))

        self.conv_128_down = []
        for _ in range(4):
            self.conv_128_down.append(tf.keras.layers.Conv2D(filters=256, kernel_size=(3,3), activation='relu', padding='same', kernel_regularizer=tf.keras.regularizers.l2(0.001)))

        self.conv_256_down = []
        for _ in range(4):
            self.conv_256_down.append(tf.keras.layers.Conv2D(filters=512, kernel_size=(3,3), activation='relu', padding='same', kernel_regularizer=tf.keras.regularizers.l2(0.001)))

        self.conv_512_down = []
        for _ in range(4):
            self.conv_512_down.append(tf.keras.layers.Conv2D(filters=1024, kernel_size=(3,3), activation='relu', padding='same', kernel_regularizer=tf.keras.regularizers.l2(0.001)))

        self.conv_32_up = []
        for _ in range(3):
            self.conv_32_up.append(tf.keras.layers.Conv2D(filters=64, kernel_size=(3,3), activation='relu', padding='same', kernel_regularizer=tf.keras.regularizers.l2(0.001)))

        self.conv_64_up = []
        for _ in range(3):
            self.conv_64_up.append(tf.keras.layers.Conv2D(filters=128, kernel_size=(3,3), activation='relu', padding='same', kernel_regularizer=tf.keras.regularizers.l2(0.001)))

        self.conv_128_up = []
        for _ in range(3):
            self.conv_128_up.append(tf.keras.layers.Conv2D(filters=256, kernel_size=(3,3), activation='relu', padding='same', kernel_regularizer=tf.keras.regularizers.l2(0.001)))

        self.conv_256_up = []
        for _ in range(3):
            self.conv_256_up.append(tf.keras.layers.Conv2D(filters=512, kernel_size=(3,3), activation='relu', padding='same', kernel_regularizer=tf.keras.regularizers.l2(0.001)))

        self.conv3 = tf.keras.layers.Conv2D(filters=3, kernel_size=(1,1))

        self.unet_pool2 = tf.keras.layers.MaxPooling2D(pool_size=(2,2), padding='same')
        self.unet_pool1 = tf.keras.layers.MaxPooling2D(pool_size=(2,2), padding='same')
        self.unet_pool3 = tf.keras.layers.MaxPooling2D(pool_size=(2,2), padding='same')
        self.unet_pool4 = tf.keras.layers.MaxPooling2D(pool_size=(2,2), padding='same')

        # Upsampling init
        self.upsampling = tf.keras.layers.UpSampling2D(size=upsampling_rate, interpolation='bilinear')

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'avg_pool_size': self.avg_pool_size,
            'upsampling_rate': self.upsampling_rate,
        })
        return config
    
    def upsample_concat(self, x1, x2, i):
        deconv = self.deconv[i](x1)
        return tf.concat([deconv, x2], axis=-1)
    
    def unet(self,input):
        # unet downsample
        conv1 = input
        for c_32 in self.conv_32_down:
            conv1 = c_32(conv1)
        pool1 = self.unet_pool1(conv1)

        conv2 = pool1
        for c_64 in self.conv_64_down:
            conv2 = c_64(conv2)
        pool2 = self.unet_pool2(conv2)

        conv3 = pool2
        for c_128 in self.conv_128_down:
            conv3 = c_128(conv3)
        pool3 = self.unet_pool3(conv3)

        conv4 = pool3
        for c_256 in self.conv_256_down:
            conv4 = c_256(conv4)
        pool4 = self.unet_pool4(conv4)

        conv5 = pool4
        for c_512 in self.conv_512_down:
            conv5 = c_512(conv5)
        
        # unet upsample
        up6 = self.upsample_concat(conv5, conv4, 0)
        conv6 = up6
        for c_256 in self.conv_256_up:
            conv6 = c_256(conv6)

        up7 = self.upsample_concat(conv6, conv3, 1)
        conv7 = up7
        for c_128 in self.conv_128_up:
            conv7 = c_128(conv7)

        up8 = self.upsample_concat(conv7, conv2, 2)
        conv8 = up8
        for c_64 in self.conv_64_up:
            conv8 = c_64(conv8)

        up9 = self.upsample_concat(conv8, conv1, 3)
        conv9 = up9
        for c_32 in self.conv_32_up:
            conv9 = c_32(conv9)

        conv10 = self.conv3(conv9)
        return conv10
    
    def call(self, input):
        avg_pool = self.avg_pool(input)
        unet = self.unet(avg_pool)
        upsample = self.upsampling(unet)
        return upsample
    
class Multi_scale_feature_extraction(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.msfe_16 = Avg_pool_Unet_Upsample_msfe(avg_pool_size=16, upsampling_rate=16)
        self.msfe_8 = Avg_pool_Unet_Upsample_msfe(avg_pool_size=8, upsampling_rate=8)
        self.msfe_4 = Avg_pool_Unet_Upsample_msfe(avg_pool_size=4, upsampling_rate=4)
        self.msfe_2 = Avg_pool_Unet_Upsample_msfe(avg_pool_size=2, upsampling_rate=2)
        self.msfe_1 = Avg_pool_Unet_Upsample_msfe(avg_pool_size=1, upsampling_rate=1)

    def call(self, x):
        up_sample_16 = self.msfe_16(x)
        up_sample_8 = self.msfe_8(x)
        up_sample_4 = self.msfe_4(x)
        up_sample_2 = self.msfe_2(x)
        up_sample_1 = self.msfe_1(x)
        return tf.concat([up_sample_16, up_sample_8, up_sample_4, up_sample_2, up_sample_1], axis=-1)
    
class Kernel_selecting_module(tf.keras.layers.Layer):
    def __init__(self, C=21, **kwargs):
        super().__init__(**kwargs)
        self.C = C
        self.c_3 = tf.keras.layers.Conv2D(filters=self.C, kernel_size=(3,3), strides=1, padding='same', kernel_regularizer=tf.keras.regularizers.l2(0.001)) 
        self.c_5 = tf.keras.layers.Conv2D(filters=self.C, kernel_size=(5,5), strides=1, padding='same', kernel_regularizer=tf.keras.regularizers.l2(0.001))
        self.c_7 = tf.keras.layers.Conv2D(filters=self.C, kernel_size=(7,7), strides=1, padding='same', kernel_regularizer=tf.keras.regularizers.l2(0.001))
        self.gap = tf.keras.layers.GlobalAveragePooling2D()
        self.dense_2 = tf.keras.layers.Dense(units=2, activation='relu')
        self.dense_c1 = tf.keras.layers.Dense(units=self.C)
        self.dense_c2 = tf.keras.layers.Dense(units=self.C)
        self.dense_c3 = tf.keras.layers.Dense(units=self.C)

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'C': self.C,
        })
        return config
    
    def call(self, x):
        x1 = self.c_3(x)
        x2 = self.c_5(x)
        x3 = self.c_7(x)
        x_dash = tf.keras.layers.Add()([x1, x2, x3])

        v_gap = self.gap(x_dash)
        v_gap = tf.reshape(v_gap, [-1, 1, 1, self.C])
        fc1 = self.dense_2(v_gap)

        alpha = self.dense_c1(fc1)
        beta = self.dense_c2(fc1)
        gamma = self.dense_c3(fc1)

        before = tf.concat([alpha, beta, gamma], axis=1)
        after = tf.keras.activations.softmax(before, axis=1)

        a1 = after[:, 0, :, :]
        a1 = tf.reshape(a1, [-1, 1, 1, self.C])
        a2 = after[:, 1, :, :]
        a2 = tf.reshape(a2, [-1, 1, 1, self.C])
        a3 = after[:, 2, :, :]
        a3 = tf.reshape(a3, [-1, 1, 1, self.C])

        y1 = tf.multiply(x1, a1)
        y2 = tf.multiply(x2, a2)
        y3 = tf.multiply(x3, a3)

        out = tf.keras.layers.Add()([y1, y2, y3])
        return out
    
def PRIDNet_model():
    input = tf.keras.layers.Input(shape=(None, None, 3))

    conv = Convolution_block()(input)
    
    ca = Channel_attention()(conv)
    ca = tf.keras.layers.Conv2D(filters=3, kernel_size=(3,3), strides=1, padding='same')(ca)
    ca = tf.concat([ca, input], axis=-1)
    
    msfe = Multi_scale_feature_extraction()(ca)

    ksm = Kernel_selecting_module()(msfe)
    ksm = tf.keras.layers.Conv2D(filters=3, kernel_size=(3,3), strides=1, padding='same')(ksm)

    model = tf.keras.Model(inputs=input, outputs=ksm)
    return model

if __name__=='__main__':
    model = PRIDNet_model()
    model.summary()