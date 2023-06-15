import tensorflow as tf

class Conv_block_Dn(tf.keras.layers.Layer):
    def __init__(self, kernel_size=(3,3), filters=64, strides=1, normalize=True, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.kernel_size = kernel_size
        self.filters = filters
        self.strides = strides
        self.normalize = normalize

        self.conv1 = tf.keras.layers.Conv2D(filters=self.filters, kernel_size=self.kernel_size, strides=self.strides, padding='same')
        if self.normalize:
            self.bn = tf.keras.layers.BatchNormalization()
        self.relu = tf.keras.layers.ReLU()

    def call(self, input):
        conv1 = self.conv1(input)
        if self.normalize:
            conv1 = self.bn(conv1)
        conv1 = self.relu(conv1)
        return conv1

class DnCNN(tf.keras.layers.Layer):
    def __init__(self, kernel_size=(3,3), filters=64, dncnn_layers=17, strides=(1,1), *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.kernel_size = kernel_size
        self.filters = filters
        self.dncnn_layers = dncnn_layers
        self.strides = strides

        self.conv1 = Conv_block_Dn(self.kernel_size, self.filters, normalize=False)
        self.convs = []
        for _ in range(2, dncnn_layers):
            self.convs.append(Conv_block_Dn(kernel_size, filters))
        self.conv_final = tf.keras.layers.Conv2D(filters=3, kernel_size=kernel_size, strides=strides, padding='same')

    def call(self, input):
        conv1 = self.conv1(input)
        for conv in self.convs:
            conv1 = conv(conv1)
        conv_final = self.conv_final(conv1)
        return conv_final


def DnCNN_model():
        input = tf.keras.layers.Input(shape=(None, None, 3))
        dncnn = DnCNN()(input)
        output = input - dncnn
        model = tf.keras.models.Model(inputs=input, outputs=output) 
        return model
        
if __name__=='__main__':
    model = DnCNN_model()
    model.summary()