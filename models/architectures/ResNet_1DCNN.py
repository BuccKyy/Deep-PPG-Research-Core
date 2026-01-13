# ResNet models for Keras.
# Reference for ResNets - [Deep Residual Learning for Image Recognition](https://arxiv.org/pdf/1512.03385.pdf))

import tensorflow as tf


def Conv_1D_Block(x, model_width, kernel, strides):
    # 1D Convolutional Block with BatchNormalization
    x = tf.keras.layers.Conv1D(model_width, kernel, strides=strides, padding="same", kernel_initializer="he_normal")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation('relu')(x)

    return x


def stem(inputs, num_filters):
    # Construct the Stem Convolution Group
    # inputs : input vector
    # First Convolutional layer, where pooled feature maps will be reduced by 75%
    conv = Conv_1D_Block(inputs, num_filters, 7, 2)
    if conv.shape[1] <= 2:
        pool = tf.keras.layers.MaxPooling1D(pool_size=1, strides=2, padding="valid")(conv)
    else:
        pool = tf.keras.layers.MaxPooling1D(pool_size=2, strides=2, padding="valid")(conv)

    return pool


def conv_block(inputs, num_filters):
    # Construct Block of Convolutions without Pooling
    # x        : input into the block
    # n_filters: number of filters
    conv = Conv_1D_Block(inputs, num_filters, 3, 2)
    conv = Conv_1D_Block(conv, num_filters, 3, 1)

    return conv


def residual_block(inputs, num_filters):
    # Construct a Residual Block of Convolutions
    # x        : input into the block
    # n_filters: number of filters
    shortcut = inputs
    #
    conv = Conv_1D_Block(inputs, num_filters, 3, 1)
    conv = Conv_1D_Block(conv, num_filters, 3, 1)
    conv = tf.keras.layers.Add()([conv, shortcut])
    out = tf.keras.layers.Activation('relu')(conv)

    return out


def residual_group(inputs, num_filters, n_blocks, conv=True):
    # x        : input to the group
    # n_filters: number of filters
    # n_blocks : number of blocks in the group
    # conv     : flag to include the convolution block connector
    out = inputs
    for i in range(n_blocks):
        out = residual_block(out, num_filters)

    # Double the size of filters and reduce feature maps by 75% (strides=2, 2) to fit the next Residual Group
    if conv:
        out = conv_block(out, num_filters * 2)

    return out


def stem_bottleneck(inputs, num_filters):
    # Construct the Stem Convolution Group
    # inputs :input vector
    # First Convolutional layer, where pooled feature maps will be reduced by 75%
    conv = Conv_1D_Block(inputs, num_filters, 7, 2)
    if conv.shape[1] <= 2:
        pool = tf.keras.layers.MaxPooling1D(pool_size=1, strides=2, padding="valid")(conv)
    else:
        pool = tf.keras.layers.MaxPooling1D(pool_size=2, strides=2, padding="valid")(conv)

    return pool


def residual_block_bottleneck(inputs, num_filters):
    # Construct a Residual Block of Convolutions
    # x        : input into the block
    # n_filters: number of filters
    shortcut = Conv_1D_Block(inputs, num_filters * 4, 1, 1)
    #
    conv = Conv_1D_Block(inputs, num_filters, 1, 1)
    conv = Conv_1D_Block(conv, num_filters, 3, 1)
    conv = Conv_1D_Block(conv, num_filters * 4, 1, 1)
    conv = tf.keras.layers.Add()([conv, shortcut])
    out = tf.keras.layers.Activation('relu')(conv)

    return out


def residual_group_bottleneck(inputs, num_filters, n_blocks, conv=True):
    # x        : input to the group
    # n_filters: number of filters
    # n_blocks : number of blocks in the group
    # conv     : flag to include the convolution block connector
    out = inputs
    for i in range(n_blocks):
        out = residual_block_bottleneck(out, num_filters)

    # Double the size of filters and reduce feature maps by 75% (strides=2, 2) to fit the next Residual Group
    if conv:
        out = conv_block(out, num_filters * 2)

    return out


def learner18(inputs, num_filters):
    # Construct the Learner
    x = residual_group(inputs, num_filters, 2)  # First Residual Block Group of 64 filters
    x = residual_group(x, num_filters * 2, 1)  # Second Residual Block Group of 128 filters
    x = residual_group(x, num_filters * 4, 1)  # Third Residual Block Group of 256 filters
    out = residual_group(x, num_filters * 8, 1, False)  # Fourth Residual Block Group of 512 filters

    return out


def learner34(inputs, num_filters):
    # Construct the Learner
    x = residual_group(inputs, num_filters, 3)  # First Residual Block Group of 64 filters
    x = residual_group(x, num_filters * 2, 3)  # Second Residual Block Group of 128 filters
    x = residual_group(x, num_filters * 4, 5)  # Third Residual Block Group of 256 filters
    out = residual_group(x, num_filters * 8, 2, False)  # Fourth Residual Block Group of 512 filters

    return out


def learner50(inputs, num_filters):
    # Construct the Learner
    x = residual_group_bottleneck(inputs, num_filters, 3)  # First Residual Block Group of 64 filters
    x = residual_group_bottleneck(x, num_filters * 2, 3)  # Second Residual Block Group of 128 filters
    x = residual_group_bottleneck(x, num_filters * 4, 5)  # Third Residual Block Group of 256 filters
    out = residual_group_bottleneck(x, num_filters * 8, 2, False)  # Fourth Residual Block Group of 512 filters

    return out


def learner101(inputs, num_filters):
    # Construct the Learner
    x = residual_group_bottleneck(inputs, num_filters, 3)  # First Residual Block Group of 64 filters
    x = residual_group_bottleneck(x, num_filters * 2, 3)  # Second Residual Block Group of 128 filters
    x = residual_group_bottleneck(x, num_filters * 4, 22)  # Third Residual Block Group of 256 filters
    out = residual_group_bottleneck(x, num_filters * 8, 2, False)  # Fourth Residual Block Group of 512 filters

    return out


def learner152(inputs, num_filters):
    # Construct the Learner
    x = residual_group_bottleneck(inputs, num_filters, 3)  # First Residual Block Group of 64 filters
    x = residual_group_bottleneck(x, num_filters * 2, 7)  # Second Residual Block Group of 128 filters
    x = residual_group_bottleneck(x, num_filters * 4, 35)  # Third Residual Block Group of 256 filters
    out = residual_group_bottleneck(x, num_filters * 8, 2, False)  # Fourth Residual Block Group of 512 filters

    return out


def classifier(inputs, class_number):
    # Construct the Classifier Group
    # inputs       : input vector
    # class_number : number of output classes
    out = tf.keras.layers.Dense(class_number, activation='softmax')(inputs)

    return out


def regressor(inputs, feature_number):
    # Construct the Regressor Group
    # inputs       : input vector
    # feature_number : number of output features
    out = tf.keras.layers.Dense(feature_number, activation='linear')(inputs)

    return out


def LSTM(inputs, layer, dense):
    out = tf.keras.layers.LSTM(layer, return_sequences=True, activation='tanh')(inputs)
    out = tf.keras.layers.Dense(dense, 'relu')(out)
    return out


class ResNet:
    def __init__(self, length, num_channel, num_filters, problem_type='Regression',
                 output_nums=1, pooling='avg', dropout_rate=False):
        self.length = length
        self.num_channel = num_channel
        self.num_filters = num_filters
        self.problem_type = problem_type
        self.output_nums = output_nums  ### chinh xiu
        self.pooling = pooling
        self.dropout_rate = dropout_rate

    def MLP(self, x):
        if self.pooling == 'avg':
            x = tf.keras.layers.GlobalAveragePooling1D()(x)
        elif self.pooling == 'max':
            x = tf.keras.layers.GlobalMaxPooling1D()(x)
        # Final Dense Outputting Layer for the outputs
        x = tf.keras.layers.Flatten()(x)
        if self.dropout_rate:
            x = tf.keras.layers.Dropout(self.dropout_rate)(x)
        outputs = tf.keras.layers.Dense(self.output_nums, activation='linear')(x)
        if self.problem_type == 'Classification':
            outputs = tf.keras.layers.Dense(self.output_nums, activation='softmax')(x)

        return outputs

    def MLP_2(self, x):
        if self.pooling == 'avg':
            x = tf.keras.layers.GlobalAveragePooling1D()(x)
        elif self.pooling == 'max':
            x = tf.keras.layers.GlobalMaxPooling1D()(x)
        # Final Dense Outputting Layer for the outputs
        x = tf.keras.layers.Flatten()(x)
        if self.dropout_rate:
            x = tf.keras.layers.Dropout(self.dropout_rate)(x)
        outputs = tf.keras.layers.Dense(self.output_nums, activation='linear')(x)
        if self.problem_type == 'Classification':
            outputs = tf.keras.layers.Dense(self.output_nums, activation='softmax')(x)

        return outputs

    def MLP_Resnet_LSTM(self, x):
        if self.pooling == 'avg':
            x = tf.keras.layers.GlobalAveragePooling1D()(x)
        elif self.pooling == 'max':
            x = tf.keras.layers.GlobalMaxPooling1D()(x)
        return x

    def ResNet18(self, **kwargs):
        inputs = tf.keras.Input((self.length, self.num_channel))  # The input tensor
        stem_ = stem(inputs, self.num_filters)  # The Stem Convolution Group
        x = learner18(stem_, self.num_filters)  # The learner
        outputs = self.MLP(x)
        # Instantiate the Model
        model = tf.keras.Model(inputs, outputs, name=kwargs.get('name', 'ResNet18'))

        return model

    def ResNet_18_LSTM(self, lstm_layer=1, near_final_output=128):
        ### Resnet
        inputs = tf.keras.Input((self.length, self.num_channel))  # The input tensor
        stem_ = stem(inputs, self.num_filters)  # The Stem Convolution Group
        x = learner18(stem_, self.num_filters)
        # The learner
        outputs_1 = self.MLP_Resnet_LSTM(x)

        ### LSTM model
        outputs_2 = LSTM(inputs, lstm_layer, dense=128)
        outputs_2 = self.MLP_Resnet_LSTM(outputs_2)
        ### adding model
        final_output = tf.keras.layers.add([outputs_1, outputs_2])
        # create dense
        outputs = tf.keras.layers.Dense(near_final_output, activation='relu')(final_output)
        outputs = tf.keras.layers.Dense(near_final_output // 2, activation='relu')(outputs)
        final_output = tf.keras.layers.Dense(self.output_nums, activation='linear')(outputs)
        # Instantiate the Model
        model = tf.keras.Model(inputs, final_output)
        return model

    def ResNet_18_calibrate(self, lstm_layer=1, near_final_output=128, **kwargs):
        ### Resnet
        inputs_1 = tf.keras.Input((self.length, self.num_channel))  # The input tensor
        stem_ = stem(inputs_1, self.num_filters)  # The Stem Convolution Group
        x = learner18(stem_, self.num_filters)
        # The learner
        outputs_1 = self.MLP_Resnet_LSTM(x)

        final_output = outputs_1
        # create dense
        near_output = tf.keras.layers.Dense(near_final_output, activation='relu')(final_output)

        ### ann model
        inputs_2 = tf.keras.Input((2,))  ######## sua
        layer_1 = tf.keras.layers.Dense(64, activation='relu')(inputs_2)
        layer_2 = tf.keras.layers.Dense(128, activation='relu')(layer_1)
        # layer_2 = tf.keras.layers.GlobalMaxPooling1D()(layer_2)
        #### add 2
        outputs = tf.keras.layers.concatenate([near_output, layer_2])

        ## final dense

        outputs = tf.keras.layers.Dense(128, activation='relu')(outputs)
        outputs = tf.keras.layers.Dense(64, activation='relu')(outputs)
        outputs = tf.keras.layers.Dense(16, activation='relu')(outputs)

        final_output = tf.keras.layers.Dense(self.output_nums, activation='linear')(outputs)
        # Instantiate the Model
        model = tf.keras.Model([inputs_1, inputs_2], final_output, name=kwargs.get('name', 'ResNet18'))
        return model

    def ResNet_18_LSTM_calibrate(self, lstm_layer=1, near_final_output=128):
        ### Resnet
        inputs_1 = tf.keras.Input((self.length, self.num_channel))  # The input tensor
        stem_ = stem(inputs_1, self.num_filters)  # The Stem Convolution Group
        x = learner18(stem_, self.num_filters)
        # The learner
        outputs_1 = self.MLP_Resnet_LSTM(x)

        ### LSTM model
        outputs_2 = LSTM(inputs_1, lstm_layer, dense=128)
        outputs_2 = self.MLP_Resnet_LSTM(outputs_2)
        ### adding model
        final_output = tf.keras.layers.add([outputs_1, outputs_2])
        # create dense
        near_output = tf.keras.layers.Dense(near_final_output, activation='relu')(final_output)

        ### ann model
        inputs_2 = tf.keras.Input((2,))  ######## sua
        layer_1 = tf.keras.layers.Dense(64, activation='relu')(inputs_2)
        layer_2 = tf.keras.layers.Dense(128, activation='relu')(layer_1)
        # layer_2 = tf.keras.layers.GlobalMaxPooling1D()(layer_2)
        #### add 2
        outputs = tf.keras.layers.concatenate([near_output, layer_2])

        ## final dense

        outputs = tf.keras.layers.Dense(128, activation='relu')(outputs)

        final_output = tf.keras.layers.Dense(self.output_nums, activation='linear')(outputs)
        # Instantiate the Model
        model = tf.keras.Model([inputs_1, inputs_2], final_output)
        return model

    def ResNet34(self):
        inputs = tf.keras.Input((self.length, self.num_channel))  # The input tensor
        stem_ = stem(inputs, self.num_filters)  # The Stem Convolution Group
        x = learner34(stem_, self.num_filters)  # The learner
        outputs = self.MLP(x)
        # Instantiate the Model
        model = tf.keras.Model(inputs, outputs)

        return model

    def ResNet18_2_output(self):

        inputs = tf.keras.Input((self.length, self.num_channel))  # The input tensor
        stem_ = stem(inputs, self.num_filters)  # The Stem Convolution Group
        x = learner18(stem_, self.num_filters)  # The learner
        outputs_1 = self.MLP(x)
        outputs_2 = self.MLP_2(x)
        # Instantiate the Model
        model = tf.keras.Model(inputs, outputs=[outputs_1, outputs_2])

        return model

    def ResNet50(self):
        inputs = tf.keras.Input((self.length, self.num_channel))  # The input tensor
        stem_b = stem_bottleneck(inputs, self.num_filters)  # The Stem Convolution Group
        x = learner50(stem_b, self.num_filters)  # The learner
        outputs = self.MLP(x)
        # Instantiate the Model
        model = tf.keras.Model(inputs, outputs)

        return model

    def ResNet101(self):
        inputs = tf.keras.Input((self.length, self.num_channel))  # The input tensor
        stem_b = stem_bottleneck(inputs, self.num_filters)  # The Stem Convolution Group
        x = learner101(stem_b, self.num_filters)  # The learner
        outputs = self.MLP(x)
        # Instantiate the Model
        model = tf.keras.Model(inputs, outputs)

        return model

    def ResNet152(self):
        inputs = tf.keras.Input((self.length, self.num_channel))  # The input tensor
        stem_b = stem_bottleneck(inputs, self.num_filters)  # The Stem Convolution Group
        x = learner152(stem_b, self.num_filters)  # The learner
        outputs = self.MLP(x)
        # Instantiate the Model
        model = tf.keras.Model(inputs, outputs)

        return model


def build_resnet():
    # Configurations
    length = 128*2  # Length of each Segment
    model_name = 'ResNet18_LSTM'  # DenseNet Models
    model_width = 32  # Width of the Initial Layer, subsequent layers start from here
    num_channel = 3  # Number of Input Channels in the Model
    problem_type = 'Regression'  # Classification or Regression
    output_nums = 32  # Number of Class for Classification Problems, always '1' for Regression Problems
    ########
    opt = tf.keras.optimizers.Adam(lr=1e-3, decay=1e-5)
    ###
    Model = ResNet(length, num_channel, model_width, problem_type=problem_type, output_nums=output_nums, pooling='avg',
                   dropout_rate=0.1)
    Model = Model.ResNet18()
    # Model = Model.ResNet_18_LSTM(lstm_layer=32)
    # Model = Model.ResNet_18_calibrate()
    # Model = Model.ResNet_18_LSTM_calibrate(lstm_layer=8)
    # Model.compile(opt, loss='', metrics=['accuracy'])
    # Model.summary()

    in_1 = tf.keras.Input((length, num_channel), name='ppg_template')
    x = Model(in_1)
    x = tf.keras.layers.ReLU()(x)
    x = tf.keras.layers.Dense(64, activation='relu')(x)
    x = tf.keras.layers.Dense(32, activation='relu')(x)
    x = tf.keras.layers.Dense(4, activation='relu')(x)
    x = tf.keras.layers.Dense(1, activation='linear')(x)

    final = tf.keras.Model(in_1, x)

    return final


# build_resnet()
