import tensorflow as tf

class Model():

    @staticmethod
    def conv2d_net(x,
                   num_filters,
                   kernel_size,
                   strides=1,
                   pad='SAME',
                   act=True,
                   bn=True,
                   rate=0.5,
                   name=""):
        """ """
        if bn:
            x = tf.keras.layers.BatchNormalization(axis=-1, name=name + '_bn')(x)
        if act:
            x = tf.keras.layers.ReLU(name=name + '_act')(x)
        if rate < 1.0:
            x = tf.keras.layers.Dropout(rate=rate,
                                        name=name + '_drop')(x)

        x = tf.keras.layers.Conv2D(filters=num_filters,
                                   kernel_size=(1, kernel_size),
                                   strides=(1, strides),
                                   padding=pad,
                                   name=name + '_conv2d')(x)
        return x

    def block2d_loop(self,
                     xx,
                     ff,
                     kernel_size,
                     stage,
                     step,
                     bn):
        """ :param xx: :param ff: :param stage: :param step: :return: """
        xx_skip = xx
        f1, f2 = ff
        # Batch norm, Activation, Dropout, Convolution (stride=1)
        xx = self.conv2d_net(x=xx,
                             num_filters=f1,
                             kernel_size=kernel_size,
                             strides=1,
                             pad='SAME',
                             act=True,
                             bn=bn,
                             rate=0.5,
                             name="resnet11a_{}_{}".format(step, stage))
        # Batch norm, Activation, Dropout, Convolution (stride=1)
        xx = self.conv2d_net(x=xx,
                             num_filters=f2,
                             kernel_size=kernel_size,
                             strides=1,
                             pad='SAME',
                             act=True,
                             bn=bn,
                             rate=0.5,
                             name="resnet11b_{}_{}".format(step, stage))
        xx = tf.keras.layers.Add(name="skip11_{}_{}".format(step, stage))([xx, xx_skip])
        return xx

    def beat_lstm_tiny(self,
                        feature_len, num_of_class=2,
                        batch_normalization=True,
                        filters_rhythm_net=None,
                        kernel_size=3,
                        num_loop=7,
                        rate=0.5,
                        is_unstack=False,
                        name_input='beat_lstm_tiny'):
        """ """
        if filters_rhythm_net is None:
            filters_rhythm_net = [(8, 8), (8, 16), (16, 32), (32, 48)]
        else:
            tmp = []
            for i, f in enumerate(filters_rhythm_net):
                tmp.append((filters_rhythm_net[0], f))
                filters_rhythm_net = tmp.copy()

        input_layer = tf.keras.layers.Input(shape=(feature_len,), name="input_raw")
        resnet_input_layer = tf.keras.layers.Reshape((1, feature_len, 1), name="input_reshape")(input_layer)
        # Convolution(stride=2)
        x = self.conv2d_net(x=resnet_input_layer,
                            num_filters=filters_rhythm_net[0][0],
                            kernel_size=kernel_size,
                            strides=2, pad='SAME',
                            act=False,
                            bn=False,
                            rate=1.0,
                            name="input_stage")
        for st, ff in enumerate(filters_rhythm_net):
            st += 1
            f1, f2 = ff
            name = 'stage_{}'.format(st)
            # 1x1 Convolution (stride=2)
            x_skip = self.conv2d_net(x=x,
                                     num_filters=f2,
                                     kernel_size=1,
                                     strides=2,
                                     pad='SAME',
                                     act=False,
                                     bn=False,
                                     rate=1.0,
                                     name="skip12_" + name)
            # Batch norm, Activation, Dropout, Convolution (stride=2)
            x = self.conv2d_net(x=x,
                                num_filters=f1,
                                kernel_size=kernel_size,
                                strides=2,
                                pad='SAME',
                                act=True,
                                bn=batch_normalization,
                                rate=rate,
                                name="resnet12" + name)
            # Batch norm, Activation, Dropout, Convolution (stride=1)
            x = self.conv2d_net(x=x, num_filters=f2, kernel_size=kernel_size, strides=1, pad='SAME', act=True, bn=batch_normalization, rate=rate, name="resnet11" + name)
            x = tf.keras.layers.Add(name="add_" + name)([x, x_skip])
            ffs = [(f2, f2) for _ in range(num_loop)]
            for sl, ffl in enumerate(ffs):
                x = self.block2d_loop(x, ffl, kernel_size, name, sl, batch_normalization)

        num = x.get_shape().as_list()[-2] * x.get_shape().as_list()[-3]
        fea = x.get_shape().as_list()[-1]
        x = tf.keras.layers.Reshape((num, fea), name="resnet_squeeze")(x)
        resnet_block = tf.unstack(x, num=num, axis=1, name="resnet_unstack")
        conv_outputs = []
        for xx in resnet_block:
            conv_outputs.append(tf.reshape((48, 1))(xx))

        output_layer = tf.keras.layers.LSTM(num_of_class, return_sequences=True, return_state=False)(tf.convert_to_tensor(conv_outputs))
        # if is_unstack:
        # resnet_block = tf.unstack(x, num=num, axis=1, name="resnet_unstack")
        # for i, st in enumerate(resnet_block):
        # resnet_block[i] = tf.keras.layers.Dense(num_of_class, activation='softmax', name="fc-{}".format(i))(st)
        #
        # output_layer = tf.stack(resnet_block, axis=1, name="resnet_stack")
        # else:
        # output_layer = tf.keras.layers.Dense(num_of_class, activation='softmax', name="fc")(x)

        model_ =tf.keras.Model(input_layer, output_layer, name=name_input)
        model_.summary()
        return tf.keras.Model(input_layer, output_layer, name=name_input)

md = Model()
md.beat_lstm_tiny(feature_len=256)