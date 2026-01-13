import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


class AutoencoderModels:

    @staticmethod
    def auto_conv_tiny(
                       signal_length,
                       signal_width,
                       signal_height,
                       filters_net=None,
                       kernel_size=None,
                       latent_dim=4,
                       decode_activation='none',
                       name_input='auto_conv_tiny'):

        if filters_net is None:
            filters_net = [8, 8]

        input_sig = tf.keras.Input(shape=(signal_length,), name='encoder_input')
        input_img = tf.keras.layers.Reshape((signal_width, signal_height, 1), name="encoder_input_reshape")(input_sig)
        x = tf.keras.layers.Conv2D(filters_net[0],
                                (kernel_size[0], kernel_size[1]),
                                strides=(2, 2),
                                padding='same',
                                activation='relu',
                                name='encoder_input_stage')(input_img)

        for i, f in enumerate(filters_net[1:]):
            x = tf.keras.layers.Conv2D(f,
                                    (kernel_size[0], kernel_size[1]),
                                    strides=(2, 2),
                                    padding='same',
                                    activation='relu',
                                    name='encoder_stage_{}'.format(i + 1))(x)

        x = tf.keras.layers.Flatten(name='encoder_flatten')(x)
        z = tf.keras.layers.Dense(latent_dim, name='encoder_dense')(x)
        encoder = tf.keras.Model(inputs=input_sig, outputs=z, name='tiny_encoder')
        encoder.summary()

        latent_inputs = tf.keras.Input((latent_dim,), name='decoder_input')

        y = tf.keras.layers.Dense((int(signal_width / pow(2, filters_net.size)) *
                                int(signal_height / pow(2, filters_net.size)) *
                                filters_net[-1]),
                               activation='relu',
                               name='decoder_dense')(latent_inputs)

        y = tf.keras.layers.Reshape(target_shape=(int(signal_width / pow(2, filters_net.size)),
                                               int(signal_height / pow(2, filters_net.size)),
                                               filters_net[-1]), name='decoder_input_reshape')(y)

        for i, f in reversed(list(enumerate(filters_net[:-1]))):
            y = tf.keras.layers.Conv2DTranspose(f,
                                             (kernel_size[0], kernel_size[1]),
                                             strides=(2, 2),
                                             padding='same',
                                             activation='relu',
                                             name='decoder_stage_{}'.format(i + 1))(y)

        if decode_activation is None or len(decode_activation) == 0 or 'none' == decode_activation:
            y = tf.keras.layers.Conv2DTranspose(1,
                                             (kernel_size[0], kernel_size[1]),
                                             strides=(2, 2),
                                             padding='same',
                                             name='decoder_output_stage')(y)
        else:
            y = tf.keras.layers.Conv2DTranspose(1,
                                             (kernel_size[0], kernel_size[1]),
                                             strides=(2, 2),
                                             padding='same',
                                             activation='{}'.format(decode_activation),
                                             name='decoder_output_stage')(y)

        y = tf.keras.layers.Reshape((signal_length,), name='decoder_output_reshape')(y)

        decoder = tf.keras.Model(inputs=latent_inputs, outputs=y, name='tiny_decoder')
        decoder.summary()

        # Define AE model.
        embedding = encoder(input_sig)
        reconstruction = decoder(embedding)
        autoencoder = tf.keras.Model(inputs=input_sig, outputs=reconstruction, name=name_input)
        autoencoder.summary()
        return autoencoder



def build_vae(input_shape, latent_dim):
    # Encoder
    encoder_input = tf.keras.layers.Input(shape=input_shape)
    x = tf.keras.layers.Conv1D(64, 4, 2, activation='relu')(encoder_input)
    x = tf.keras.layers.Conv1D(128, 4, 2, activation='relu')(x)
    x = tf.keras.layers.Conv1D(64, 4, 2, activation='relu')(x)
    x = tf.keras.layers.Conv1D(8, 4, 2, activation='relu')(x)
    x = tf.keras.layers.Flatten()(x)
    encoder = tf.keras.models.Model(encoder_input, x, name='Encoder')

    latent_mean = tf.keras.layers.Dense(latent_dim)(x)
    latent_log_var = tf.keras.layers.Dense(latent_dim)(x)

    # Reparameterization trick
    def sampling(args):
        latent_mean, latent_log_var = args
        epsilon = tf.random.normal(shape=(tf.shape(latent_mean)[0], latent_dim))
        return latent_mean + tf.exp(0.5 * latent_log_var) * epsilon

    latent_vector = tf.keras.layers.Lambda(sampling)([latent_mean, latent_log_var])

    # Decoder
    decoder_input = tf.keras.layers.Input(shape=(496, 1))
    x = decoder_input
    # x = tf.keras.layers.Dense(496, activation='relu')(decoder_input)
    # x = tf.keras.layers.Reshape((496, 1))(x)
    x = tf.keras.layers.Reshape((62, 8))(x)
    # x = tf.keras.layers.Conv1DTranspose(8, 3, 3, activation='relu', padding='same')(x)
    x = tf.keras.layers.Conv1DTranspose(8, 4, 2, activation='relu')(x)
    x = tf.keras.layers.Conv1DTranspose(64, 4, 2, activation='relu')(x)
    x = tf.keras.layers.Conv1DTranspose(128, 4, 2, activation='relu')(x)
    x = tf.keras.layers.Conv1DTranspose(64, 4, 2, activation='relu')(x)
    decoder_output = tf.keras.layers.Conv1DTranspose(1, 3, 1, activation='sigmoid')(x)

    # Define the full VAE model
    decoder = tf.keras.models.Model(decoder_input, decoder_output, name='Decoder')

    encoder.summary()
    decoder.summary()

    vae_input = encoder_input
    x = encoder(vae_input)
    vae_output = decoder(x)
    # vae_output = decoder(decoder_output)
    vae = tf.keras.models.Model(vae_input, vae_output,  name='VAE')

    # Define custom loss function for VAE
    # def vae_loss(x, x_decoded_mean, latent_mean, latent_log_var):
    #     reconstruction_loss = tf.keras.losses.binary_crossentropy(x, x_decoded_mean) * input_shape[0] * input_shape[1]
    #     kl_loss = -0.5 * tf.reduce_sum(1 + latent_log_var - tf.square(latent_mean) - tf.exp(latent_log_var), axis=-1)
    #     return reconstruction_loss + kl_loss

    # def vae_loss(x, x_decoded_mean):
    #     reconstruction_loss = tf.keras.losses.binary_crossentropy(x, x_decoded_mean) * np.prod(input_shape)
    #     kl_loss = -0.5 * tf.reduce_sum(1 + latent_log_var - tf.square(latent_mean) - tf.exp(latent_log_var), axis=-1)
    #     return reconstruction_loss + kl_loss

    # vae.compile(optimizer='adam', loss=vae_loss)

    return vae


# Load and preprocess your data (e.g., MNIST dataset)
# For this example, let's assume you have a variable 'x_train' containing your training data
# You may need to reshape and normalize the data before using it

# input_shape = (1024, 1)
# latent_dim = 32
# model = build_vae(input_shape, latent_dim)
# model.summary()


