import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, BatchNormalization, ReLU, MaxPooling2D, GlobalAveragePooling2D, Dense
from tensorflow.keras.models import Model

def conv_block(x, filters, kernel_size, strides=(1, 1), padding="same"):
    x = Conv2D(filters=filters, kernel_size=kernel_size, strides=strides, padding=padding)(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    return x

def identity_block(x, filters, kernel_size, strides=(1, 1)):
    shortcut = x
    x = conv_block(x, filters, kernel_size, strides)
    x = conv_block(x, filters, kernel_size, strides)
    x = conv_block(x, filters, kernel_size, strides)
    x = tf.keras.layers.add([x, shortcut])
    x = ReLU()(x)
    return x

def resnet18(input_shape, num_classes):
    inputs = Input(shape=input_shape)

    # Initial Convolution
    x = conv_block(inputs, filters=64, kernel_size=(7, 7), strides=(2, 2))
    x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding="same")(x)

    # Residual Blocks
    x = identity_block(x, filters=64, kernel_size=(3, 3))
    x = identity_block(x, filters=64, kernel_size=(3, 3))

    x = identity_block(x, filters=128, kernel_size=(3, 3), strides=(2, 2))
    x = identity_block(x, filters=128, kernel_size=(3, 3))

    x = identity_block(x, filters=256, kernel_size=(3, 3), strides=(2, 2))
    x = identity_block(x, filters=256, kernel_size=(3, 3))

    x = identity_block(x, filters=512, kernel_size=(3, 3), strides=(2, 2))
    x = identity_block(x, filters=512, kernel_size=(3, 3))

    # Global Average Pooling and Fully Connected Layer
    x = GlobalAveragePooling2D()(x)
    x = Dense(units=num_classes, activation="softmax")(x)

    model = Model(inputs, x, name="resnet18")
    return model

# Define input shape and number of classes
input_shape = (224, 224, 3)
num_classes = 1000

# Create ResNet-18 model
model = resnet18(input_shape, num_classes)

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Print model summary
model.summary()
