import os
from keras import backend as K
from pathlib import Path

import tensorflow as tf


from keras import layers, models
from keras.applications import VGG19


input_shape = (224, 224, 3)


# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
def vgg19_feature_extractor(input_shape):
    # Create VGG16 base model
    base_model = VGG19(include_top=False, input_shape=input_shape, weights="imagenet")

    # Freeze the layers of the base model
    for layer in base_model.layers:
        layer.trainable = False

    # Get the output of the base model
    output = base_model.output

    # Flatten the output feature vectors
    output = layers.GlobalAveragePooling2D()(output)

    # Create the model
    model = models.Model(inputs=base_model.input, outputs=output)

    return model


def build_siamese_vgg16(input_shape):
    # Define the input layer for the first image
    input_a = layers.Input(shape=input_shape, name="input_a")
    # Define the input layer for the second image
    input_b = layers.Input(shape=input_shape, name="input_b")

    # data_augmentation = tf.keras.Sequential([
    #   layers.RandomFlip("horizontal_and_vertical"),
    #   layers.RandomRotation(0.2),
    # ], name = "data_augmentation")
    # augmented_input_a = data_augmentation(input_a)
    # augmented_input_b = data_augmentation(input_b)
    # Define the VGG19 model (excluding the top layers)
    base_model = vgg19_feature_extractor(input_shape)

    # Get the output feature vectors from the base model for both inputs
    output_a = base_model(input_a)
    output_b = base_model(input_b)

    # output_a = Dropout(0.2)(output_a)
    # output_b = Dropout(0.2)(output_b)
    # concatenated_features = layers.Concatenate()([output_a, output_b])

    # Distance calculation
    distance = layers.Lambda(lambda x: K.abs(x[0] - x[1]), name="distance")(
        [output_a, output_b]
    )

    # Output layer
    output = layers.Dense(1, activation="sigmoid", name="output")(distance)

    # Create the Siamese model
    siamese_model = models.Model(
        inputs=[input_a, input_b], outputs=output, name="siamese_vgg16"
    )

    return siamese_model


def load_model():
    # Set the input shape for the VGG19 model
    input_shape = (224, 224, 3)

    # Build the Siamese VGG19 twins model

    siamese_vgg16 = build_siamese_vgg16(input_shape)

    # num_epochs = 10
    learning_rate = 0.001

    # siamese_vgg16.compile(optimizer =tf.keras.optimizers.legacy.Adam(learning_rate=learning_rate),
    #                       loss = tf.keras.losses.BinaryCrossentropy(from_logits=True),
    #                       metrics = ["accuracy"])
    # Fit the model using generators
    siamese_vgg16.compile(
        optimizer=tf.keras.optimizers.legacy.Adam(learning_rate=learning_rate),
        loss=tf.keras.losses.BinaryCrossentropy(),
        metrics=["accuracy"],
    )

    BASE_DIR = Path(__file__).resolve().parent.parent
    path = os.path.join(BASE_DIR, "backend", "current.h5")
    siamese_vgg16.load_weights(path)

    return siamese_vgg16
