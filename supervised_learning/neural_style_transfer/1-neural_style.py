#!/usr/bin/env python3
import numpy as np
import tensorflow as tf


class NST:
    """Performs tasks for neural style transfer."""

    style_layers = [
        'block1_conv1',
        'block2_conv1',
        'block3_conv1',
        'block4_conv1',
        'block5_conv1'
    ]
    content_layer = 'block5_conv2'

    def __init__(self, style_image, content_image, alpha=1e4, beta=1):
        """Initialise the neural style transfer instance."""
        if not isinstance(style_image, np.ndarray) or \
                style_image.ndim != 3 or style_image.shape[2] != 3:
            raise TypeError(
                "style_image must be a numpy.ndarray with shape (h, w, 3)"
            )

        if not isinstance(content_image, np.ndarray) or \
                content_image.ndim != 3 or content_image.shape[2] != 3:
            raise TypeError(
                "content_image must be a numpy.ndarray with shape (h, w, 3)"
            )

        if not isinstance(alpha, (int, float)) or alpha < 0:
            raise TypeError("alpha must be a non-negative number")

        if not isinstance(beta, (int, float)) or beta < 0:
            raise TypeError("beta must be a non-negative number")

        if hasattr(tf, 'enable_eager_execution'):
            tf.enable_eager_execution()

        self.style_image = self.scale_image(style_image)
        self.content_image = self.scale_image(content_image)
        self.alpha = alpha
        self.beta = beta

        self.load_model()

    @staticmethod
    def scale_image(image):
        """  Rescale an image so its largest side is 512 pixels."""
        if not isinstance(image, np.ndarray) or \
                image.ndim != 3 or image.shape[2] != 3:
            raise TypeError(
                "image must be a numpy.ndarray with shape (h, w, 3)"
            )

        image = tf.convert_to_tensor(image, dtype=tf.float32)

        shape = tf.shape(image)
        height = tf.cast(shape[0], tf.float32)
        width = tf.cast(shape[1], tf.float32)

        scale = 512.0 / tf.maximum(height, width)

        new_height = tf.cast(tf.round(height * scale), tf.int32)
        new_width = tf.cast(tf.round(width * scale), tf.int32)

        image = tf.image.resize(
            image,
            [new_height, new_width],
            method=tf.image.ResizeMethod.BICUBIC
        )

        image = image / 255.0
        image = tf.clip_by_value(image, 0.0, 1.0)

        return tf.expand_dims(image, axis=0)

    def load_model(self):
    """Creates the model used to calculate the style transfer cost."""
    base = tf.keras.applications.VGG19(
        include_top=False,
        weights='imagenet'
    )

    inputs = tf.keras.Input(shape=(None, None, 3))
    x = inputs

    for layer in base.layers[1:]:
        if isinstance(layer, tf.keras.layers.MaxPooling2D):
            x = tf.keras.layers.AveragePooling2D(
                pool_size=layer.pool_size,
                strides=layer.strides,
                padding=layer.padding,
                name=layer.name
            )(x)
        else:
            x = layer(x)

    model = tf.keras.Model(inputs=inputs, outputs=x)

    for layer in model.layers:
        layer.trainable = False

    outputs = [
        model.get_layer(name).output
        for name in self.style_layers
    ]
    outputs.append(model.get_layer(self.content_layer).output)

    self.model = tf.keras.Model(
        inputs=model.input,
        outputs=outputs
    )
