
#!/usr/bin/env python3
"""
Defines class NST that performs tasks for neural style transfer.
"""

import numpy as np
import tensorflow as tf


class NST:
    """
    Performs tasks for Neural Style Transfer.

    Class attributes:
        style_layers: VGG19 layers used to extract style features.
        content_layer: VGG19 layer used to extract content features.

    Instance attributes:
        style_image: Preprocessed style image.
        content_image: Preprocessed content image.
        alpha: Weight for content cost.
        beta: Weight for style cost.
        model: VGG19 feature extraction model.
        gram_style_features: Gram matrices of style features.
        content_feature: Content features.
    """

    style_layers = [
        'block1_conv1',
        'block2_conv1',
        'block3_conv1',
        'block4_conv1',
        'block5_conv1'
    ]

    content_layer = 'block5_conv2'

    def __init__(self, style_image, content_image,
                 alpha=1e4, beta=1):
        """
        Initialize the NST instance.

        Args:
            style_image: NumPy array of shape (h, w, 3).
            content_image: NumPy array of shape (h, w, 3).
            alpha: Non-negative weight for content cost.
            beta: Non-negative weight for style cost.

        Raises:
            TypeError: If an argument has an invalid type or shape.
        """
        if type(style_image) is not np.ndarray or \
                len(style_image.shape) != 3:
            raise TypeError(
                "style_image must be a numpy.ndarray with shape (h, w, 3)"
            )

        if type(content_image) is not np.ndarray or \
                len(content_image.shape) != 3:
            raise TypeError(
                "content_image must be a numpy.ndarray with shape (h, w, 3)"
            )

        style_h, style_w, style_c = style_image.shape
        content_h, content_w, content_c = content_image.shape

        if style_h <= 0 or style_w <= 0 or style_c != 3:
            raise TypeError(
                "style_image must be a numpy.ndarray with shape (h, w, 3)"
            )

        if content_h <= 0 or content_w <= 0 or content_c != 3:
            raise TypeError(
                "content_image must be a numpy.ndarray with shape (h, w, 3)"
            )

        if type(alpha) not in (float, int) or alpha < 0:
            raise TypeError("alpha must be a non-negative number")

        if type(beta) not in (float, int) or beta < 0:
            raise TypeError("beta must be a non-negative number")

        self.style_image = self.scale_image(style_image)
        self.content_image = self.scale_image(content_image)
        self.alpha = alpha
        self.beta = beta

        self.load_model()
        self.generate_features()

    @staticmethod
    def scale_image(image):
        """
        Rescale an image to [0, 1] with its largest side equal to 512.

        Args:
            image: NumPy array of shape (h, w, 3).

        Returns:
            Tensor of shape (1, h_new, w_new, 3).

        Raises:
            TypeError: If image is not a valid NumPy image array.
        """
        if type(image) is not np.ndarray or len(image.shape) != 3:
            raise TypeError(
                "image must be a numpy.ndarray with shape (h, w, 3)"
            )

        h, w, c = image.shape

        if h <= 0 or w <= 0 or c != 3:
            raise TypeError(
                "image must be a numpy.ndarray with shape (h, w, 3)"
            )

        if h > w:
            h_new = 512
            w_new = int(w * 512 / h)
        else:
            w_new = 512
            h_new = int(h * 512 / w)

        image_tensor = tf.convert_to_tensor(
            np.expand_dims(image, axis=0),
            dtype=tf.float32
        )

        resized = tf.image.resize(
            image_tensor,
            size=(h_new, w_new),
            method='bicubic'
        )

        rescaled = resized / 255.0
        rescaled = tf.clip_by_value(rescaled, 0.0, 1.0)

        return rescaled

    def load_model(self):
        """
        Load VGG19 and construct the feature extraction model.

        The model outputs the selected style layer outputs followed
        by the content layer output. All model layers are frozen.
        """
        vgg = tf.keras.applications.VGG19(
            include_top=False,
            weights='imagenet'
        )

        outputs = []

        for layer_name in self.style_layers:
            outputs.append(vgg.get_layer(layer_name).output)

        content_output = vgg.get_layer(self.content_layer).output
        outputs.append(content_output)

        self.model = tf.keras.models.Model(
            inputs=vgg.input,
            outputs=outputs
        )

        self.model.trainable = False

    @staticmethod
    def gram_matrix(input_layer):
        """
        Calculate the Gram matrix of a feature map.

        Args:
            input_layer: Tensor of rank 4 with shape (1, h, w, c).

        Returns:
            Gram matrix with shape (1, c, c).

        Raises:
            TypeError: If input_layer is not a rank-4 Tensor or Variable.
        """
        if not isinstance(
                input_layer, (tf.Tensor, tf.Variable)
        ) or len(input_layer.shape) != 4:
            raise TypeError("input_layer must be a tensor of rank 4")

        shape = tf.shape(input_layer)
        h = shape[1]
        w = shape[2]
        c = shape[3]

        features = tf.reshape(input_layer, (h * w, c))

        gram = tf.matmul(
            features,
            features,
            transpose_a=True
        )

        gram = tf.expand_dims(gram, axis=0)

        gram = gram / tf.cast(h * w, tf.float32)

        return gram

    def generate_features(self):
        """
        Extract style and content features from the reference images.

        Stores the Gram matrices of style features in
        gram_style_features and the content features in content_feature.
        """
        vgg19 = tf.keras.applications.vgg19

        style_input = vgg19.preprocess_input(
            self.style_image * 255.0
        )

        content_input = vgg19.preprocess_input(
            self.content_image * 255.0
        )

        style_outputs = self.model(style_input)
        content_outputs = self.model(content_input)

        style_features = style_outputs[:-1]
        self.content_feature = content_outputs[-1]

        self.gram_style_features = [
            self.gram_matrix(feature)
            for feature in style_features
        ]

    def layer_style_cost(self, style_output, gram_target):
        """
        Calculate the style cost for one layer.

        Args:
            style_output: Tensor of rank 4 containing generated features.
            gram_target: Tensor with shape (1, c, c).

        Returns:
            Scalar tensor containing the style cost.

        Raises:
            TypeError: If either argument has an invalid type or shape.
        """
        if not isinstance(
                style_output, (tf.Tensor, tf.Variable)
        ) or len(style_output.shape) != 4:
            raise TypeError("style_output must be a tensor of rank 4")

        channels = style_output.shape[-1]

        if not isinstance(
                gram_target, (tf.Tensor, tf.Variable)
        ) or len(gram_target.shape) != 3 or \
                gram_target.shape != (1, channels, channels):
            raise TypeError(
                "gram_target must be a tensor of shape [1, {}, {}]".format(
                    channels, channels
                )
            )

        gram_style = self.gram_matrix(style_output)

        return tf.reduce_mean(
            tf.square(gram_style - gram_target)
        )

    def style_cost(self, style_outputs):
        """
        Calculate the total style cost across all selected layers.

        Args:
            style_outputs: List of generated style layer outputs.

        Returns:
            Scalar tensor containing the average style cost.

        Raises:
            TypeError: If style_outputs is not a list with the expected
                number of elements.
        """
        length = len(self.style_layers)

        if not isinstance(style_outputs, list) or \
                len(style_outputs) != length:
            raise TypeError(
                "style_outputs must be a list with a length of {}".format(
                    length
                )
            )

        weight = 1.0 / length
        cost = tf.constant(0.0, dtype=tf.float32)

        for i in range(length):
            cost += weight * self.layer_style_cost(
                style_outputs[i],
                self.gram_style_features[i]
            )

        return cost
