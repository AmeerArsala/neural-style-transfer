import numpy as np
import tensorflow as tf

import matplotlib.pyplot as plt

import src.nst.cost as cost
from src.utils import add_uniform_noise, clip_0_1, tensor_to_image, printProgressBar


def reformat_data_if_invalid(img):
    if type(img) is tf.Variable:
        return img
    else:
        # not in tensorflow form; convert it to something tensorflow can easily use
        return tf.Variable(tf.image.convert_image_dtype(img, tf.float32))


class StyleInputs:
    def __init__(self, image, layer_delegations):
        super().__init__()
        self.image = image
        self.layer_delegations = layer_delegations

    def validate(self):
        reformat_data_if_invalid(self.image)


class NstInputs:
    def __init__(self, content_img, style: StyleInputs, generated_img=None):
        super().__init__()
        self.content_img = content_img
        self.style = style
        self.generated_img = generated_img

    def validate(self):
        reformat_data_if_invalid(self.content_img)
        self.style.validate()

        if self.generated_img is not None:
            reformat_data_if_invalid(self.generated_img)


class NstHyperparameters:
    # alpha = J_content coefficient, beta = J_style coefficient
    def __init__(self, optimizer, alpha, beta):
        super().__init__()
        self.optimizer = optimizer
        self.alpha = alpha
        self.beta = beta


class NeuralStyleTransfer:
    def __init__(self,
                 encode,
                 nst_inputs=NstInputs(
                     content_img=None,
                     style=StyleInputs(image=None, layer_delegations=None),
                     generated_img=None),
                 hyperparameters=NstHyperparameters(
                     optimizer=tf.keras.optimizers.Adam(learning_rate=0.01),
                     alpha=10, beta=40)
                 ):
        super().__init__()
        self.encode = encode  # either an encoder or a function
        self.nst_inputs = nst_inputs
        self.hyperparameters = hyperparameters
        self.output_image = None
        self.epoch_outputs = None
        self.a_C = None
        self.a_S = None

        self.update_encodings()

    def update_encodings(self):
        if self.nst_inputs.content_img is not None:
            self.a_C = self.encode(self.nst_inputs.content_img)

        if self.nst_inputs.style is not None:
            self.a_S = self.encode(self.nst_inputs.style.image)

    def validate(self):
        self.nst_inputs.validate()

    @tf.function()
    def train_step(self, generated_image):
        with tf.GradientTape() as tape:
            a_G = self.encode(generated_image)  # generated image encoding

            J_content = cost.compute_content_cost(self.a_C, a_G)
            J_style = cost.compute_style_cost(self.a_S, a_G, self.nst_inputs.style.layer_delegations)

            J = cost.total_cost(J_content, J_style, alpha=self.hyperparameters.alpha, beta=self.hyperparameters.beta)

        grad = tape.gradient(J, generated_image)
        self.hyperparameters.optimizer.apply_gradients([(grad, generated_image)])
        generated_image.assign(clip_0_1(generated_image))

        return J

    def train(self,
              nst_inputs=None,
              hyperparameters=None,
              safe=True,
              epochs=2500, epoch_interval=250,
              show_image_on_interval=True, output_dir_filename=None, ext="jpg"):
        if safe:
            # ingrain parameters
            if nst_inputs is not None:
                self.nst_inputs = nst_inputs
                self.update_encodings()

            if self.nst_inputs.generated_img is None:
                self.nst_inputs.generated_img = add_uniform_noise(self.nst_inputs.content_img)

            if hyperparameters is not None:
                self.hyperparameters = hyperparameters

            self.validate()

        # Training
        self.epoch_outputs = []
        for epoch in range(epochs+1):
            J = self.train_step(self.nst_inputs.generated_img)

            if epoch % epoch_interval == 0:
                print(f"Epoch {epoch}\nJ = {J}")
                image = tensor_to_image(self.nst_inputs.generated_img)
                self.epoch_outputs.append(image)

                if type(output_dir_filename) is str:
                    image.save(output_dir_filename + str(epoch) + "." + ext)

                if show_image_on_interval:
                    plt.imshow(image)
                    plt.show()

            # show progress bar update
            printProgressBar(epoch+1, epochs)

        self.output_image = tensor_to_image(self.nst_inputs.generated_img)
        print("Training Complete.")

        return self.output_image

    def save_image(self, output_dir_filename="out/image", ext=".jpg"):
        filepath = output_dir_filename + ext
        self.output_image.save(filepath)