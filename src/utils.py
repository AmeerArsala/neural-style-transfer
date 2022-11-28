import numpy as np
import tensorflow as tf

from PIL import Image


# img_dimensions = (-1, -1) means no change to the dimensions of the image
def tf_load_img(filename, img_dimensions, prefix="../data/"):
    filepath = prefix + filename
    image = Image.open(filepath)

    (original_width, original_height) = image.size
    (height, width) = img_dimensions

    if height < 0:
        height = original_height

    if width < 0:
        width = original_width

    img = np.array(image.resize((height, width)))
    img = tf.constant(np.reshape(img, ((1,) + img.shape)))

    return img


def add_uniform_noise(img, min=-0.25, max=0.25):
    noisy_image = tf.Variable(tf.image.convert_image_dtype(img, tf.float32))

    noise = tf.random.uniform(tf.shape(noisy_image), min, max)
    noisy_image = tf.add(noisy_image, noise)
    noisy_image = tf.clip_by_value(noisy_image, clip_value_min=0.0, clip_value_max=1.0)

    return noisy_image


def clip_0_1(image):
    """
    Truncate all the pixels in the tensor to be between 0 and 1

    Arguments:
    image -- Tensor
    J_style -- style cost coded above

    Returns:
    Tensor
    """
    return tf.clip_by_value(image, clip_value_min=0.0, clip_value_max=1.0)


def tensor_to_image(tensor):
    """
    Converts the given tensor into a PIL image

    Arguments:
    tensor -- Tensor

    Returns:
    Image: A PIL image
    """
    tensor = tensor * 255
    tensor = np.array(tensor, dtype=np.uint8)
    if np.ndim(tensor) > 3:
        assert tensor.shape[0] == 1
        tensor = tensor[0]
    return Image.fromarray(tensor)

# Print iterations progress
def printProgressBar (iteration, total, prefix = '', suffix = '', decimals = 1, length = 100, fill = '█', printEnd = "\r"):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
        printEnd    - Optional  : end character (e.g. "\r", "\r\n") (Str)
    """
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    print(f'\r{prefix} |{bar}| {percent}% {suffix}', end = printEnd)
    # Print New Line on Complete
    if iteration == total:
        print()
