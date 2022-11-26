import numpy as np
import tensorflow as tf


def compute_content_cost(content_output, generated_output):
    """
    Computes the content cost

    Arguments:
    a_C -- tensor of dimension (1, n_H, n_W, n_C), hidden layer activations representing content of the image C
    a_G -- tensor of dimension (1, n_H, n_W, n_C), hidden layer activations representing content of the image G

    Returns:
    J_content -- scalar that you compute using equation 1 above.
    """
    a_C = content_output[-1]
    a_G = generated_output[-1]

    ### START CODE HERE

    # Retrieve dimensions from a_G (≈1 line)
    _, n_H, n_W, n_C = a_G.get_shape().as_list()

    entries = n_H * n_W * n_C

    diff = tf.reshape(tf.subtract(a_C, a_G), [entries, 1])
    error = tf.linalg.matmul(tf.transpose(diff), diff)  # L2 norm: ||diff||^2

    J_content = error / (4 * entries)

    # Reshape a_C and a_G (≈2 lines)
    #a_C_unrolled = None
    #a_G_unrolled = None

    # compute the cost with tensorflow (≈1 line)
    #J_content = None
    ### END CODE HERE

    return J_content


def compute_layer_style_cost(a_S, a_G):
    """
    Arguments:
    a_S -- tensor of dimension (1, n_H, n_W, n_C), hidden layer activations representing style of the image S
    a_G -- tensor of dimension (1, n_H, n_W, n_C), hidden layer activations representing style of the image G

    Returns:
    J_style_layer -- tensor representing a scalar value, style cost defined above by equation (2)
    """
    ### START CODE HERE

    # Retrieve dimensions from a_G (≈1 line)
    _, n_H, n_W, n_C = a_G.get_shape().as_list()

    entries_per_channel = n_H * n_W
    entries = entries_per_channel * n_C

    # Reshape the images from (n_H * n_W, n_C) to have them of shape (n_C, n_H * n_W) (≈2 lines)
    # Note: transpose is because the activation dims are (m, n_H, n_W, n_C)
    # However, the desired unrolled matrix shape is (n_C, n_H*n_W). If no transpose, the shape will be (n_H*n_W, n_C)
    a_S_matrix = tf.reshape(tf.transpose(a_S), [n_C, entries_per_channel])
    a_G_matrix = tf.reshape(tf.transpose(a_G), [n_C, entries_per_channel])

    def outerProduct(A):
        # If A is (n, m) shape, the returned shape will be (n, n)
        return tf.linalg.matmul(A, tf.transpose(A))

    # Computing gram_matrices for both images S and G (≈2 lines)
    GS = outerProduct(a_S_matrix)  # "y"    (target)     [n_C, n_C]
    GG = outerProduct(a_G_matrix)  # "yhat" (prediction) [n_C, n_C]

    diff = tf.subtract(GS, GG)
    error = tf.reshape(diff, [-1, 1])
    _error_squared = tf.linalg.matmul(tf.transpose(error), error)

    divider = 4 * entries*entries

    # Computing the loss (≈1 line)
    J_style_layer = _error_squared / divider #tf.reduce_sum(tf.square(diff)) / divider

    ### END CODE HERE

    return J_style_layer


def compute_style_cost(style_image_output, generated_image_output, STYLE_LAYERS):
    """
    Computes the overall style cost from several chosen layers

    Arguments:
    style_image_output -- our tensorflow model
    generated_image_output --
    STYLE_LAYERS -- A python list containing:
                        - the names of the layers we would like to extract style from
                        - a coefficient for each of them

    Returns:
    J_style -- tensor representing a scalar value, style cost defined above by equation (2)
    """

    # initialize the overall style cost
    J_style = 0

    # Set a_S to be the hidden layer activation from the layer we have selected.
    # The last element of the array contains the content layer image, which must not be used.
    a_S = style_image_output[:-1]

    # Set a_G to be the output of the chosen hidden layers.
    # The last element of the list contains the content layer image which must not be used.
    a_G = generated_image_output[:-1]
    for i, weight in zip(range(len(a_S)), STYLE_LAYERS):
        # Compute style_cost for the current layer
        J_style_layer = compute_layer_style_cost(a_S[i], a_G[i])

        # Add weight * J_style_layer of this layer to overall style cost
        J_style += weight[1] * J_style_layer

    return J_style


@tf.function()
def total_cost(J_content, J_style, alpha=10, beta=40):
    """
    Computes the total cost function

    Arguments:
    J_content -- content cost coded above
    J_style -- style cost coded above
    alpha -- hyperparameter weighting the importance of the content cost
    beta -- hyperparameter weighting the importance of the style cost

    Returns:
    J -- total cost as defined by the formula above.
    """
    ### START CODE HERE

    #(≈1 line)
    J = (alpha * J_content) + (beta * J_style)

    ### START CODE HERE

    return J
