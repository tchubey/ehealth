import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow.keras.backend as K


def plot_hist(history, validation, name = "history_plot"):
    legend = ['Train']
    plt.plot(history.history['loss'])
    if validation:
        plt.plot(history.history['val_loss'])
        legend.extend('Validation')
    plt.xlabel('Epoch')
    plt.legend(legend)
    plt.savefig(f'{name}.png')


def sparse_crossentropy_masked(y_true, y_pred):
    y_true_masked = tf.boolean_mask(y_true, tf.not_equal(y_true, -1))
    y_pred_masked = tf.boolean_mask(y_pred, tf.not_equal(y_true, -1))
    return K.mean(K.sparse_categorical_crossentropy(y_true_masked, y_pred_masked))
        