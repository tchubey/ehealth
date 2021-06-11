import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow.keras.backend as K

def sparse_crossentropy_masked(y_true, y_pred, pad_idx = 0):
    y_true_masked = tf.boolean_mask(y_true, tf.not_equal(y_true, pad_idx))
    y_pred_masked = tf.boolean_mask(y_pred, tf.not_equal(y_true, pad_idx))
    return K.mean(K.sparse_categorical_crossentropy(y_true_masked, y_pred_masked))



def plot_hist(history, validation, name = "history_plot"):
    try:
        hist = history.history
        (_, axs) = plt.subplots(3,1)
        epochs = np.arange(0, len(hist['loss']))
        axs[0].plot(epochs-0.5, hist['relations_loss'], label = "train_loss")
        axs[1].plot(epochs-0.5, hist['bilou_loss'], label = "train_loss")
        axs[2].plot(epochs-0.5, hist['entities_loss'], label = "train_loss")
        if validation:
           axs[0].plot(epochs, hist['val_relations_loss'], label = "valid_loss")
           axs[1].plot(epochs, hist['val_bilou_loss'], label = "valid_loss")
           axs[2].plot(epochs, hist['val_entities_loss'], label = "valid_loss")

        axs[0].set_title("Relation")
        axs[1].set_title("BILOU")
        axs[2].set_title("Entity")

        for i in range(0,3):
           axs[i].set_xlabel("Epoch")
           axs[i].set_ylabel("Loss")

        plt.savefig(f'{name}.png')

    except: print("<<<<<Error have occurred while plotting learning curve. Couldnt save the figure>>>>")
        