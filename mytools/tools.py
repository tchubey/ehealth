import matplotlib.pyplot as plt

def plot_hist(history, name = "history_plot"):
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Valid'])
    plt.savefig(f'{name}.png')