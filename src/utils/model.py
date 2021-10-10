import tensorflow as tf
import time
import os
import pandas as pd
import matplotlib.pyplot as plt


def create_model(LOSS_FUNCTION, OPTIMIZER, METRICS, NUM_CLASSES):

    LAYERS = [tf.keras.layers.Flatten(input_shape=[28, 28], name="inputLayer"),
          tf.keras.layers.Dense(300, activation="relu", name="hiddenLayer1"),
          tf.keras.layers.Dense(100, activation="relu", name="hiddenLayer2"),
          tf.keras.layers.Dense(NUM_CLASSES, activation="softmax", name="outputLayer")]

    model_clf = tf.keras.models.Sequential(LAYERS)

    model_clf.summary()

    model_clf.compile(loss=LOSS_FUNCTION,
                optimizer=OPTIMIZER,
                metrics=METRICS)

    return model_clf ## <<< untrained model

def get_unique_filename(filename):
    unique_filename = time.strftime(f"%Y%m%d_%H%M%S_{filename}")
    return unique_filename

def save_model(model, model_name, model_dir):
    unique_filename = get_unique_filename(model_name)
    path_to_model = os.path.join(model_dir, unique_filename)
    model.save(path_to_model)

def plot_history(history, file_name):
    """It plots the fitting history and saves it to a file.
    Args:
        history: Evaluation data during training
        file_name (str): Filename to save the plot to
    """
    #logging.info('Plotting training history')
    pd.DataFrame(history.history).plot(figsize=(10,7))
    plt.grid(True)
    plt.show()
    plot_dir = "plots"
    os.makedirs(plot_dir, exist_ok=True) # ONLY CREATE IF MODEL_DIR DOESN"T EXISTS
    plotPath = os.path.join(plot_dir, file_name) # model/filename
    plt.savefig(plotPath)
    #logging.info('Saving the plots at {plotPath}')
    #logging.info("#####"*15)