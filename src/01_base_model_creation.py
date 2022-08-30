import argparse
import os
from tqdm import tqdm
import logging
from utils.common import read_yaml, create_directories
import random
import tensorflow as tf
import numpy as np

STAGE = "Creating Base Model" ## <<< change stage name 

logging.basicConfig(
    filename=os.path.join("logs", 'running_logs.log'), 
    level=logging.INFO, 
    format="[%(asctime)s: %(levelname)s: %(module)s]: %(message)s",
    filemode="a"
    )


def main(config_path):
    ## read config files
    config = read_yaml(config_path)
    
    ## get the data

    (X_train_full, y_train_full), (X_test, y_test) = tf.keras.datasets.mnist.load_data()
    X_train_full = X_train_full/255.0
    X_test = X_test/255
    X_valid, X_train = X_train_full[:5000], X_train_full[5000:]
    y_valid, y_train = y_train_full[:5000], y_train_full[5000:]
    
    ## set the seed
    seed = 2021
    tf.random.set_seed(seed)
    np.random.seed(seed)

    ## define the layers

    Layers = [
        tf.keras.layers.Flatten(input_shape = [28,28], name = "inputLayer"),
        tf.keras.layers.Dense(300, name = "hiddenLayer1"),
        tf.keras.layers.LeakyReLU(),
        tf.keras.layers.Dense(100, name = "hiddenLayer2"),
        tf.keras.layers.LeakyReLU(),
        tf.keras.layers.Dense(10, activation = "softmax", name = "outputLayer")
    ]

    ## define the model and compile it

    model = tf.keras.models.Sequential(Layers)

    LOSS_FUNCTION = "sparse_categorical_crossentropy"    #------ Used for classification problems
    OPTIMIZER = tf.keras.optimizers.SGD(learning_rate=1e-3)
    METRICS = ["accuracy"]
    model.compile(loss = LOSS_FUNCTION, optimizer = OPTIMIZER, metrics = METRICS)

    model.summary()

    ## Train the model 

    EPOCHS = 30
    VALIDATION = (X_valid, y_valid)

    history = model.fit(X_train, y_train, epochs = EPOCHS, validation_data=VALIDATION)

    ## save the model 

    model_dir_path = os.path.join("artifacts", "models")
    create_directories([model_dir_path])

    model_file_path = os.path.join(model_dir_path, "base_model.h5")
    model.save(model_file_path)

    logging.info(f"Base model is saved at {model_file_path}") 
    logging.info(f"Evaluation metrics {model.evaluate(X_test, y_test)}")

    
if __name__ == '__main__':
    args = argparse.ArgumentParser()
    args.add_argument("--config", "-c", default="configs/config.yaml")
    parsed_args = args.parse_args()

    try:
        logging.info("\n********************")
        logging.info(f">>>>> stage {STAGE} started <<<<<")
        main(config_path=parsed_args.config)
        logging.info(f">>>>> stage {STAGE} completed!<<<<<\n")
    except Exception as e:
        logging.exception(e)
        raise e