import os
from pathlib import Path
from absl import app
from absl import flags
from absl import logging
import gin
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import datasets
from dense import cv_loss
from plot_eval import Eval


tfkl = tf.keras.layers
tfk = tf.keras

DATA_DIR = Path(os.environ["NeuralBoltzmann_DATA"])
MODEL_DIR = Path(os.environ["NeuralBoltzmann_MODEL"])

def retrain(model, x_train, y_train, learning_rate, batch_size=100, epochs=10):
    
    optimizer = tf.optimizers.Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss='mse')
    model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, verbose=2, shuffle=True)
    return model

   
def main(argv):
    del argv

    dset = datasets.LoadDataset()
    (x_train, y_train) = dset["train"]
    (x_test, y_test) = dset["raw_test"]
    del dset
    model_dir =  MODEL_DIR / "dense_128_tt"/ "checkpoints" / "checkpoint" 
#     model_dir = Path('/home/pc/codes/NeuralBoltzmann/dense_128_tt/checkpoints/checkpoint/')
#     model = tfk.models.load_model(model_dir, custom_objects={'cv_loss': cv_loss })
    model = tfk.models.load_model(model_dir, compile=False)
    
    model = retrain(model, x_train, y_train,1e-4)
    model = retrain(model, x_train, y_train,1e-5)
    model = retrain(model, x_train, y_train,1e-6)
    model.save(model_dir)

    results_dir = MODEL_DIR / 'results'/ 'dense_128_tt' 
    Eval(model,x_test,y_test)
    return

if __name__ =="__main__":
    gin.parse_config_file('./configs/dense_128_tt.gin')
    app.run(main)
