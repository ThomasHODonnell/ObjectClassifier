import tensorflow as tf
from tensorflow import keras
import os

#avoid out of memory (OOM) error when tf expands to fill vram 
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus: 
    tf.config.experimental.set_memory_growth(gpu, True)

