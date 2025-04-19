import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

import numpy as np
import tensorflow as tf
import keras

# tenseurs
a = np.array([[1, 2], [3, 4]])
a = tf.constant(a)
print((a + 2).numpy())


