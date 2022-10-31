import keras
import tensorflow as tf
# import tensorflow_federated as tff
from keras.losses import binary_crossentropy

from models import default_model


# TODO!
def train_model(x_train):
    # training_process = tff.learning.reconstruction.build_training_process(
    #     model_fn=default_model,
    #     loss_fn=binary_crossentropy,
    #     metrics_fn=None,
    #     server_optimizer_fn=lambda: tf.keras.optimizers.SGD(1.0),
    #     client_optimizer_fn=lambda: tf.keras.optimizers.SGD(0.5),
    #     reconstruction_optimizer_fn=lambda: tf.keras.optimizers.SGD(0.1)
    # )

    return
