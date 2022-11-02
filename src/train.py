import keras
import numpy as np
import tensorflow as tf
# import tensorflow_federated as tff
from keras.optimizers import Adam, SGD
from keras.losses import binary_crossentropy

from spec import *
from models import default_model


def get_train_instances(train, num_negatives):
    user_input, item_input, labels = [], [], []
    n_users, n_items = train.shape[0], train.shape[1]
    for (u, i) in train.keys():
        # positive instance
        user_input.append(u)
        item_input.append(i)
        labels.append(1)
        # negative instances
        # TODO: this random choosing of `j` can probably be optimized
        for t in range(num_negatives):
            j = np.random.randint(n_items)
            while (u, j) in train:
                j = np.random.randint(n_items)
            user_input.append(u)
            item_input.append(j)
            labels.append(0)
    return user_input, item_input, labels


def train_model(model, train_data, epochs=1, batch_size=32, save=True):
    user_input, item_input, labels = get_train_instances(train_data, NUM_NEGATIVES)

    model.compile(optimizer=SGD(learning_rate=LEARNING_RATE), loss=binary_crossentropy)
    model.fit([np.array(user_input), np.array(item_input)], np.array(labels),
              batch_size=batch_size, epochs=epochs, verbose=True, shuffle=True)
    if save:
        model.save_weights(SAVED_MODEL_PATH)

    return model


def train_federated(x_train):
    # TODO!
    # training_process = tff.learning.reconstruction.build_training_process(
    #     model_fn=default_model,
    #     loss_fn=binary_crossentropy,
    #     metrics_fn=None,
    #     server_optimizer_fn=lambda: tf.keras.optimizers.SGD(1.0),
    #     client_optimizer_fn=lambda: tf.keras.optimizers.SGD(0.5),
    #     reconstruction_optimizer_fn=lambda: tf.keras.optimizers.SGD(0.1)
    # )

    return
