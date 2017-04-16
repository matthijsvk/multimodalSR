from __future__ import print_function

import os
import time
from tqdm import tqdm
import lasagne
import numpy as np


# Given a dataset and a model, this function trains the model on the dataset for several epochs
# (There is no default trainer function in Lasagne yet)
def train(train_fn, val_fn,
          model,
          batch_size,
          LR_start, LR_decay,
          num_epochs,
          X_train, y_train,
          X_val, y_val,
          X_test, y_test,
          save_path=None,
          shuffle_parts=1):
    # A function which shuffles a dataset
    def shuffle(X, y):

        chunk_size = len(X) / shuffle_parts
        shuffled_range = range(chunk_size)

        X_buffer = np.copy(X[0:chunk_size])
        y_buffer = np.copy(y[0:chunk_size])

        for k in range(shuffle_parts):

            np.random.shuffle(shuffled_range)

            for i in range(chunk_size):
                X_buffer[i] = X[k * chunk_size + shuffled_range[i]]
                y_buffer[i] = y[k * chunk_size + shuffled_range[i]]

            X[k * chunk_size:(k + 1) * chunk_size] = X_buffer
            y[k * chunk_size:(k + 1) * chunk_size] = y_buffer

        return X, y

    # This function trains the model a full epoch (on the whole dataset)
    def train_epoch(X, y, LR):
        loss = 0
        print("training with a batchsize of: ", batch_size)
        nb_batches = len(X) / batch_size
        print("len X: ", len(X))
        print("so number of batches per epoch: ", nb_batches)

        for i in tqdm(range(nb_batches),total=nb_batches):
            batch_X = X[i * batch_size:(i + 1) * batch_size]
            batch_y = y[i * batch_size:(i + 1) * batch_size]
            # print("batch_X.shape: ", batch_X.shape)
            # print("batch_y.shape: ", batch_y.shape)
            loss += train_fn(batch_X, batch_y , LR)

        loss /= nb_batches

        return loss

    # This function tests the model a full epoch (on the whole dataset)
    def val_epoch(X, y):
        err = 0
        loss = 0
        batches = len(X) / batch_size

        for i in range(batches):
            new_loss, new_err = val_fn(X[i * batch_size:(i + 1) * batch_size], y[i * batch_size:(i + 1) * batch_size])
            err += new_err
            loss += new_loss

        err = err / batches * 100
        loss /= batches

        return err, loss

    # shuffle the train set
    X_train, y_train = shuffle(X_train, y_train)
    best_val_err = 100
    best_epoch = 1
    LR = LR_start

    # We iterate over epochs:
    print("starting training for ", num_epochs, " epochs...")
    for epoch in range(num_epochs):
        print("epoch ", epoch + 1, "started...")
        start_time = time.time()

        train_loss = train_epoch(X_train, y_train, LR)
        X_train, y_train = shuffle(X_train, y_train)

        val_err, val_loss = val_epoch(X_val, y_val)

        # test if validation error went down
        if val_err <= best_val_err:

            best_val_err = val_err
            best_epoch = epoch + 1

            test_err, test_loss = val_epoch(X_test, y_test)

            if save_path is None:
                save_path = "./bestModel"
            if not os.path.exists(os.dirname(save_path)):
                os.makedirs(os.dirname(save_path))
            np.savez(save_path, *lasagne.layers.get_all_param_values(model))

        epoch_duration = time.time() - start_time

        # Then we print the results for this epoch:
        print("Epoch " + str(epoch + 1) + " of " + str(num_epochs) + " took " + str(epoch_duration) + "s")
        print("  LR:                            " + str(LR))
        print("  training loss:                 " + str(train_loss))
        print("  validation loss:               " + str(val_loss))
        print("  validation error rate:         " + str(val_err) + "%")
        print("  best epoch:                    " + str(best_epoch))
        print("  best validation error rate:    " + str(best_val_err) + "%")
        print("  test loss:                     " + str(test_loss))
        print("  test error rate:               " + str(test_err) + "%")

        # decay the LR
        LR *= LR_decay

    print("Done.")
