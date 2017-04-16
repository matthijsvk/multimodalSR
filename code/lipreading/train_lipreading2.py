from __future__ import print_function

import os
import time
from tqdm import tqdm
import lasagne
import numpy as np

import preprocessLipreading

import logging
logger_train = logging.getLogger('lipreading.train')
logger_train.setLevel(logging.DEBUG)


# Given a dataset and a model, this function trains the model on the dataset for several epochs
# (There is no default trainer function in Lasagne yet)
def train(#X_train, y_train, X_val, y_val, X_test, y_test,
          train_fn, val_fn,
          network_output_layer,
          batch_size,
          LR_start, LR_decay,
          num_epochs,
          dataset,
          loadPerSpeaker=False,
          save_path=None,
          shuffleEnabled=True):

    if loadPerSpeaker:
        trainingSpeakerFiles, testSpeakerFiles = dataset
        logger_train.info("train files: \n%s", trainingSpeakerFiles)
        logger_train.info("test files:  \n %s", testSpeakerFiles)
    else:
        X_train, y_train, X_val, y_val, X_test, y_test = dataset
        logger_train.info("the number of training examples is: %s", len(X_train))
        logger_train.info("the number of valid examples is:    %s", len(X_val))
        logger_train.info("the number of test examples is:     %s", len(X_test))


    # A function which shuffles a dataset
    def shuffle(X, y):
        shuffle_parts = 1
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

    # This function runs the model a full epoch (on the whole dataset)
    # is a LR is specified, the model will be trained, and the ouput is 'cost' (no accuracy for speed)
    # if no LR is specified, there's not training and the output is 'cost, accuracy'
    def run_epoch(X, y, LR=None):
        loss = 0;
        lss = 0;
        error = 0;
        err = 0;
        nb_batches = len(X) / batch_size

        for i in tqdm(range(nb_batches), total=nb_batches):
            batch_X = X[i * batch_size:(i + 1) * batch_size]
            batch_y = y[i * batch_size:(i + 1) * batch_size]
            # logger_train.info("batch_X.shape: %s", batch_X.shape)
            # logger_train.info("batch_y.shape: %s", batch_y.shape)
            if LR != None:  lss = train_fn(batch_X, batch_y, LR)  # training
            else:           lss, err = val_fn(batch_X, batch_y)  # validation
            loss += lss;  error += err

        loss /= nb_batches
        error /= nb_batches

        if LR != None:   return loss, nb_batches  # for training, only loss (faster)
        else:            return loss, error, nb_batches  # for validation, get both

    # This function trains the model a full epoch (on the whole dataset)
    def train_epoch(X, y, LR):
        loss = 0
        print("training with a batchsize of: ", batch_size)
        nb_batches = len(X) / batch_size
        print("len X: ", len(X))
        print("so number of batches per epoch: ", nb_batches)

        for i in tqdm(range(nb_batches), total=nb_batches):
            batch_X = X[i * batch_size:(i + 1) * batch_size]
            batch_y = y[i * batch_size:(i + 1) * batch_size]
            # print("batch_X.shape: ", batch_X.shape)
            # print("batch_y.shape: ", batch_y.shape)
            loss += train_fn(batch_X, batch_y, LR)

        loss /= nb_batches

        return loss, nb_batches

    # This function tests the model a full epoch (on the whole dataset)
    def val_epoch(X, y):
        err = 0
        loss = 0
        nb_batches = len(X) / batch_size

        for i in range(nb_batches):
            new_loss, new_err = val_fn(X[i * batch_size:(i + 1) * batch_size], y[i * batch_size:(i + 1) * batch_size])
            err += new_err
            loss += new_loss

        err = err / nb_batches * 100
        loss /= nb_batches

        return err, loss, nb_batches

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

        train_loss, nb_train_batches = train_epoch(X_train, y_train, LR)
        X_train, y_train = shuffle(X_train, y_train)

        val_err, val_loss, nb_val_batches = val_epoch(X_val, y_val)

        # test if validation error went down
        if val_err <= best_val_err:

            best_val_err = val_err
            best_epoch = epoch + 1

            test_err, test_loss, nb_test_batches = val_epoch(X_test, y_test)

            if save_path is None:
                save_path = "./bestModel"
            if not os.path.exists(os.path.dirname(save_path)):
                os.makedirs(os.path.dirname(save_path))
            np.savez(save_path, *lasagne.layers.get_all_param_values(network_output_layer))

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
