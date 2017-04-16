from __future__ import print_function

import os
import time
from tqdm import tqdm
import lasagne
import numpy as np
import preprocessLipreading
import general_tools

import logging
logger_train = logging.getLogger('lipreading.train')
logger_train.setLevel(logging.DEBUG)


# Given a dataset and a model, this function trains the model on the dataset for several epochs
# (There is no default trainer function in Lasagne yet)
def train(train_fn, val_fn,
          network_output_layer,
          batch_size,
          LR_start, LR_decay,
          num_epochs,
          dataset,
          loadPerSpeaker=False,
          save_name=None,
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

    # This function trains the model a full epoch (on the whole dataset)
    def train_epoch(X, y, LR):
        cost = 0
        nb_batches = len(X) / batch_size

        for i in tqdm(range(nb_batches), total=nb_batches):
            batch_X = X[i * batch_size:(i + 1) * batch_size]
            batch_y = y[i * batch_size:(i + 1) * batch_size]
            # print("batch_X.shape: ", batch_X.shape)
            # print("batch_y.shape: ", batch_y.shape)
            cost += train_fn(batch_X, batch_y, LR)

        return cost, nb_batches

    # This function tests the model a full epoch (on the whole dataset)
    def val_epoch(X, y):
        err = 0
        cost = 0
        nb_batches = len(X) / batch_size

        for i in tqdm(range(nb_batches)):
            batch_X = X[i * batch_size:(i + 1) * batch_size]
            batch_y = y[i * batch_size:(i + 1) * batch_size]
            new_cost, new_err = val_fn(batch_X, batch_y)
            err += new_err
            cost += new_cost

        return err, cost, nb_batches

    # evaluate many TRAINING speaker files -> train loss, val loss and vall error. Load them in one by one (so they fit in memory)
    def evalTRAINING(trainingSpeakerFiles, LR, shuffleEnabled, verbose=False):
        train_cost = 0;
        val_err = 0;
        val_cost = 0;
        nb_train_batches = 0;
        nb_val_batches = 0;
        # for each speaker, pass over the train set, then val set. (test is other files). save the results.
        for speakerFile in tqdm(trainingSpeakerFiles, total=len(trainingSpeakerFiles)):
            # TODO: pallelize this with the GPU evaluation to eliminate waiting
            logger_train.debug("processing %s", speakerFile)
            X_train, y_train, X_val, y_val, X_test, y_test = preprocessLipreading.prepLip_one(speakerFile=speakerFile,
                                                                                              trainFraction=0.8,
                                                                                              validFraction=0.2)
            if verbose:
                logger_train.debug("the number of training examples is: %s", len(X_train))
                logger_train.debug("the number of valid examples is:    %s", len(X_val))
                logger_train.debug("the number of test examples is:     %s", len(X_test))

            if shuffleEnabled: X_train, y_train = shuffle(X_train, y_train)
            train_cost_one, train_batches_one = train_epoch(X=X_train, y=y_train, LR=LR)
            train_cost += train_cost_one;
            nb_train_batches += train_batches_one

            # get results for validation  set
            val_err_one, val_cost_one, val_batches_one = val_epoch(X=X_val, y=y_val)
            val_err += val_err_one;
            val_cost += val_cost_one;
            nb_val_batches += val_batches_one;

            if verbose:
                logger_train.debug("  this speaker results: ")
                logger_train.debug("\ttraining cost:     %s", train_cost_one / train_batches_one)
                logger_train.debug("\tvalidation cost:   %s", val_cost_one / val_batches_one)
                logger_train.debug("\vvalidation error rate:  %s %%", val_err_one / val_batches_one)

        # get the average over all speakers
        train_cost /= nb_train_batches
        val_err = val_err / nb_val_batches * 100  # convert to %
        val_cost /= nb_val_batches
        return train_cost, val_cost, val_err

    # evaluate many TEST speaker files. Load them in one by one (so they fit in memory)
    def evalTEST(testSpeakerFiles, verbose=False):
        test_err = 0;
        test_cost = 0;
        nb_test_batches = 0;
        # for each speaker, pass over the train set, then test set. (test is other files). save the results.
        for speakerFile in tqdm(testSpeakerFiles, total=len(testSpeakerFiles)):
            # TODO: pallelize this with the GPU evaluation to eliminate waiting
            logger_train.debug("processing %s", speakerFile)
            X_train, y_train, X_val, y_val, X_test, y_test = preprocessLipreading.prepLip_one(
                    speakerFile=speakerFile, trainFraction=0.0, validFraction=0.0)

            if verbose:
                logger_train.debug("the number of training examples is: %s", len(X_train))
                logger_train.debug("the number of valid examples is:    %s", len(X_val))
                logger_train.debug("the number of test examples is:     %s", len(X_test))

            # get results for testidation  set
            test_err_one, test_cost_one, test_batches_one = val_epoch(X=X_test, y=y_test)
            test_err += test_err_one;
            test_cost += test_cost_one;
            nb_test_batches += test_batches_one;

            if verbose:
                logger_train.debug("  this speaker results: ")
                logger_train.debug("\ttest cost:   %s", test_cost_one / test_batches_one)
                logger_train.debug("\vtest error rate:  %s %%", test_err_one / test_batches_one)

        # get the average over all speakers
        test_err = test_err / nb_test_batches * 100
        test_cost /= nb_test_batches
        return test_cost, test_err



    best_val_err = 100
    best_epoch = 1
    LR = LR_start
    # for storage of training info
    network_train_info = np.zeros([5, num_epochs])  #train cost, val cost, val acc, test cost, test acc

    logger_train.info("starting training for %s epochs...", num_epochs)
    for epoch in range(num_epochs):
        logger_train.info("\n\n\n Epoch %s started", epoch + 1)
        start_time = time.time()

        if not loadPerSpeaker:
            total_train_cost, nb_train_batches = train_epoch(X=X_train, y=y_train, LR=LR); train_cost = total_train_cost / nb_train_batches
            X_train, y_train = shuffle(X_train, y_train)

            val_err, val_cost, nb_val_batches = val_epoch(X=X_val, y=y_val)
            val_err = val_err / nb_val_batches * 100; val_cost /=nb_val_batches

        else:
            train_cost, val_cost, val_err = evalTRAINING(trainingSpeakerFiles, LR, shuffleEnabled)

        # test if validation error went down
        printTest = False
        if val_err <= best_val_err:
            printTest = True
            best_val_err = val_err
            best_epoch = epoch + 1

            logger_train.info("\n\nBest ever validation score; evaluating TEST set...")

            if not loadPerSpeaker:  #all at once
                test_err, test_cost, nb_test_batches = val_epoch(X_test, y_test)
                test_err = test_err / nb_test_batches * 100;  test_cost /= nb_test_batches

            else:  # process each speaker seperately
                test_cost, test_err = evalTEST(testSpeakerFiles)

            logger_train.info("TEST results: ")
            logger_train.info("\t  test cost:        %s", str(test_cost))
            logger_train.info("\t  test error rate:  %s %%", str(test_err))

            if save_name is None:
                save_name = "./bestModel"
            if not os.path.exists(os.path.dirname(save_name)):
                os.makedirs(os.path.dirname(save_name))
            logger_train.info("saving model to %s", save_name)
            np.savez(save_name, *lasagne.layers.get_all_param_values(network_output_layer))

        epoch_duration = time.time() - start_time

        # Then we logger_train.info the results for this epoch:
        logger_train.info("Epoch %s of %s took %s seconds", epoch + 1, num_epochs, epoch_duration)
        logger_train.info("  LR:                            %s",    str(LR))
        logger_train.info("  training cost:                 %s",    str(train_cost))
        logger_train.info("  validation cost:               %s",    str(val_cost))
        logger_train.info("  validation error rate:         %s %%", str(val_err))
        logger_train.info("  best epoch:                    %s",    str(best_epoch))
        logger_train.info("  best validation error rate:    %s %%", str(best_val_err))
        if printTest:
            logger_train.info("  test cost:                 %s",    str(test_cost))
            logger_train.info("  test error rate:           %s %%", str(test_err))

        # save the training info
        network_train_info[0][epoch] = train_cost
        network_train_info[1][epoch] = val_cost
        network_train_info[2][epoch] = 100 - val_err
        network_train_info[3][epoch] = test_cost
        network_train_info[4][epoch] = 100 - test_err
        store_path = save_name + '_trainInfo.pkl'
        general_tools.saveToPkl(store_path, network_train_info)
        logger_train.info("Train info written to:\t %s", store_path)

        # decay the LR
        LR *= LR_decay

    logger_train.info("Done.")





