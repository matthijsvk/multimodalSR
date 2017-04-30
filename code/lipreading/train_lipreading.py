from __future__ import print_function

import os
import time
from tqdm import tqdm
import lasagne
import numpy as np
import preprocessLipreading
import general_tools
import theano

import logging
logger_train = logging.getLogger('lipreading.train')
logger_train.setLevel(logging.DEBUG)


# Given a dataset and a model, this function trains the model on the dataset for several epochs
# (There is no default trainer function in Lasagne yet)
def train(train_fn, val_fn, out_fn, topk_acc_fn, k,
          network_output_layer,
          batch_size,
          LR_start, LR_decay,
          num_epochs,
          dataset,
          database_binaryDir,
          storeProcessed,
          processedDir,
          loadPerSpeaker=False,
          save_name=None,
          shuffleEnabled=True):

    if loadPerSpeaker:
        trainingSpeakerFiles, testSpeakerFiles = dataset
        logger_train.info("train files: \n%s", [os.path.basename(speakerFile) for speakerFile in trainingSpeakerFiles])
        logger_train.info("test files:  \n %s", [os.path.basename(speakerFile) for speakerFile in testSpeakerFiles])
    else:
        X_train, y_train, X_val, y_val, X_test, y_test = dataset
        logger_train.info("the number of training examples is: %s", len(X_train))
        logger_train.info("the number of valid examples is:    %s", len(X_val))
        logger_train.info("the number of test examples is:     %s", len(X_test))

    #import pdb; pdb.set_trace()


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

        i=0
        for i in tqdm(range(nb_batches), total=nb_batches):
            batch_X = X[i * batch_size:(i + 1) * batch_size]
            batch_y = y[i * batch_size:(i + 1) * batch_size]
            # print("batch_X.shape: ", batch_X.shape)
            # print("batch_y.shape: ", batch_y.shape)
            cost += train_fn(batch_X, batch_y, LR)

            # if i==0:
            #     out = out_fn(batch_X)
            #     import pdb;pdb.set_trace()

        return cost, nb_batches

    # This function tests the model a full epoch (on the whole dataset)
    def val_epoch(X, y):
        err = 0
        cost = 0
        topk_acc = 0
        nb_batches = len(X) / batch_size

        for i in tqdm(range(nb_batches)):
            batch_X = X[i * batch_size:(i + 1) * batch_size]
            batch_y = y[i * batch_size:(i + 1) * batch_size]
            new_cost, new_acc, new_topk_acc = val_fn(batch_X, batch_y)
            err += new_acc
            cost += new_cost
            topk_acc += new_topk_acc

        return cost, err, topk_acc, nb_batches

    # evaluate many TRAINING speaker files -> train loss, val loss and vall acc. Load them in one by one (so they fit in memory)
    def evalTRAINING(trainingSpeakerFiles, LR, shuffleEnabled, verbose=False, sourceDataDir=None, storeProcessed = False, processedDir=None):
        train_cost = 0;
        val_acc = 0;
        val_cost = 0;
        val_topk_acc = 0;
        nb_train_batches = 0;
        nb_val_batches = 0;
        # for each speaker, pass over the train set, then val set. (test is other files). save the results.
        for speakerFile in tqdm(trainingSpeakerFiles, total=len(trainingSpeakerFiles)):
            # TODO: pallelize this with the GPU evaluation to eliminate waiting
            logger_train.debug("processing %s", speakerFile)
            X_train, y_train, X_val, y_val, X_test, y_test = preprocessLipreading.prepLip_one(speakerFile=speakerFile,
                                                                                              trainFraction=0.8,
                                                                                              validFraction=0.2,
                                                                                              sourceDataDir=sourceDataDir,
                                                                                              storeProcessed=storeProcessed,
                                                                                              processedDir=processedDir)
            if verbose:
                logger_train.debug("the number of training examples is: %s", len(X_train))
                logger_train.debug("the number of valid examples is:    %s", len(X_val))
                logger_train.debug("the number of test examples is:     %s", len(X_test))

            if shuffleEnabled: X_train, y_train = shuffle(X_train, y_train)
            train_cost_one, train_batches_one = train_epoch(X=X_train, y=y_train, LR=LR)
            train_cost += train_cost_one;
            nb_train_batches += train_batches_one

            # get results for validation  set
            val_cost_one, val_acc_one, val_topk_acc_one, val_batches_one = val_epoch(X=X_val, y=y_val)
            val_cost += val_cost_one;
            val_acc += val_acc_one;
            val_topk_acc += val_topk_acc_one
            nb_val_batches += val_batches_one;

            if verbose:
                logger_train.debug("  this speaker results: ")
                logger_train.debug("\ttraining cost:     %s", train_cost_one / train_batches_one)
                logger_train.debug("\tvalidation cost:   %s", val_cost_one / val_batches_one)
                logger_train.debug("\vvalidation acc rate:  %s %%", val_acc_one / val_batches_one * 100)
                logger_train.debug("\vvalidation top %s acc rate:  %s %%", k, val_topk_acc_one / val_batches_one * 100)

        # get the average over all speakers
        train_cost /= nb_train_batches
        val_cost /= nb_val_batches
        val_acc = val_acc / nb_val_batches * 100  # convert to %
        val_topk_acc = val_topk_acc / nb_val_batches * 100  # convert to %

        return train_cost, val_cost, val_acc, val_topk_acc

    # evaluate many TEST speaker files. Load them in one by one (so they fit in memory)
    def evalTEST(testSpeakerFiles, verbose=False, sourceDataDir=None, storeProcessed=False, processedDir=None):
        test_acc = 0;
        test_cost = 0; test_topk_acc = 0;
        nb_test_batches = 0;
        # for each speaker, pass over the train set, then test set. (test is other files). save the results.
        for speakerFile in tqdm(testSpeakerFiles, total=len(testSpeakerFiles)):
            logger_train.debug("processing %s", speakerFile)
            X_train, y_train, X_val, y_val, X_test, y_test = preprocessLipreading.prepLip_one(
                    speakerFile=speakerFile, trainFraction=0.0, validFraction=0.0, sourceDataDir=sourceDataDir, storeProcessed=storeProcessed, processedDir=processedDir)

            if verbose:
                logger_train.debug("the number of training examples is: %s", len(X_train))
                logger_train.debug("the number of valid examples is:    %s", len(X_val))
                logger_train.debug("the number of test examples is:     %s", len(X_test))

            # get results for testidation  set
            test_cost_one, test_acc_one, test_topk_acc_one, test_batches_one = val_epoch(X=X_test, y=y_test)
            test_acc += test_acc_one;
            test_cost += test_cost_one;
            test_topk_acc += test_topk_acc_one
            nb_test_batches += test_batches_one;

            if verbose:
                logger_train.debug("  this speaker results: ")
                logger_train.debug("\ttest cost:   %s", test_cost_one / test_batches_one)
                logger_train.debug("\vtest acc rate:  %s %%", test_acc_one / test_batches_one * 100)
                logger_train.debug("\vtest  top %s acc rate:  %s %%", k, test_topk_acc_one / test_batches_one * 100)

        # get the average over all speakers
        test_acc = test_acc / nb_test_batches * 100
        test_cost /= nb_test_batches
        test_topk_acc = test_topk_acc / nb_test_batches * 100
        return test_cost, test_acc, test_topk_acc


    def updateLR(LR, LR_decay, network_train_info, epochsNotImproved):
        this_cost = network_train_info['val_cost'][-1] #validation cost
        try:      last_cost = network_train_info['val_cost'][-2]
        except:   last_cost = 10 * this_cost  # first time it will fail because there is only 1 result stored

        # only reduce LR if not much improvment anymore
        if this_cost / float(last_cost) >= 0.99:
            logger_train.info(" Error not much reduced: %s vs %s. Reducing LR: %s", this_cost, last_cost, LR * LR_decay)
            epochsNotImproved += 1
            return LR * LR_decay, epochsNotImproved
        else:
            epochsNotImproved = max(epochsNotImproved - 1, 0)  # reduce by 1, minimum 0
            return LR, epochsNotImproved

    # try to load performance metrics of stored model
    # if os.path.exists(save_name + ".npz") and os.path.exists(save_name + "_trainInfo.pkl"):
    #     old_train_info = preprocessLipreading.unpickle(save_name + '_trainInfo.pkl')
    #     # backward compatibility
    #     if type(old_train_info) == list:
    #         old_train_info = old_train_info[0]
    #         best_val_acc = min(old_train_info[2])
    #         test_cost = min(old_train_info[3])
    #         test_acc = min(old_train_info[3])
    #     elif type(old_train_info) == dict:  # normal case
    #         best_val_acc = min(old_train_info['val_acc'])
    #         test_cost = min(old_train_info['test_cost'])
    #         test_acc = min(old_train_info['test_acc'])
    #         try:
    #             test_topk_acc = min(old_train_info['test_topk_acc'])
    #         except:
    #             test_topk_acc = 0
    #     else:
    #         best_val_acc = 0
    #         test_topk_acc = 0
    #         test_cost = 0
    #         test_acc = 0
    # else:
    best_val_acc = 0
    test_topk_acc = 0
    test_cost = 0
    test_acc = 0

    best_epoch = 1
    LR = LR_start
    # for storage of training info
    network_train_info = {
        'train_cost':[],
        'val_cost' :[], 'val_acc' : [], 'val_topk_acc' : [],
        'test_cost' : [], 'test_acc' : [], 'test_topk_acc' : []
    } #used to be list of lists
    epochsNotImproved = 0

    logger_train.info("starting training for %s epochs...", num_epochs)
    # now run through the epochs

# TODO: remove this
    if not loadPerSpeaker:  # all at once
        test_cost, test_acc, test_topk_acc, nb_test_batches = val_epoch(X_test, y_test)
        test_acc = test_acc / nb_test_batches * 100;
        test_cost /= nb_test_batches
        test_topk_acc = test_topk_acc / nb_test_batches * 100

    else:  # process each speaker seperately
        test_cost, test_acc, test_topk_acc = evalTEST(testSpeakerFiles,
                                                      sourceDataDir=database_binaryDir,
                                                      storeProcessed=storeProcessed,
                                                      processedDir=processedDir)
    logger_train.info("TEST results: ")
    logger_train.info("\t  test cost:        %s", test_cost)
    logger_train.info("\t  test acc rate:  %s %%", test_acc)
    logger_train.info("\t  test top %s acc:  %s %%", k, test_topk_acc)
# # TODO: end remove


    for epoch in range(num_epochs):
        logger_train.info("\n\n\n Epoch %s started", epoch + 1)
        start_time = time.time()

        if not loadPerSpeaker:
            total_train_cost, nb_train_batches = train_epoch(X=X_train, y=y_train, LR=LR); train_cost = total_train_cost / nb_train_batches
            X_train, y_train = shuffle(X_train, y_train)

            val_cost, val_acc, val_topk_acc, nb_val_batches = val_epoch(X=X_val, y=y_val)
            val_acc = val_acc / nb_val_batches * 100; val_cost /=nb_val_batches
            val_topk_acc /= nb_val_batches

        else:
            train_cost, val_cost, val_acc, val_topk_acc = evalTRAINING(trainingSpeakerFiles, LR, shuffleEnabled,
                        sourceDataDir=database_binaryDir,
                        storeProcessed=storeProcessed,
                        processedDir=processedDir)

        # test if validation acc went down
        printTest = False
        if val_acc > best_val_acc:
            printTest = True
            best_val_acc = val_acc
            best_epoch = epoch + 1

            logger_train.info("\n\nBest ever validation score; evaluating TEST set...")

            if not loadPerSpeaker:  # all at once
                test_cost, test_acc, test_topk_acc, nb_test_batches = val_epoch(X_test, y_test)
                test_acc = test_acc / nb_test_batches * 100;
                test_cost /= nb_test_batches
                test_topk_acc = test_topk_acc / nb_test_batches * 100

            else:  # process each speaker seperately
                test_cost, test_acc, test_topk_acc = evalTEST(testSpeakerFiles,
                                                              sourceDataDir=database_binaryDir,
                                                              storeProcessed=storeProcessed,
                                                              processedDir=processedDir)
            logger_train.info("TEST results: ")
            logger_train.info("\t  test cost:        %s", test_cost)
            logger_train.info("\t  test acc rate:  %s %%", test_acc)
            logger_train.info("\t  test top %s acc:  %s %%", k, test_topk_acc)

            if save_name is None:
                save_name = "./bestModel"
            if not os.path.exists(os.path.dirname(save_name)):
                os.makedirs(os.path.dirname(save_name))
            logger_train.info("saving model to %s", save_name)
            np.savez(save_name, *lasagne.layers.get_all_param_values(network_output_layer))

        epoch_duration = time.time() - start_time

        # Then we logger_train.info the results for this epoch:
        logger_train.info("Epoch %s of %s took %s seconds", epoch + 1, num_epochs, epoch_duration)
        logger_train.info("  LR:                            %s",    LR)
        logger_train.info("  training cost:                 %s",    train_cost)
        logger_train.info("  validation cost:               %s",    val_cost)
        logger_train.info("  validation acc rate:         %s %%", val_acc)
        logger_train.info("  validation top %s acc rate:         %s %%", k, val_topk_acc)
        logger_train.info("  best epoch:                    %s",    best_epoch)
        logger_train.info("  best validation acc rate:    %s %%", best_val_acc)
        if printTest:
            logger_train.info("  test cost:                 %s",    test_cost)
            logger_train.info("  test acc rate:           %s %%", test_acc)
            logger_train.info("  test top %s acc rate:    %s %%", k, test_topk_acc)

        # save the training info
        network_train_info['train_cost'].append(train_cost)
        network_train_info['val_cost'].append(val_cost)
        network_train_info['val_acc'].append(val_acc)
        network_train_info['val_topk_acc'].append(val_topk_acc)
        network_train_info['test_cost'].append(test_cost)
        network_train_info['test_acc'].append(test_acc)
        network_train_info['test_topk_acc'].append(test_topk_acc)

        store_path = save_name + '_trainInfo.pkl'
        general_tools.saveToPkl(store_path, network_train_info)
        logger_train.info("Train info written to:\t %s", store_path)

        # decay the LR
        #LR *= LR_decay
        LR, epochsNotImproved = updateLR(LR, LR_decay, network_train_info, epochsNotImproved)

        if epochsNotImproved > 8:
            logger_train.warning("\n\n NO MORE IMPROVEMENTS -> stop training")
            test_cost, test_acc, test_topk_acc = evalTEST(testSpeakerFiles,
                                                          sourceDataDir=database_binaryDir,
                                                          storeProcessed=storeProcessed,
                                                          processedDir=processedDir)

            logger_train.info("FINAL TEST results: ")
            logger_train.info("\t  test cost:        %s", test_cost)
            logger_train.info("\t  test acc rate:  %s %%", test_acc)
            logger_train.info("\t  test top %s acc:  %s %%", k, test_topk_acc)
            break

    logger_train.info("Done.")





