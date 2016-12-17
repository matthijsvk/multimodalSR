from __future__ import print_function

import numpy as np
import theano
import theano.tensor as T
import lasagne

try:
    input_text = open("shakespeare_input.txt", "r").read()
    input_text = input_text.decode("utf-8-sig").encode("utf-8")
except Exception as e:
    raise IOError("Couldn't read input file")

#Based on training input, predict what follows:
generation_phrase = "First Citizen:\nBefore we proceed any further, hear me speak."

vocabulary = list(set(input_text))
input_size = len(input_text)
vocabulary_size = len(vocabulary)

character_to_ix = {char:i for i, char in enumerate(vocabulary)}
ix_to_character = {i:char for i, char in enumerate(vocabulary)}

lasagne.random.set_rng(np.random.RandomState(1))

#Constants. Constants everywhere.
SEQUENCE_SIZE = 20
HIDDEN_SIZE = 512 #Amount of units in the two LSTM layers
LEARNING_RATE = 0.01
GRADIENT_CLAMP = 100 #Remove gradients above this number.
PRINT_INTERVAL = 1 #How often to check output.
EPOCHS = 50 #Number of epochs to train network.
BATCH_SIZE = 128

def generate_data(p, batch_size=BATCH_SIZE, data=input_text, pass_target=True):
    x = np.zeros((batch_size, SEQUENCE_SIZE, vocabulary_size))
    y = np.zeros(batch_size)

    for n in range(batch_size):
        pointer = n
        for i in range(SEQUENCE_SIZE):
            x[n, i, character_to_ix[data[p + pointer + i]]] = 1
        if pass_target:
            y[n] = character_to_ix[data[p + pointer + SEQUENCE_SIZE]]
    return x, np.array(y, dtype="int32")

def main(epochs=EPOCHS):
    print("Now building network ...")
    #Build the network, starting at input layer.
    #Recurrent layers need input of shape:
        #(batch_size, SEQUENCE_SIZE, number of feature)
    layer_input = lasagne.layers.InputLayer(shape=(None, None, vocabulary_size))
    #Build Long Short Term Memory layer taking "layer_input" as first input.
    #Clamp the gradient to avoid the problem of exploding gradients.
        #Clamping/Clipping is defined by the "GRADIENT_CLAMP" ...
    layer_forward_01 = lasagne.layers.LSTMLayer(
        layer_input, HIDDEN_SIZE, grad_clipping=GRADIENT_CLAMP,
        nonlinearity=lasagne.nonlinearities.tanh)

    layer_forward_02 = lasagne.layers.LSTMLayer(
        layer_forward_01, HIDDEN_SIZE, grad_clipping=GRADIENT_CLAMP,
        nonlinearity=lasagne.nonlinearities.tanh)
    #The layer_forward creates output of dimension:
        #(batch_size, SEQUENCE_SIZE, HIDDEN_SIZE)
    #We care only about the final prediction, so we
        #isolate that quantity and feed it to the next layer.
    #Output of the sliced layer will be of dimension:
        #(batch_size, vocabulary_size)
    layer_forward_slice = lasagne.layers.SliceLayer(layer_forward_02, -1, 1)
    #The sliced output is parsed through softmax function to create
        #probability distribution
    layer_output = lasagne.layers.DenseLayer(
        layer_forward_slice, num_units=vocabulary_size,
        W=lasagne.init.Normal(), nonlinearity=lasagne.nonlinearities.softmax)
    target_values = T.ivector("target_output")
    network_output = lasagne.layers.get_output(layer_output)

    cost = T.nnet.categorical_crossentropy(network_output, target_values).mean()
    #Recieve all parameters from the network.
    all_parameters = lasagne.layers.get_all_params(layer_output, trainable=True)
    #Compute AdaGrad updates for training.
    print("Computing updates ...")
    updates = lasagne.updates.adagrad(cost, all_parameters, LEARNING_RATE)

    print("Compiling functions ...")
    train = theano.function([layer_input.input_var, target_values], cost, updates=updates, allow_input_downcast=True)
    compute_cost = theano.function([layer_input.input_var, target_values], cost, allow_input_downcast=True)

    probs = theano.function([layer_input.input_var], network_output, allow_input_downcast=True)

    def try_stuff(N=200):
        assert(len(generation_phrase)>=SEQUENCE_SIZE)
        sample_ix = []
        x,_ = generate_data(len(generation_phrase) - SEQUENCE_SIZE, 1, generation_phrase,0)
        for i in range(N):
            # Pick the character that got assigned the highest probability
            ix = np.argmax(probs(x).ravel())
            # Alternatively, to sample from the distribution instead:
            # ix = np.random.choice(np.arange(vocab_size), p=probs(x).ravel())
            sample_ix.append(ix)
            x[:, 0:SEQUENCE_SIZE - 1,:] = x[:, 1:, :]
            x[:, SEQUENCE_SIZE - 1,:] = 0
            x[0, SEQUENCE_SIZE - 1, sample_ix[-1]] = 1.

        random_snippet = generation_phrase + "".join(ix_to_character[ix] for ix in sample_ix)
        print("----\n %s \n----" % random_snippet)
    print("Training ...")
    print("Seed for generation is: %s" % generation_phrase)
    p = 0
    try:
        for it in xrange(input_size * epochs / BATCH_SIZE):
            try_stuff() #Generate text using p^th character as the start.

            average_cost = 0;
            for _ in range(PRINT_INTERVAL):
                x, y = generate_data(p)

                p += SEQUENCE_SIZE + BATCH_SIZE - 1

                if(p + BATCH_SIZE + SEQUENCE_SIZE >= input_size):
                    print("Carriage return")
                    p = 0
                average_cost += train(x, y)
            print("Epoch {} average loss = {}".format(it * 1.0 * PRINT_INTERVAL / input_size * BATCH_SIZE, average_cost / PRINT_INTERVAL))
    except KeyboardInterrupt:
        pass

if __name__ == "__main__":
    main()
