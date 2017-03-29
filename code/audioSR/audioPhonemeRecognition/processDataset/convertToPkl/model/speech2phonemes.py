import numpy as np
from keras.layers import Activation, Dense
from keras.models import Sequential, model_from_json
from keras.optimizers import Adam

import dataset


def train(summarize=False, data_limit=None):
    print('test1')
    X_train, y_train, y_train_onehot = dataset.Speech2Phonemes().load_train_data(limit=data_limit)

    print("length of training data: ", len(X_train))

    # Number of features for each sample in X_train...
    # if each 20ms corresponds to 13 MFCC coefficients + delta + delta2, then 39
    input_dim = X_train.shape[1]
    # Number of distinct classes in the dataset (number of distinct phonemes)
    output_dim = np.max(y_train) + 1
    # Model takes as input arrays of shape (*, input_dim) and outputs arrays
    # of shape (*, hidden_num)
    hidden_num = 256

    print("1")

    # Architecture of the model
    model = Sequential()

    print("2")

    model.add(Dense(input_dim=input_dim, output_dim=hidden_num))
    print("2b")
    model.add(Activation('sigmoid'))
    print("2c")
    # model.add(Dropout(0.25))
    print("2d")
    model.add(Dense(output_dim=output_dim))
    model.add(Activation('softmax'))

    print("3")

    model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=0.001))

    stats = model.fit(X_train, y_train_onehot,
                      shuffle=True,
                      batch_size=256,
                      nb_epoch=200,
                      verbose=1
                      )

    print("4")

    save_model(model)

    print("5")

    if summarize:
        print(model.summary())

        import matplotlib.pyplot as plt
        plt.plot(stats.history['loss'])
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.title('Loss function for %d samples' % X_train.shape[0])
        plt.show()


def test(data_limit=None):
    model = load_model()
    X_test, y_test = dataset.Speech2Phonemes().load_test_data()

    out = model.predict_classes(X_test,
                                batch_size=256,
                                verbose=0
                                )

    acc = sum(out == y_test) * 1.0 / len(out)
    print('Accuracy using %d testing samples: %f' % (X_test.shape[0], acc))


def predict(X_test):
    model = load_model()

    return model.predict_classes(X_test,
                                 batch_size=256,
                                 verbose=0
                                 )


def save_model(model):
    reader = dataset.Speech2Phonemes()

    with open(reader.params('speech2phonemes_arch', 'json'), 'w') as archf:
        archf.write(model.to_json())

    model.save_weights(
            filepath=reader.params('speech2phonemes_weights', 'h5'),
            overwrite=True
    )


def load_model():
    reader = dataset.Speech2Phonemes()

    with open(reader.params('speech2phonemes_arch', 'json')) as arch:
        model = model_from_json(arch.read())
        model.load_weights(reader.params('speech2phonemes_weights', 'h5'))

        model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=0.001))
        return model


if __name__ == "__main__":
    train()
