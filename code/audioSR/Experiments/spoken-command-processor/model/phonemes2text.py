import numpy as np
from keras.layers import Activation, Dense, Dropout
from keras.models import Sequential, model_from_json
from keras.optimizers import Adam

from . import dataset


def train(summarize=False, data_limit=None):
    X_train, y_train = dataset.Phonemes2Text().load_train_data(limit=data_limit)

    # Number of phonemes per word (sample)
    input_dim = X_train.shape[1]
    # Number of distinct classes in the dataset (number of distinct words)
    output_dim = y_train.shape[1]
    # Arbitrary parameter. For 20 epochs...
    # 256 --> 32.7% accuracy
    # 500 --> 46.0% accuracy
    # 1500 --> 49.5% accuracy
    hidden_num = 1500

    # Architecture of the model
    model = Sequential()

    model.add(Dense(input_dim=input_dim, output_dim=hidden_num))
    model.add(Activation('sigmoid'))
    model.add(Dropout(0.25))
    model.add(Dense(output_dim=output_dim))
    model.add(Activation('sigmoid'))
    model.add(Dropout(0.40))
    model.add(Dense(output_dim=output_dim))
    model.add(Activation('softmax'))

    model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=0.001))

    stats = model.fit(X_train, y_train,
        shuffle=True,
        batch_size=256,
        nb_epoch=50,
        verbose=1
    )

    save_model(model)

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
    X_test, y_test = dataset.Phonemes2Text().load_test_data()

    out = model.predict_classes(X_test,
        batch_size=256,
        verbose=0
    )

    acc = sum(out == np.argmax(y_test, axis=1)) * 1.0 / len(out)
    print('Accuracy using %d testing samples: %f' % (X_test.shape[0], acc))

def predict(X_test):
    model = load_model()

    predicted_classes = model.predict_classes(X_test,
        batch_size=256,
        verbose=0
    )

    class_numbers = dataset.TIMITReader().load_unique_words_as_class_numbers()
    return [k for k,v in class_numbers.items() if v in predicted_classes]

def save_model(model):
    reader = dataset.Phonemes2Text()

    with open(reader.params('phonemes2text_arch', 'json'), 'w') as archf:
        archf.write(model.to_json())

    model.save_weights(
        filepath=reader.params('phonemes2text_weights', 'h5'),
        overwrite=True
    )

def load_model():
    reader = dataset.Phonemes2Text()

    with open(reader.params('phonemes2text_arch', 'json')) as arch:
        model = model_from_json(arch.read())
        model.load_weights(reader.params('phonemes2text_weights', 'h5'))

        model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=0.001))
        return model
