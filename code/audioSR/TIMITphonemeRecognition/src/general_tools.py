from six.moves import cPickle


def path_reader(filename):
    with open(filename) as f:
        path_list = f.read().splitlines()
    return path_list


def load_dataset(file_path):
    with open(file_path, 'rb') as cPickle_file:
        [X_train, y_train, X_val, y_val, X_test, y_test] = cPickle.load(cPickle_file)
    if not X_train:
        print('WARNING: X_train is empty')
    if not y_train:
        print('WARNING:  y_train is empty')
    if not X_val:
        print('WARNING: X_val is empty')
    if not y_val:
        print('WARNING: y_val is empty')
    if not X_test:
        print('WARNING: X_test is empty')
    if not y_test:
        print('WARNING: y_test is empty')
    return X_train, y_train, X_val, y_val, X_test, y_test
