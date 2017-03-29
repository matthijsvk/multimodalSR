# load dataset



# split Train -> Val and Train
print('Separating validation and training set ...')
X_train = [];
X_val = []
y_train = [];
y_val = []
for i in range(len(X_train_all)):
    if i in val_idx:
        X_val.append(X_train_all[i])
        y_val.append(y_train_all[i])
    else:
        X_train.append(X_train_all[i])
        y_train.append(y_train_all[i])

if VERBOSE:
    print()
    print('Length of train, val, test')
    print(len(X_train))
    print(len(y_train))

    print(len(X_val))
    print(len(y_val))

    print(len(X_test))
    print(len(y_test))

if VERBOSE:
    print()
    print('Type of train')
    print(type(X_train))
    print(type(y_train))
    print(type(X_train[0]), X_train[0].shape)
    print(type(y_train[0]), y_train[0].shape)
