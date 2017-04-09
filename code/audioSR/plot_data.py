import os

import matplotlib.pyplot as plt

from general_tools import *

store_dir = '/home/matthijs//home/matthijs/TCDTIMIT/TIMIT/binary_list39/speech2phonemes26Mels
dataset_path = os.path.join(paths[0], 'std_preprocess_26_ch.pkl')

X_train, y_train, X_val, y_val, X_test, y_test = load_dataset(dataset_path)

plt.figure(1)
plt.title('Preprocessed data visualization')
for i in range(1, 5):
    plt.subplot(2, 2, i)
    plt.axis('off')
    plt.imshow(X_train[i].T)
# plt.imshow(np.log(X_train[i].T))
# print(X_train[i].shape)

plt.tight_layout()
plt.show()
