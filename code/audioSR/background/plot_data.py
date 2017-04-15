import os

import matplotlib.pyplot as plt

from general_tools import *

store_dir = os.path.expanduser('~/TCDTIMIT/audioSR/TIMIT/binary39/TIMIT')
dataset_path = os.path.join(store_dir, 'TIMIT_39_ch.pkl')

X_train, y_train, valid_frames_train, X_val, y_val, valid_frames_val, X_test, y_test, valid_frames_test = load_dataset(dataset_path)

plt.figure(1)
plt.title('Preprocessed data visualization')
for i in range(1, 5):
    plt.subplot(2, 2, i)
    plt.axis('off')
    plt.imshow(X_train[i].T)
plt.imshow(np.log(X_train[i].T))
print(X_train[i].shape)

plt.tight_layout()
plt.show()
