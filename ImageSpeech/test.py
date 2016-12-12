def convertToCifar10Files (rootDir, targetDir):
    from PIL import Image
    import numpy as np
    
    out = np.empty([20, 7501])
    for j in xrange(0, 10):
        im = Image.open('%d_receipt.jpg' % j)
        im = (np.array(im))
    import numpy as np
    import scipy.io
    
    mat = scipy.io.loadmat('train_32x32.mat')
    data = mat['X']
    label = mat['y']
    
    R_data = data[:, :, 0, :]
    G_data = data[:, :, 1, :]
    B_data = data[:, :, 2, :]
    
    R_data = np.transpose(R_data, (2, 0, 1))
    G_data = np.transpose(G_data, (2, 0, 1))
    B_data = np.transpose(B_data, (2, 0, 1))
    
    R_data = np.reshape(R_data, (73257, 32 * 32))
    G_data = np.reshape(G_data, (73257, 32 * 32))
    B_data = np.reshape(B_data, (73257, 32 * 32))
    
    outdata = np.concatenate((label, R_data, G_data, B_data), axis=1)
    step = 10000
    for i in range(1, 6):
        temp = outdata[i * step:(i + 1) * step, :]
        temp.tofile('SVHN_train_data_batch%d.bin' % i)
        print('save data %d' % i)