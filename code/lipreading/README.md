The files in this directory can be used to train a CNN, for example to learn to predict visemes/phoneme from images of a speaking persons mouth
Performance of these networks can improved by combining outputs of the CNN over several timesteps using LSTM networks. TO do that, you need to bundle the frames per video, which makes is quite hard to use batching efficiently.
If you want to train such a network, first train your CNN here, and then go to ../combinedSR where you can specify to train only the lipreading part, but add LSTM networks on top.

1. Extract the images and labelfiles from the TCDTIMIT dataset: see [TCDTIMITprocessing](https://github.com/matthijsvk/TCDTIMITprocessing)
    - after running `TCDTIMITprocessing/main.y`, extract the images and labels from `processed` dir to `database` dir using `lipreading/fileDirOps`.  
    - it also generates pkl files; you can specify to generate either phoneme or viseme-type pickle files
1. Set your desired parameters in lipreading.py. (Learning Rate, batch size etc). You can choose the network as well:  
    - network type: (see [Lasagne examples](https://github.com/Lasagne/Recipes/tree/master/modelzoo), here defined in `buildNetwork.py`)  
        1. Network from ChungSVZ16 ([this paper](https://arxiv.org/abs/1611.05358))  
        2. custom CNN network, based on CIFAR10 Lasagne example  
        3. ResNet50 network (Kaiming He et al. 2016 (https://arxiv.org/abs/1603.05027)  
    - recognize on phonemes (39 classes) or visemes (12 classes)