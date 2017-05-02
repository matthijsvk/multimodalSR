#
# class Box:
#     def area(self):
#         return self.width * self.height
#     def volume(self):
#         return self.area() * self.depth
#
#     def __init__(self, width, height, depth):
#         self.width = width
#         self.height = height
#         self. depth = depth
#
# # Create an instance of Box.
# x = Box(10, 2, 5)
#
# # Print area.
# print(x.area())
# print(x.volume())
#
#
# class networkToTrain:
#     def __init__(self,
#                  AUDIO_LSTM_HIDDEN_LIST=[256,256],
#                  CNN_NETWORK="google",
#                  cnn_features="conv",
#                  LIP_RNN_HIDDEN_LIST=[256,256],
#                  DENSE_HIDDEN_LIST=[128,64,64],
#                  datasetType="lipspeakers",
#                  runType="combined",
#                  LR_start=0.001):
#         self.AUDIO_LSTM_HIDDEN_LIST = AUDIO_LSTM_HIDDEN_LIST
#         self.CNN_NETWORK            = CNN_NETWORK
#         self.cnn_features           = cnn_features
#         self.LIP_RNN_HIDDEN_LIST    = LIP_RNN_HIDDEN_LIST
#         self.DENSE_HIDDEN_LIST      = DENSE_HIDDEN_LIST
#         self.datasetType            = datasetType
#         self.runType                = runType
#         self.LR_start               = LR_start
#
#
# # Create an instance of Box.
# a = networkToTrain(cnn_features="dense", LIP_RNN_HIDDEN_LIST=None, DENSE_HIDDEN_LIST=[128, 64, 64])
#
# from pprint import pprint
# pprint(vars(a))

dense_hidden_list=[]
combinedNet= []
for i in range(len(dense_hidden_list)):
    n_hidden = dense_hidden_list[i]
    print(n_hidden)
    print("soatiernsotai")
print(len(combinedNet))