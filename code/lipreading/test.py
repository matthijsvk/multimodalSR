from general_tools import *

save_dir= '/home/matthijs/TCDTIMIT/lipreading/TCDTIMIT/results/CNN_binaryNet/'
save_name = save_dir + 'lipspeakers_google_phoneme39_binary'
old_train_info = unpickle(save_name + '_trainInfo.pkl')
if type(old_train_info) == dict:  # normal case
    print(old_train_info.keys())
    best_val_acc = max(old_train_info['val_acc'])
    test_cost = min(old_train_info['test_cost'])
    test_acc = max(old_train_info['test_acc'])
    test_topk_acc = max(old_train_info['test_topk_acc'])
    network_train_info = old_train_info  # load old train info so it won't get lost on retrain

    if not 'final_test_cost' in network_train_info.keys():
        network_train_info['final_test_cost'] = min(network_train_info['test_cost'])
    if not 'final_test_acc' in network_train_info.keys():
        network_train_info['final_test_acc'] = max(network_train_info['test_acc'])
    if not 'final_test_top3_acc' in network_train_info.keys():
        network_train_info['final_test_top3_acc'] = max(network_train_info['test_topk_acc'])