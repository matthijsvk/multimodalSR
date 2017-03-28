from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import pickle
import argparse

from tqdm import tqdm

from utils import load_wavPhn, load_wav, process_data, process_raw_phn
from train import train_model
from transform import transform_wav, copy_phn


class Env:
    pass

ENV = Env()


def train(ENV, args):
    processed_train_data_path = os.path.join(ENV.processed_data_path, 'processed_train.pkl')
    processed_test_data_path = os.path.join(ENV.processed_data_path, 'processed_test.pkl')
    if os.path.exists(processed_train_data_path) and os.path.exists(processed_test_data_path):
        processed_train_data = pickle.load(open(processed_train_data_path, 'r'))
        processed_test_data = pickle.load(open(processed_test_data_path, 'r'))
    else:
        print('Process train data...')
        train_wav_files, train_phn_files = load_wavPhn(ENV.train_data)
        processed_train_data = process_data(train_wav_files, train_phn_files)
        pickle.dump(processed_train_data, open(processed_train_data_path, 'w'))

        print('Process test data...')
        test_wav_files, test_phn_files = load_wavPhn(ENV.test_data)
        processed_test_data = process_data(test_wav_files, test_phn_files)
        pickle.dump(processed_test_data, open(processed_test_data_path, 'w'))

    # print(processed_train_data[0][1])
    print("Define graph...")
    train_model(ENV, processed_train_data, processed_test_data)


def decode(ENV, args):
    train_model(ENV, decode=True, file_decode=args.file_decode)


def transform(ENV, args):
    train_wav_files, train_phn_files = load_wavPhn(ENV.train_data)
    test_wav_files = load_wav(ENV.test_data)
    train_output_path = os.path.join(ENV.output, 'train')
    test_output_path = os.path.join(ENV.output, 'test')
    if not os.path.exists(train_output_path):
        os.makedirs(train_output_path)
    if not os.path.exists(test_output_path):
        os.makedirs(test_output_path)

    for i in tqdm(range(len(train_wav_files))):
        transform_wav(train_wav_files[i], train_output_path)
        phn_file = os.path.join(os.path.dirname(train_wav_files[i]), 
                                os.path.basename(train_wav_files[i]).split('.')[0] + '.phn')
        copy_phn(phn_file, train_output_path)

    for i in tqdm(range(len(test_wav_files))):
        transform_wav(test_wav_files[i], test_output_path)
        phn_file = os.path.join(os.path.dirname(test_wav_files[i]), 
                                os.path.basename(test_wav_files[i]).split('.')[0] + '.phn')
        copy_phn(phn_file, test_output_path)


def phoneme(ENV, args):
    while True:
        phn_file = raw_input('Enter the path for phn file:')
        phn_list = process_raw_phn(phn_file)
        print(phn_list)

def setup_env(args):
    if 'output_path' in args and not os.path.exists(args.output_path):
        os.makedirs(args.output_path)
    if 'train_data_path' in args and not os.path.exists(args.train_data_path):
        raise Exception('Can not find train data path')
    if 'test_data_path' in args and not os.path.exists(args.test_data_path):
        raise Exception('Can not find test data path')
    ENV.proj_path = os.path.dirname(os.path.abspath(__file__))
    if 'output_path' in args:
        ENV.output = args.output_path
    if 'train_data_path' in args:
        ENV.train_data = args.train_data_path
    if 'test_data_path' in args:
        ENV.test_data = args.test_data_path
    if 'model_path' in args:
        ENV.model_path = args.model_path
    if 'processed_data_path' in args:
        if not os.path.exists(args.processed_data_path):
            os.makedirs(args.processed_data_path)
        ENV.processed_data_path = args.processed_data_path


def prepare_parser():
    parser = argparse.ArgumentParser()
    sub_parsers = parser.add_subparsers()

    ## TRAIN ##
    train_parser = sub_parsers.add_parser('train')
    train_parser.add_argument('-n', '--train_data_path',
                              help="the directory store training data",
                              required=True)
    train_parser.add_argument('-t','--test_data_path',
                              help="the directory store test data",
                              required=True)
    train_parser.add_argument('-o','--output_path',
                              help="the directory store output data",
                              default='./out/')
    train_parser.add_argument('-p','--processed_data_path',
                              help="the path store processed data",
                              default='./processed_data/')
    train_parser.set_defaults(func=train)


    ## DECODE ##
    decode_parser = sub_parsers.add_parser('decode')
    decode_parser.add_argument('-m', '--model_path',
                               help="the directory store the model") 
    decode_parser.add_argument('-f', '--file_decode',
                               action='store_true', help='decode from file')
    decode_parser.set_defaults(func=decode)


    ## TRANSFORM ##
    transform_parser = sub_parsers.add_parser('transform')
    transform_parser.add_argument('-n', '--train_data_path',
                              help="the directory store training data",
                              required=True)
    transform_parser.add_argument('-t','--test_data_path',
                              help="the directory store test data",
                              required=True)
    transform_parser.add_argument('-o','--output_path',
                              help="the directory store output data",
                              required=True)
    transform_parser.set_defaults(func=transform)

    ## PHONEME ##
    phoneme_parser = sub_parsers.add_parser('phoneme')
    phoneme_parser.set_defaults(func=phoneme)
    return parser


if __name__ == '__main__':
    arg_parser = prepare_parser()
    args = arg_parser.parse_args(sys.argv[1:])
    setup_env(args)
    args.func(ENV, args)
