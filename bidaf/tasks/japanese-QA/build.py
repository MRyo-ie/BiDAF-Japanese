"""Downloads SQuAD train and dev sets, preprocesses and writes tokenized versions to file"""

import os
import json
from .preprocess import PreprocessSQuAD


# 実行した pythonスクリプトのディレクトリの絶対パス
base_dir = os.path.dirname(__file__)


def data_from_json(filename):
    """Loads JSON data from filename and returns"""
    with open(filename) as data_file:
        data = json.load(data_file)
    return data


def data_download_and_preprocess(squad_version=1.1, do_lowercase=True):
    # SQuAD クラスをインスタンス化
    preSQuAD = PreprocessSQuAD()

    data_dir = os.path.join(base_dir, 'tmp')
    print('[確認] 出力ディレクトリ： ', data_dir)

    print("Will download SQuAD datasets to {} if required".format(data_dir))
    print("Will put preprocessed SQuAD datasets in {}".format(data_dir))

    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    train_filename = "train-v{}.json".format(squad_version)
    dev_filename = "dev-v{}.json".format(squad_version)

    # download train set
    preSQuAD.maybe_download(preSQuAD.DOWNLOAD_DATA_URL, train_filename, data_dir)

    # read train set
    train_data = data_from_json(os.path.join(data_dir, train_filename))
    print("Train data has %i examples total" % preSQuAD.total_examples(train_data))

    # preprocess train set and write to file
    if not os.path.isfile(os.path.join(data_dir, 'train-v{}.context'.format(squad_version))):
        print("Preprocessing training data")
        preSQuAD.preprocess_and_write(train_data, 'train', data_dir, squad_version, do_lowercase=do_lowercase)
    print("Train data preprocessed!")

    # download dev set
    preSQuAD.maybe_download(preSQuAD.DOWNLOAD_DATA_URL, dev_filename, data_dir)

    # read dev set
    dev_data = data_from_json(os.path.join(data_dir, dev_filename))
    print("Dev data has %i examples total" % preSQuAD.total_examples(dev_data))

    # preprocess dev set and write to file
    if not os.path.isfile(os.path.join(data_dir, 'dev-v{}.context'.format(squad_version))):
        print("Preprocessing development data")
        preSQuAD.preprocess_and_write(dev_data, 'dev', data_dir, squad_version, do_lowercase=do_lowercase)
    print("Dev data preprocessed!")
