import os
from bidaf import BidirectionalAttentionFlow
from bidaf.mode_train import train_model
from bidaf.scripts import load_data_generators
from _settings.config_path_utils import ConfigPathUtils

#import sys

import argparse
parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('mode', choices=['squad', 'SQuAD', 'ja_qa', 'ja_QA'], type=str,
                    help='どのタスクを解かせるかを指定する。')
parser.add_argument('-sv', '--squad_version', choices=[1.1, 2.0], type=float,
                    default=2.0, help='SQuAD dataset version')
parser.add_argument('-c', '--continue_epoch_num', type=int,
                    action='store', default=0, help='学習を続きから始める')

parser.add_argument('-b', '--batch_size', type=int,
                    action='store', default=8, help='バッチサイズを指定（8GBで7くらい、11GBで15くらい。）')
parser.add_argument('-div', '--divide_epoch', type=int,
                    action='store', default=50, help='1epoch をなん分割するかを指定（1epoch が長すぎるため。50くらい？）')


# Train を実行。
if __name__ == "__main__":
    args = parser.parse_args()
    task = args.mode.lower()
    # 現在のディレクトリ
    root_dir = os.path.dirname(os.path.abspath(__file__))
    # log系のパスを読み込み： _settings/SavePaths.cfg  から デフォルト設定を読み込み
    cfg_pb_Log = ConfigPathUtils('Log', root_dir)
    # Drive系のパスを読み込み： _settings/SavePaths.cfg  から デフォルト設定を読み込み
    cfg_pb_Drive = ConfigPathUtils('Drive', root_dir)
    # task系のパスを読み込み
    cfg_pb_base = ConfigPathUtils(task, root_dir)
    # パスをひとまとめにする
    train_paths = {
        'base_dir' : cfg_pb_base.get_path('data_path'),
        'Drive_wordDB': cfg_pb_Drive.get_path('wordDB_dir_path'),
        'Drive_weight': cfg_pb_Drive.get_path('weights_dir_path'),
        'Log_base_dir' : cfg_pb_Log.base_path,
        'Log_tboard' : cfg_pb_Log.get_path('tensorboard_dir_path'),
    }


    ### モデル初期化
    bidaf_model = BidirectionalAttentionFlow(400)
    # 続きから学習　の準備
    init_epoch = 0
    if args.continue_epoch_num > 0:
        init_epoch = args.continue_epoch_num
        #print('     initial_epoch : ', init_epoch)
        bidaf_model.load_bidaf("{}/bidaf_{:02}.h5".format(train_paths['Drive_weight'], init_epoch)) # when you want to resume training
    train_generator, validation_generator = load_data_generators(train_paths, args.batch_size, 400,
                                                                squad_version=args.squad_version,
                                                                div_epoch_num=args.divide_epoch)
    ## 学習実行
    keras_model = train_model(bidaf_model, train_paths, train_generator, validation_generator=validation_generator
                                        , workers=1, use_multiprocessing=True, initial_epoch=init_epoch, epochs=10)
