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



# Train を実行。
if __name__ == "__main__":
    args = parser.parse_args()
    # Drive のパスを読み込み： _settings/SavePaths.cfg  から デフォルト設定を読み込み
    cfg_path_builder = ConfigPathUtils('Drive')
    saved_wordDB_path = cfg_path_builder.get_path('wordDB_dir_path')
    saved_weight_path = cfg_path_builder.get_path('weights_dir_path')
    # log系のパスを読み込み： _settings/SavePaths.cfg  から デフォルト設定を読み込み
    cfg_path_builder = ConfigPathUtils('Log')
    log_tboard_path = cfg_path_builder.get_path('tensorboard_dir_path')
    # パスをひとまとめにする
    train_paths = {
        'Drive_wordDB': saved_wordDB_path,
        'Drive_weight': saved_weight_path,
        'Log_base_dir' : cfg_path_builder.base_path,
        'Log_tboard' : log_tboard_path,
    }
    # task系のパスを読み込み
    cfg_path_builder = ConfigPathUtils('Drive')


    ### モデル初期化
    bidaf_model = BidirectionalAttentionFlow(400)
    # 続きから学習　の準備
    init_epoch = 0
    if args.continue_epoch_num > 0:
        init_epoch = args.continue_epoch_num
        #print('     initial_epoch : ', init_epoch)
        bidaf_model.load_bidaf("{}/bidaf_{:02}.h5".format(saved_weight_path, init_epoch)) # when you want to resume training
    train_generator, validation_generator = load_data_generators(8, 400, squad_version=args.squad_version, div_epoch_num=20)
    ## 学習実行
    keras_model = train_model(bidaf_model, train_paths, train_generator, validation_generator=validation_generator
                                        , workers=1, use_multiprocessing=True, initial_epoch=init_epoch, epochs=10)
