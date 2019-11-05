from bidaf import BidirectionalAttentionFlow
from bidaf.mode_predict import predict_ans
from _settings.config_path_utils import ConfigPathUtils
import os

import argparse
parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('mode', choices=['squad', 'SQuAD', 'ja_qa', 'ja_QA'], type=str,
                    help='どのタスクを解かせるかを指定する。')
parser.add_argument('-sv', '--squad_version', choices=[1.1, 2.0], type=float,
                    default=2.0, help='SQuAD dataset version')
parser.add_argument('--model', type=str, help='予測に使うmodelのパス。初期値：bidaf/Drive/weights/bidaf_50.h5',
                    default='bidaf/Drive/weights/bidaf_50.h5')


if __name__ == "__main__":
    args = parser.parse_args()
    mode = 'squad'
    if args.mode in ['ja_qa', 'ja_QA']:
        mode = 'ja_qa'
    w_path = args.model

    # 初期化
    bidaf_model = BidirectionalAttentionFlow(400)
    if not os.path.exists(w_path):
        raise Exception('model ファイルが見つかりませんでした。\nSQuAD のデモの場合は、https://drive.google.com/open?id=10C56f1DSkWbkBBhokJ9szXM44P9T-KfW からダウンロードしてください。')
    # Drive系のパスを読み込み： _settings/SavePaths.cfg  から デフォルト設定を読み込み
    cfg_pb_Drive = ConfigPathUtils('Drive', os.path.dirname(os.path.abspath(__file__)))
    wordDB_dir = cfg_pb_Drive.get_path('wordDB_dir_path')

    print('\nWarming up を開始します！')
    print(predict_ans(bidaf_model, mode, wordDB_dir,
            "This is a tree", "What is this?"))
    #=> {'answer': 'tree'}

    print(predict_ans(bidaf_model, mode, wordDB_dir,
            "This is a tree which highest in my country.", "What is this?"))
    #=> {'answer': 'tree which highest in my country'}

    # 対話モード
    while True:
        print('\n\n')
        sentence = input('\n問題文 を入力（改行はしないでね！）\n>>  ')
        query = input('\n質問文 を入力\n>>  ')

        print('')
        ans = predict_ans(bidaf_model, mode, wordDB_dir, sentence, query)
        print('\n\n回答： {}'.format(ans['answer']))

