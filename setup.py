# bidaf-keras/bidaf/__main__.py  から、SQuADのダウンロード部分を切り出した。
import sys
from _settings.config_path_utils import ConfigPathUtils
from bidaf.tasks.build import build, OptionValues

import argparse
parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('mode', choices=['squad', 'SQuAD', 'ja_qa', 'ja_QA'], type=str,
                    help='どのタスクを解かせるかを指定する。')
parser.add_argument('-sv', '--squad_version', choices=[1.1, 2.0], type=float,
                    default=2.0, help='SQuAD dataset version')
parser.add_argument('-l', '--do_lowercase', action='store_true', default=False, help='Convert input to lowercase')


def main():
    args = parser.parse_args()
    task = args.mode.lower()
    # タスクデータ のパスを読み込み： _settings/SavePaths.cfg  から デフォルト設定を読み込み
    cfg_path_builder = ConfigPathUtils(task)
    task_data_path = cfg_path_builder.get_path('data_path')

    opts = OptionValues(task_data_path, args.squad_version, args.do_lowercase)
    build(task, opts)



if __name__ == '__main__':
    main()


