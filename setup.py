# bidaf-keras/bidaf/__main__.py  から、SQuADのダウンロード部分を切り出した。
import sys

import argparse
parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('mode', choices=['squad', 'SQuAD', 'ja_qa', 'ja_QA'], type=str,help='どのタスクを解かせるかを指定する。')
parser.add_argument('-sv', '--squad_version', choices=[1.1, 2.0], type=float,
                    default=2.0, help='SQuAD dataset version')
parser.add_argument('-l', '--do_lowercase', action='store_true', default=False, help='Convert input to lowercase')


def main():
    # if len(sys.argv) == 1:
    #     parser.print_help(sys.stderr)
    #     sys.exit()
    args = parser.parse_args()
    if args.mode in ['squad', 'SQuAD']:
        from bidaf.tasks.squad import build_squad
        build_squad(squad_version=args.squad_version, do_lowercase=args.do_lowercase)
    elif args.mode in ['ja_qa', 'ja_QA']:
        from bidaf.tasks.ja_QA import build_ja_QA
        build_ja_QA()



if __name__ == '__main__':
    main()


