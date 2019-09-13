# bidaf-keras/bidaf/__main__.py  から、SQuADのダウンロード部分を切り出した。
from bidaf.tasks.squad import data_download_and_preprocess

import argparse
import sys
parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('-sv', '--squad_version', choices=[1.1, 2.0], type=float,
                    action='store', default=1.1, help='SQuAD dataset version')
parser.add_argument('-l', '--do_lowercase', action='store_true', default=False, help='Convert input to lowercase')



def main():
    # if len(sys.argv) == 1:
    #     parser.print_help(sys.stderr)
    #     sys.exit()
    args = parser.parse_args()
    #print('\nargs.squad_version : {}\n\n'.format(args.squad_version))
    data_download_and_preprocess(squad_version=args.squad_version, do_lowercase=args.do_lowercase)



if __name__ == '__main__':
    main()


