from bidaf import BidirectionalAttentionFlow
from bidaf.scripts import load_data_generators

import argparse
import sys

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('-sv', '--squad_version', choices=[1.1, 2.0], type=float,
                    action='store', default=1.1, help='SQuAD dataset version')

# Train を実行する。
if __name__ == "__main__":
    args = parser.parse_args()
    bidaf_model = BidirectionalAttentionFlow(400)
    #bidaf_model.load_bidaf("bidaf/data/tmp/bidaf_06.h5") # when you want to resume training
    train_generator, validation_generator = load_data_generators(8, 400, squad_version=args.squad_version, div_epoch_num=20)
    """
    引数
    ・initial_epoch ： epoch の開始番号（続きからやる場合は、bidaf_n.h5 の n を入れる。）
    """
    keras_model = bidaf_model.train_model(train_generator, validation_generator=validation_generator
                                        , workers=1, use_multiprocessing=True, initial_epoch=0, epochs=10)
