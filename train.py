from bidaf import BidirectionalAttentionFlow
from bidaf.mode_train import train_model
from bidaf.scripts import load_data_generators

import argparse
import sys

# argparse
parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('-sv', '--squad_version', choices=[1.1, 2.0], type=float,
                    action='store', default=1.1, help='SQuAD dataset version')
parser.add_argument('-c', '--continue_epoch_num', type=int,
                    action='store', default=0, help='学習を続きから始める')

# Train を実行。
if __name__ == "__main__":
    args = parser.parse_args()
    bidaf_model = BidirectionalAttentionFlow(400)
    # 続きから学習　の準備
    init_epoch = 0
    if args.continue_epoch_num > 0:
        init_epoch = args.continue_epoch_num
        print('     initial_epoch : ', init_epoch)
        bidaf_model.load_bidaf("bidaf/data/tmp/bidaf_{:02}.h5".format(init_epoch)) # when you want to resume training
    train_generator, validation_generator = load_data_generators(8, 400, squad_version=args.squad_version, div_epoch_num=20)
    keras_model = train_model(bidaf_model, train_generator, validation_generator=validation_generator
                                        , workers=1, use_multiprocessing=True, initial_epoch=init_epoch, epochs=10)
