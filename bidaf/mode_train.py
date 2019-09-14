from keras.callbacks import CSVLogger, ModelCheckpoint, TensorBoard
### add for TensorBoard
import keras.backend.tensorflow_backend as KTF
import tensorflow as tf
from . import BidirectionalAttentionFlow

import configparser
import os


# Drive のパスを読み込み： _settings/BiDAF.cfg  から デフォルト設定を読み込み
cfg = configparser.ConfigParser()
cfg.read(os.path.join('_settings', 'BiDAF.cfg'), 'UTF-8')
# dict(cfg.items('Drive'))
base_dir_path = cfg.get('Drive', 'base_path').split('/')
saved_wordDB_dir_name_dir = os.path.join( *(base_dir_path +[cfg.get('Drive', 'wordDB_dir_name')]) )
saved_weight_dir = os.path.join( *(base_dir_path +[cfg.get('Drive', 'weights_dir_name')]) )
if not os.path.exists(saved_weight_dir):
    os.makedirs(saved_weight_dir)


# 学習のテンプレート
def train_model(bidaf: BidirectionalAttentionFlow,
                train_generator, validation_generator=None, validation_steps=None, 
                steps_per_epoch=None, initial_epoch=0, epochs=1,
                workers=1, use_multiprocessing=True, shuffle=True, 
                save_history=True, save_model_per_epoch=True):

    ### 学習準備（collbackの用意）
    callbacks = []

    tb_cb = TensorBoard(
                log_dir="bidaf/Logs/tf_log/",
                histogram_freq=0,
                write_grads=True,
                write_images=1,
                #embeddings_freq=1,
            )
    callbacks.append(tb_cb)

    if save_history:
        history_file = os.path.join(saved_weight_dir, 'history')
        csv_logger = CSVLogger(history_file, append=True)
        callbacks.append(csv_logger)

    if save_model_per_epoch:
        model_file_path = os.path.join(saved_weight_dir, 'bidaf_{epoch:02d}.h5')
        checkpointer = ModelCheckpoint(filepath=model_file_path, verbose=1)
        callbacks.append(checkpointer)

    history = bidaf.model.fit_generator(train_generator, steps_per_epoch=steps_per_epoch, epochs=epochs,
                                        callbacks=callbacks, validation_data=validation_generator,
                                        validation_steps=validation_steps, workers=workers,
                                        use_multiprocessing=use_multiprocessing, shuffle=shuffle,
                                        initial_epoch=initial_epoch)
    if not save_model_per_epoch:
        bidaf.model.save(os.path.join(saved_weight_dir, 'bidaf.h5'))

    return history, bidaf.model


