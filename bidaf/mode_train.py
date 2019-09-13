from keras.callbacks import CSVLogger, ModelCheckpoint, TensorBoard
### add for TensorBoard
import keras.backend.tensorflow_backend as KTF
import tensorflow as tf
from . import BidirectionalAttentionFlow

import configparser
import os


# 学習のテンプレート
def train_model(bidaf: BidirectionalAttentionFlow,
                train_generator, validation_generator=None, validation_steps=None, 
                steps_per_epoch=None, initial_epoch=0, epochs=1,
                workers=1, use_multiprocessing=True, shuffle=True, 
                save_history=True, save_model_per_epoch=True):

    # _settings/BiDAF.cfg  から デフォルト設定を読み込み
    # (未) Windows や他の OS のパス形式に対応させる必要あり。
    inifile = configparser.ConfigParser()
    inifile.read(os.path.join('_settings', 'BiDAF.cfg'), 'UTF-8')
    print(inifile)
    saved_model_dir = inifile.get('data', 'model_dir_path')
    saved_tmp_dir = os.path.join(saved_model_dir, os.pardir, 'tmp')
    if not os.path.exists(saved_tmp_dir):
        os.makedirs(saved_tmp_dir)

    ### 学習準備（collbackの用意）
    callbacks = []

    # old_session = KTF.get_session()

    # session = tf.Session('')
    # KTF.set_session(session)
    # KTF.set_learning_phase(1)
    tb_cb = TensorBoard(
                log_dir="bidaf/data/tmp/tf_log/",
                histogram_freq=0,
                write_grads=True,
                write_images=1,
                #embeddings_freq=1,
            )
    callbacks.append(tb_cb)

    if save_history:
        history_file = os.path.join(saved_tmp_dir, 'history')
        csv_logger = CSVLogger(history_file, append=True)
        callbacks.append(csv_logger)

    if save_model_per_epoch:
        model_file_path = os.path.join(saved_tmp_dir, 'bidaf_{epoch:02d}.h5')
        checkpointer = ModelCheckpoint(filepath=model_file_path, verbose=1)
        callbacks.append(checkpointer)

    history = bidaf.model.fit_generator(train_generator, steps_per_epoch=steps_per_epoch, epochs=epochs,
                                        callbacks=callbacks, validation_data=validation_generator,
                                        validation_steps=validation_steps, workers=workers,
                                        use_multiprocessing=use_multiprocessing, shuffle=shuffle,
                                        initial_epoch=initial_epoch)
    if not save_model_per_epoch:
        bidaf.model.save(os.path.join(saved_model_dir, 'bidaf.h5'))

    return history, bidaf.model


