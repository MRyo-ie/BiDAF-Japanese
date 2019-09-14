from keras.callbacks import CSVLogger, ModelCheckpoint, TensorBoard
### add for TensorBoard
import keras.backend.tensorflow_backend as KTF
import tensorflow as tf
from . import BidirectionalAttentionFlow

import configparser
import os



# 学習のテンプレート
def train_model(bidaf: BidirectionalAttentionFlow, paths,
                train_generator, validation_generator=None, validation_steps=None, 
                steps_per_epoch=None, initial_epoch=0, epochs=1,
                workers=1, use_multiprocessing=True, shuffle=True, 
                save_history=True, save_model_per_epoch=True):

    ### 学習準備（collbackの用意）
    callbacks = []

    tb_cb = TensorBoard(
                log_dir = paths['Log_tboard'],
                histogram_freq = 0,
                write_grads = True,
                write_images = 1,
                #embeddings_freq=1,
            )
    callbacks.append(tb_cb)

    if save_history:
        history_file = os.path.join(paths['Log_base_dir'], 'history')
        csv_logger = CSVLogger(history_file, append=True)
        callbacks.append(csv_logger)

    if save_model_per_epoch:
        model_file_path = os.path.join(paths['Drive_weight'], 'bidaf_{epoch:02d}.h5')
        checkpointer = ModelCheckpoint(filepath=model_file_path, verbose=1)
        callbacks.append(checkpointer)

    history = bidaf.model.fit_generator(train_generator, steps_per_epoch=steps_per_epoch, epochs=epochs,
                                        callbacks=callbacks, validation_data=validation_generator,
                                        validation_steps=validation_steps, workers=workers,
                                        use_multiprocessing=use_multiprocessing, shuffle=shuffle,
                                        initial_epoch=initial_epoch)
    if not save_model_per_epoch:
        bidaf.model.save(os.path.join(paths['Drive_weight'], 'bidaf.h5'))

    return history, bidaf.model


