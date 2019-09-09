import tensorflow as tf
from keras import backend as K

def limit_GPU_memory_size(percent:int=0):
    config = tf.ConfigProto()
    if percent == 0:
        config.gpu_options.allow_growth = True
    else:
        print(percent/100)
        config.gpu_options.per_process_gpu_memory_fraction = percent/100
    sess = tf.Session(config=config)
    K.set_session(sess)


