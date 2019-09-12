from ..scripts import BatchGenerator


def load_data_generators(batch_size, emdim, squad_version=1.1, max_passage_length=None, max_query_length=2,
                         shuffle=False, div_epoch_num=1):
    train_generator = BatchGenerator('train', batch_size, emdim, squad_version, max_passage_length, max_query_length,
                                     shuffle, div_epoch_num)
    validation_generator = BatchGenerator('dev', batch_size, emdim, squad_version, max_passage_length, max_query_length,
                                          shuffle, div_epoch_num)
    return train_generator, validation_generator
