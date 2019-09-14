from keras.utils import Sequence
import os
import numpy as np
from .magnitude import MagnitudeVectors


class BatchGenerator(Sequence):
    'Generates data for Keras'

    vectors = None

    def __init__(self, gen_type, batch_size, emdim, squad_version, max_passage_length, max_query_length, shuffle, div_epoch_num=1):
        base_dir = os.path.join(os.path.dirname(__file__), os.pardir, 'Drive')

        self.vectors = MagnitudeVectors(emdim).load_vectors()
        self.squad_version = squad_version

        self.max_passage_length = max_passage_length
        self.max_query_length = max_query_length

        self.context_file = os.path.join(base_dir, 'squad', gen_type + '-v{}.context'.format(squad_version))
        self.question_file = os.path.join(base_dir, 'squad', gen_type + '-v{}.question'.format(squad_version))
        self.span_file = os.path.join(base_dir, 'squad', gen_type + '-v{}.span'.format(squad_version))
        if self.squad_version == 2.0:
            self.is_impossible_file = os.path.join(base_dir, 'squad', gen_type +
                                                   '-v{}.is_impossible'.format(squad_version))
        
        # batch の設計
        self.batch_size = batch_size
        i = 0
        with open(self.span_file, 'r', encoding='utf-8') as f:
            for i, _ in enumerate(f):
                pass
        ### 【高速化１】 1 epoch を無理やり分割
        self.div_epoch_num = div_epoch_num
        self.div_epoch_count = 0
        self.len_dev_epochs = (i // self.div_epoch_num) + 1
        self.len_batches = (self.len_dev_epochs // self.batch_size) + 1
        print('     i : {} ,   len_batches : {}'.format(i, self.len_batches))
        
        self.indices = np.arange(i + 1)
        self.shuffle = shuffle
        ### 【高速化２】 メモリに全て読み込んでしまう
        ##　→ OOM(out of memory エラー なので、全部は恐らく無理。)

    def __len__(self):
        # 1 epoch あたりの step数
        return self.len_batches

    def __getitem__(self, index):
        """
        index : 0 ~ __len()
        """
        start_index = (index * self.batch_size) + (self.len_batches * self.div_epoch_count)
        end_index = start_index + self.batch_size
        
        # シャッフル用。
        inds = self.indices[start_index:end_index]

        ### 読み込み
        contexts = []
        with open(self.context_file, 'r', encoding='utf-8') as cf:
            # (3, '3行目の文字列')
            for i, line in enumerate(cf, start=1):
                # 改行を削除
                line = line[:-1]
                # 今回の処理対象の行なら
                if i in inds:
                    # <！> 英語なので、空白で区切る
                    contexts.append(line.split(' '))

        questions = []
        with open(self.question_file, 'r', encoding='utf-8') as qf:
            for i, line in enumerate(qf, start=1):
                line = line[:-1]
                if i in inds:
                    questions.append(line.split(' '))

        answer_spans = []
        with open(self.span_file, 'r', encoding='utf-8') as sf:
            for i, line in enumerate(sf, start=1):
                line = line[:-1]
                if i in inds:
                    answer_spans.append(line.split(' '))

        if self.squad_version == 2.0:
            is_impossible = []
            with open(self.is_impossible_file, 'r', encoding='utf-8') as isimpf:
                for i, line in enumerate(isimpf, start=1):
                    line = line[:-1]
                    if i in inds:
                        is_impossible.append(line)

            for i, flag in enumerate(is_impossible):
                contexts[i].insert(0, "unanswerable")
                if flag == "1":
                    answer_spans[i] = [0, 0]
                else:
                    answer_spans[i] = [int(val) + 1 for val in answer_spans[i]]

        ### 単語 Embedding
        context_batch = self.vectors.query(contexts, pad_to_length=self.max_passage_length)
        question_batch = self.vectors.query(questions, pad_to_length=self.max_query_length)
        if self.max_passage_length is not None:
            span_batch = np.expand_dims(np.array(answer_spans, dtype='float32'),
                                        axis=1).clip(0, self.max_passage_length - 1)
        else:
            span_batch = np.expand_dims(np.array(answer_spans, dtype='float32'), axis=1)
        return [context_batch, question_batch], [span_batch]

    def on_epoch_end(self):
        # div_epoch_count = 1 なら、1 になって if に入る。
        self.div_epoch_count += 1
        if self.div_epoch_count == self.div_epoch_num:
            if self.shuffle:
                np.random.shuffle(self.indices)
            self.div_epoch_count = 0
