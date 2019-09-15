import abc
import json
import numpy as np
import os
from tqdm import tqdm
from six.moves.urllib.request import urlretrieve


###   「タスクを Build するために必要な情報」の抽象クラス  ###
class TaskData(metaclass=abc.ABCMeta):
    def __init__(self, local_dirpath):
        self.local_dirpath = local_dirpath

    @abc.abstractproperty  # getter： train のファイル名
    def train_fname(self):
        pass
    @abc.abstractproperty  # getter： val のファイル名
    def val_fname(self):
        pass

    @abc.abstractproperty  # getter： train の json データ
    def train_data(self):
        pass
    @abc.abstractproperty  # getter： val の json データ
    def val_data(self):
        pass

    @abc.abstractmethod
    def download_data(self):
        """ 1. データをダウンロード : data_download() """
        pass

    def exec_download(self, dl_URL, local_fpath, show_progress=True):
        """
        1. の補助
        ダウンロード のテンプレート（ super().exec_download(dl_URL, local_fpath) で使えるはず。 ）
        """
        class DownloadProgressBar(tqdm):
            def update_to(self, b=1, bsize=1, tsize=None):
                """
                b: int, optional
                    Number of blocks just transferred [default: 1].
                bsize: int, optional
                    Size of each block (in tqdm units) [default: 1].
                tsize: int, optional
                    Total size (in tqdm units). If [default: None] remains unchanged.
                """
                if tsize is not None:
                    self.total = tsize
                self.update(b * bsize - self.n)

        filename = dl_URL.split('/')[-1]
        if not os.path.exists(local_fpath):
            try:
                if show_progress:
                    print("[・・]  {} をダウンロード中...".format(filename))
                    # Download with a progress bar
                    with DownloadProgressBar(unit='B', unit_scale=True,
                                            miniters=1, desc=filename) as t:
                        urlretrieve(dl_URL, filename=local_fpath,
                                    reporthook=t.update_to)
                else:
                    # Simple download with no progress bar
                    urlretrieve(dl_URL, filename=local_fpath)

                print("[ OK ] File {} の ダウンロードが 完了しました！\n".format(filename))
            except AttributeError as e:
                print("[Error] An error occurred when downloading the file! Please get the dataset using a browser.")
                raise e
            except KeyboardInterrupt as k:
                if os.path.exists( local_fpath ):
                    os.remove( local_fpath )
                raise k
        return None


    @abc.abstractmethod
    def load_data(self):
        """ 2. trainデータ と valデータに分割（-> json x2）: divide_data() """
        pass

    def exec_load_json(self, local_fpath):
        """
        2. の補助
        json を読み込む
        """
        with open(local_fpath) as f:
            data = json.load(f)
        # print("[確認] データの総数 ： ", self.total_examples(data))
        return data


    # @abc.abstractmethod
    # def total_examples(self, dataset):
    #     """Returns the total number of (context, question, answer) triples, given the data loaded from the SQuAD json file"""
    #     # squad の場合↓
    #     '''
    #     total = 0
    #     for article in dataset['data']:
    #         for para in article['paragraphs']:
    #             total += len(para['qas'])
    #     return total
    #     '''
    #     pass


######################
######################


###  「build を実行」の抽象クラス  ###
class TaskBuilder(metaclass = abc.ABCMeta):

    def __init__(self, task_data):
        self.task_d: TaskData = task_data
        np.random.seed(42)

    @abc.abstractproperty  # getter： 「回答可能か」のフラグも問題に含むか :bool
    def is_there_is_impossible(self):
        pass
    @abc.abstractmethod  # getter： 出力ファイル名 :str
    def outf_base_name(self, tier:str)->str:
        pass

    # これだけ実行。
    def preprocess_data(self):
        """
        3. 文中の単語に番号を振る作業？ : get_char_word_loc_mapping()
        4. Tokenize → 
        ・ ~.span ：答えの文を文中から探して、最初と最後の単語が何番目か数える
        ・ ~.context ：問題文
        ・ ~.question ：質問文
        ・ ~.answer ：答え
        を、それぞれのファイルに保存する。
        """
        # Train
        train_fname = self.outf_base_name('train')
        if self.is_already_preprocessed(train_fname + '.span'):  # すでに終わってる場合は飛ばす。
            print('[確認] {} は、すでに build 済みです。'.format(train_fname))
        else:
            builded_data = self.exec_preprocess('train')
            self.exec_write(builded_data, train_fname)
        # Val
        val_fname = self.outf_base_name('val')
        if self.is_already_preprocessed(val_fname + '.span'):  # すでに終わってる場合は飛ばす。
            print('[確認] {} は、すでに build 済みです。'.format(val_fname))
        else:
            builded_data = self.exec_preprocess('val')
            self.exec_write(builded_data, val_fname)


    def is_already_preprocessed(self, filename):
        """ train.context がすでにあるかどうかを調べる """
        return os.path.isfile(os.path.join( self.task_d.local_dirpath, filename ))


    def get_char_word_loc_mapping(self, context, context_tokens):
        """
        （恐らく、答えが文の部分選択なので）文中の単語に番号を振る作業？
        Return a mapping that maps from character locations to the corresponding token locations.
        If we're unable to complete the mapping e.g. because of special characters, we return None.

        Inputs:
        ・context: string (unicode)
        ・context_tokens: list of strings (unicode)

        Returns:
        ・mapping: dictionary from ints (character locations) to (token, token_idx) pairs
                Only ints corresponding to non-space character locations are in the keys
                e.g. if context = "hello world" and context_tokens = ["hello", "world"] then
                0,1,2,3,4 are mapped to ("hello", 0) and 6,7,8,9,10 are mapped to ("world", 1)
                { 0:("hello", 0),  1:("hello", 0)...,  5:("hello", 0),
                  6:("world", 1),  7:...,  10:("world", 1) }
        """
        acc = ''  # accumulator
        current_token_idx = 0  # current word loc
        mapping = dict()

        # step through original characters
        for char_idx, char in enumerate(context):
            if char != u' ' and char != u'\n':  # if it's not a space:
                acc += char  # add to accumulator
                context_token = context_tokens[current_token_idx]  # current word token
                if acc == context_token:  # if the accumulator now matches the current word token
                    # char loc of the start of this word
                    syn_start = char_idx - len(acc) + 1
                    for char_loc in range(syn_start, char_idx + 1):
                        mapping[char_loc] = (acc, current_token_idx)  # add to mapping
                    acc = ''  # reset accumulator
                    current_token_idx += 1

        if current_token_idx != len(context_tokens):
            # print('[Error] get_char_word_loc_mapping()  :  current_token_idx != len(context_tokens)')
            # print(context)
            return None
        else:
            return mapping


    @abc.abstractmethod
    def exec_preprocess(self, tier):
        """Reads the dataset, extracts context, question, answer, tokenizes them, and calculates answer span in terms of token indices.
        Note: due to tokenization issues, and the fact that the original answer spans are given in terms of characters, some examples are discarded because we cannot get a clean span in terms of tokens.

        This function produces the {train/dev}.{context/question/answer/span} files.

        Inputs:
        dataset: read from JSON
        tier: string ("train" or "dev")
        out_dir: directory to write the preprocessed files
        Returns:
        the number of (context, question, answer) triples written to file by the dataset.
        """
        pass


    def exec_write(self, examples, out_fname):
        out_fpath_base = os.path.join(self.task_d.local_dirpath, out_fname)

        # shuffle examples
        indices = list(range(len(examples)))
        np.random.shuffle(indices)

        with open(out_fpath_base + '.context', 'w', encoding='utf-8') as context_f, \
                open(out_fpath_base + '.question', 'w', encoding='utf-8') as question_f, \
                open(out_fpath_base + '.answer', 'w', encoding='utf-8') as ans_text_f, \
                open(out_fpath_base + '.span', 'w', encoding='utf-8') as span_f:

            if self.is_there_is_impossible:
                is_impossible_f = open(out_fpath_base + '.is_impossible', 'w', encoding='utf-8')

            for i in indices:
                if self.is_there_is_impossible:
                    (context, question, answer, answer_span, is_impossible) = examples[i]
                else:
                    (context, question, answer, answer_span) = examples[i]

                # write tokenized data to file
                context_f.write( context + '\n' )
                question_f.write( question + '\n')
                ans_text_f.write( answer + '\n')
                span_f.write( answer_span + '\n')

                if self.is_there_is_impossible:
                    is_impossible_f.write( is_impossible + '\n')

            if self.is_there_is_impossible:
                is_impossible_f.close()


