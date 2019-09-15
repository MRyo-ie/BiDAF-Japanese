import gzip
import json
import os, sys
import numpy as np
from six.moves.urllib.request import urlretrieve
from tqdm import tqdm
from .tokenizer import tokenize
from ..task_data_builder import TaskBuilder, TaskData


class JaQA_Data(TaskData):
    def __init__(self, local_dirpath):
        super().__init__(local_dirpath)
        self.download_base_url = "http://www.cl.ecei.tohoku.ac.jp/rcqa/data/all-v1.0.json.gz"
        gz_fname = self.download_base_url.split('/')[-1]
        self.gz_fpath = os.path.join(self.local_dirpath, gz_fname)

        self._train_fname = "train.json"
        self._val_fname = "val.json"

        self.train_fpath = os.path.join(self.local_dirpath, self._train_fname)
        self.val_fpath = os.path.join(self.local_dirpath, self._val_fname)
        
        self.val_rate = 0.2  #valのデータ数の割合 → 20%
        self._train_data = []
        self._val_data = []
        np.random.seed(42)

    @property
    def train_fname(self):
        return self._train_fname
    @property
    def val_fname(self):
        return self._val_fname

    @property
    def train_data(self):
        return self._train_data
    @property
    def val_data(self):
        return self._val_data

    # @override
    def download_data(self, show_progress=True):
        # Orizinal gz
        self.exec_download(self.download_base_url, self.gz_fpath)

    def load_data(self):
        """ 2. trainデータ と valデータに分割（-> json x2）: divide_data() """
        # .json.gz を解凍
        with gzip.open(self.gz_fpath, 'rb') as in_f, \
                open(self.train_fpath, 'wb') as train_f, \
                open(self.val_fpath, 'wb') as val_f:

            # train : val = 8 : 2 くらいで。
            data_lines = in_f.readlines()
            # シャッフル
            #d_num = len(in_f)  # ← gzip は len がないらしい...。
            d_num = len(data_lines)
            indices = list(range(d_num))  # valに割り当てる番号
            np.random.shuffle(indices)
            val_num = int(d_num * self.val_rate)  # 20% を val にする。比率：self.val_rate
            indices = indices[:val_num]

            json_d = None
            for i, line in enumerate(data_lines):
                json_d = json.loads(line)
                if i in indices:
                    # Val
                    val_f.write(line)
                    self._val_data.append(json_d)
                else:
                    # Train
                    train_f.write(line)
                    self._train_data.append(json_d)
            
        print('    _train_data の要素数 : {}'.format(len(self._train_data)))
        print('    _val_data   の要素数 : {}'.format(len(self._val_data)))

    # @override
    def total_examples(self, dataset):
        # exec_load_json() を使わないなら、実装しなくてOK
        pass


######################
######################


class JaQA_Builder(TaskBuilder):
    def __init__(self, task_d: JaQA_Data):
        super().__init__(task_d)
        # 「回答可能か」のフラグもデータに含まれている場合。
        self._is_there_is_impossible = True

    @property
    def is_there_is_impossible(self):
        return self._is_there_is_impossible
    # 出力ファイルの base name
    def outf_base_name(self, tier):
        return os.path.join('{}-v{}'.format(tier, self.squad_v))

    # @override
    def exec_preprocess(self, tier)->list:
        # 各種変数初期化
        do_lowercase = self.do_lowercase
        dataset = None
        if tier == 'train':
            dataset = self.task_d.train_data
        elif tier == 'val':
            dataset = self.task_d.val_data

        # build !!
        num_exs = 0  # number of examples written to file
        num_mappingprob, num_tokenprob, num_spanalignprob = 0, 0, 0
        examples = []

        for articles_id in tqdm(range(len(dataset['data'])), desc="Preprocessing {}".format(tier)):

            article_paragraphs = dataset['data'][articles_id]['paragraphs']
            for pid in range(len(article_paragraphs)):

                context = article_paragraphs[pid]['context'].strip()  # string

                # The following replacements are suggested in the paper
                # BidAF (Seo et al., 2016)
                context = context.replace("''", '" ')
                context = context.replace("``", '" ')

                context_tokens = tokenize(context, do_lowercase=do_lowercase)  # list of strings (lowercase)

                if do_lowercase:
                    context = context.lower()

                qas = article_paragraphs[pid]['qas']  # list of questions

                # charloc2wordloc maps the character location (int) of a context token to a pair giving (word (string), word loc (int)) of that token
                charloc2wordloc = self.get_char_word_loc_mapping(
                                            context, context_tokens)

                if charloc2wordloc is None:  # there was a problem
                    num_mappingprob += len(qas)
                    continue  # skip this context example

                # for each question, process the question and answer and write to file
                for qn in qas:

                    # read the question text and tokenize
                    question = qn['question'].strip()  # string
                    question_tokens = tokenize(question, do_lowercase=do_lowercase)  # list of strings

                    # of the three answers, just take the first
                    # get the answer text
                    # answer start loc (character count)
                    if not self.is_there_is_impossible:  # squad 1.1
                        ans_text = qn['answers'][0]['text']
                        ans_start_charloc = qn['answers'][0]['answer_start']

                    elif qn['is_impossible'] == True:
                        # some questions in squad 2.0 don't even have plausible answers
                        if qn['plausible_answers'] == []:
                            continue

                        is_impossible = 1
                        ans_text = qn['plausible_answers'][0]['text']
                        ans_start_charloc = qn['plausible_answers'][0]['answer_start']
                    else:
                        is_impossible = 0
                        ans_text = qn['answers'][0]['text']
                        ans_start_charloc = qn['answers'][0]['answer_start']

                    if do_lowercase:
                        ans_text = ans_text.lower()

                    # answer end loc (character count) (exclusive)
                    ans_end_charloc = ans_start_charloc + len(ans_text)

                    # Check that the provided character spans match the provided answer text
                    if context[ans_start_charloc:ans_end_charloc] != ans_text:
                        # Sometimes this is misaligned, mostly because "narrow builds" of Python 2 interpret certain Unicode characters to have length 2 https://stackoverflow.com/questions/29109944/python-returns-length-of-2-for-single-unicode-character-string
                        # We should upgrade to Python 3 next year!
                        num_spanalignprob += 1
                        continue

                    # get word locs for answer start and end (inclusive)
                    # answer start word loc
                    ans_start_wordloc = charloc2wordloc[ans_start_charloc][1]
                    # answer end word loc
                    ans_end_wordloc = charloc2wordloc[ans_end_charloc - 1][1]
                    assert ans_start_wordloc <= ans_end_wordloc

                    # Check retrieved answer tokens match the provided answer text.
                    # Sometimes they won't match, e.g. if the context contains the phrase "fifth-generation"
                    # and the answer character span is around "generation",
                    # but the tokenizer regards "fifth-generation" as a single token.
                    # Then ans_tokens has "fifth-generation" but the ans_text is "generation", which doesn't match.
                    ans_tokens = context_tokens[ans_start_wordloc:ans_end_wordloc + 1]
                    if "".join(ans_tokens) != "".join(ans_text.split()):
                        num_tokenprob += 1
                        continue  # skip this question/answer pair

                    if self.is_there_is_impossible:
                        examples.append((' '.join(context_tokens), ' '.join(question_tokens), ' '.join(
                            ans_tokens), ' '.join([str(ans_start_wordloc), str(ans_end_wordloc)]), str(is_impossible)))
                    else:
                        examples.append((' '.join(context_tokens), ' '.join(question_tokens), ' '.join(
                            ans_tokens), ' '.join([str(ans_start_wordloc), str(ans_end_wordloc)])))

                    num_exs += 1

        print("Number of (context, question, answer) triples discarded due to char -> token mapping problems: ", num_mappingprob)
        print("Number of (context, question, answer) triples discarded because character-based answer span is unaligned with tokenization: ", num_tokenprob)
        print("Number of (context, question, answer) triples discarded due character span alignment problems (usually Unicode problems): ", num_spanalignprob)
        print("Processed %i examples of total %i\n" %
            (num_exs, num_exs + num_mappingprob + num_tokenprob + num_spanalignprob))

        return examples

