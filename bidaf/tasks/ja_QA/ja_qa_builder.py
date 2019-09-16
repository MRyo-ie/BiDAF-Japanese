import gzip
import json
import os, sys
import re
import numpy as np
from six.moves.urllib.request import urlretrieve
from tqdm import tqdm
from .tokenizer import TokenizerSP
from ..task_data_builder import TaskBuilder, TaskData


class JaQA_Data(TaskData):
    def __init__(self, local_dirpath):
        super().__init__(local_dirpath)
        self.download_base_url = "http://www.cl.ecei.tohoku.ac.jp/rcqa/data/all-v1.0.json.gz"
        gz_fname = self.download_base_url.split('/')[-1]
        self.gz_fpath = os.path.join(self.local_dirpath, gz_fname)

        # ファイル名
        self._train_fname = "train.json"
        self._val_fname = "val.json"
        # パス + ファイル名
        self.train_fpath = os.path.join(self.local_dirpath, self._train_fname)
        self.val_fpath = os.path.join(self.local_dirpath, self._val_fname)
        # json データ
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
            
        print('      (JaQA_Data)    _train_data の要素数 : {}'.format(len(self._train_data)))
        print('      (JaQA_Data)    _val_data   の要素数 : {}'.format(len(self._val_data)))

    # @override
    def total_examples(self, dataset):
        # exec_load_json() を使わないなら、実装しなくてOK
        pass


######################
######################


class JaQA_Builder(TaskBuilder):
    def __init__(self, task_d: JaQA_Data):
        super().__init__(task_d)

    @property
    def is_there_is_impossible(self):
        # 「回答可能か」のフラグもデータに含まれている場合。
        return True
    # 出力ファイルの base name
    def buildf_base_name(self, tier):
        # ver 違いとか、タスク内で分岐がある場合に手直しする。
        return tier

    # @override
    def exec_preprocess(self, tier)->list:
        ### 変数初期化
        dataset = None
        if tier == 'train':
            dataset = self.task_d.train_data
        elif tier == 'val':
            dataset = self.task_d.val_data

        # カウンター
        num_exs = 0  # number of examples written to file
        num_mappingprob, num_multiansprob, num_tokenprob = 0, 0, 0
        num_spanalignprob = 0

        # Tokenizer
        tokenizer = TokenizerSP()
        # 結果
        examples = []

        ### build !!
        """
        ・ SQuAD との違い
            ・ SQuAD： １文章／５質問＋？
            ・ 日本語： ５文章／１質問
            すなわち、質問と文章 の比が逆。
        """
        # dataset = [ json1, json2, json3,... ]
        ### 質問文：Q
        for question_id in tqdm(range(len(dataset)), desc="Preprocessing {}".format(tier)):
            q_json = dataset[question_id]
            # << Q >>
            question = q_json['question'].strip()  # string
            question_tokens = tokenizer.tokenize(question)

            ### 問題文：H,  答え：A
            help_docs = q_json['documents']
            for dataid in range(len(help_docs)):
                # 解答可能性（0~5）のうち、とりあえず 3以上と 0（不可能）を使う。
                score = help_docs[dataid]['score']

                if score in [1, 2]:
                    # 微妙なデータなので飛ばす。
                    continue

                # << H >>
                context = help_docs[dataid]['text'].strip()  # string
                # The following replacements are suggested in the paper
                # BidAF (Seo et al., 2016)
                context = context.replace("``", '" ').replace("''", '" ')
                context_tokens = tokenizer.tokenize(context)

                print(question)
                print(score)
                print(context)
                print(context_tokens)
                ##### マッピング
                '''
                charloc2wordloc
                    { 0:("hello", 0),  1:("hello", 0)...,  5:("hello", 0),
                    　6:("world", 1),  7:...,  10:("world", 1) }
            　　を作る。
                '''
                # charloc2wordloc maps the character location (int) of a context token to a pair giving (word (string), word loc (int)) of that token
                charloc2wordloc = self.get_char_word_loc_mapping(
                                            context, context_tokens)
                # (con)text の char mapping に失敗したら、その tsxt は飛ばす。
                if charloc2wordloc is None:
                    num_mappingprob += 1
                    print('[Error] char mapping が失敗したようです...。')
                    print('        qid : {},    title : {}'.format(q_json['qid'], help_docs[dataid]['title']))
                    continue  # skip this context example

                # 試し
                print(context)
                sys.exit(0)

                # << A >>
                ans_text = help_docs[dataid]['answer']
                # 用意のしようがない...。
                #ans_start_charloc = qn['answers'][0]['answer_start']
                if score == 0:
                    is_impossible = 1
                else:
                    is_impossible = 0

                """
                答えを抜き出す系（答えの文章中の位置を表す数： answer_start が合ってるか確認してる）
                ・ num_spanalignprob：その位置の単語と答えの単語が違う（エラー文）のカウント
                [手順]
                    1. 文章中から、answer に完全一致する箇所を探す。
                        ・ 複数ある場合は、2-gram 中の 問題文の一致率で 算出。
                            本当は、そこもh人手ででやるべきなんだけど...。
                        ・ 最大３つを別問題として登録？
                    2. 1.で選んだ「答え単語の位置」を、答えとする。
                """
                ans_charposi_list = [m.span() for m in re.finditer(ans_text, context)]
                ans_start_c, ans_end_c = ans_charposi_list[-1]
                if len(ans_charposi_list) > 1:
                    """
                    複数箇所ある場合： 2-gram 一致率で計算
                    1. 前後の単語(2-gram分)を context_tokens から取得。
                    2. 1.の単語が 質問文(Q) にいくつ含まれるかを計算
                    3. 1.2.をans_charposi_list全てに対して行い、最も多かった単語位置を採用する
                    """
                    num_multiansprob += 1

                # Unicode の 2バイト文字対策？
                if context[ans_start_c:ans_end_c] != ans_text:
                    # Sometimes this is misaligned, mostly because "narrow builds" of Python 2 interpret certain Unicode characters to have length 2 https://stackoverflow.com/questions/29109944/python-returns-length-of-2-for-single-unicode-character-string
                    # We should upgrade to Python 3 next year!
                    num_spanalignprob += 1
                    continue

                """
                ・num_tokenprob：最終確認
                    '-' とかがあると、たまにおかしくなるらしい？
                """
                # get word locs for answer start and end (inclusive)
                # answer start word loc
                ans_start_w = charloc2wordloc[ans_start_c][1]
                # answer end word loc
                ans_end_w = charloc2wordloc[ans_end_c - 1][1]
                assert ans_start_w <= ans_end_w

                # Check retrieved answer tokens match the provided answer text.
                # Sometimes they won't match, e.g. if the context contains the phrase "fifth-generation"
                # and the answer character span is around "generation",
                # but the tokenizer regards "fifth-generation" as a single token.
                # Then ans_tokens has "fifth-generation" but the ans_text is "generation", which doesn't match.
                ans_tokens = context_tokens[ans_start_w:ans_end_w + 1]
                if "".join(ans_tokens) != "".join(ans_text.split()):
                    num_tokenprob += 1
                    print('''
[Error] 最終テスト失敗...。')
        join(ans_tokens) : ', "".join(ans_tokens))
        join(ans_text)   : ', "".join(ans_text))
                    ''')
                    continue  # skip this question/answer pair

                examples.append((
                    ' '.join(context_tokens),
                    ' '.join(question_tokens),
                    ' '.join(ans_tokens),
                    ' '.join([str(ans_start_w), str(ans_end_w)]), str(is_impossible)))
                num_exs += 1





        print("[確認] char mapping との数が会わずに失敗した数 : ", num_mappingprob)
        print("Number of (context, question, answer) triples discarded because character-based answer span is unaligned with tokenization: ", num_tokenprob)
        print("Number of (context, question, answer) triples discarded due character span alignment problems (usually Unicode problems): ", num_spanalignprob)
        print("Processed %i examples of total %i\n" %
            (num_exs, num_exs + num_mappingprob + num_tokenprob + num_spanalignprob))

        return examples

