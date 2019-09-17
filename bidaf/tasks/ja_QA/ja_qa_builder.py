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
    def get_char_word_loc_mapping(self, context, context_tokens):
        """
        （恐らく、答えが文の部分選択なので）文中の単語に番号を振る作業？
        Returns:
        ・mapping: 
                e.g. if context = "hello world" and context_tokens = ["hello", "world"] then
                0,1,2,3,4 are mapped to ("hello", 0) and 6,7,8,9,10 are mapped to ("world", 1)
                { 0:("hello", 0),  1:("hello", 0)...,  5:("hello", 0),
                  6:("world", 1),  7:...,  10:("world", 1) }
        """
        '''
        【問題点】
        ・「元文」と「分割」　：　原因
        ・　' '　と　'▁'　：　SP は空白を ＿　に変換する。
        ・　（　と　(　：　全角／半角
        ・　…　と　...　：　字数が違う
        【アルゴリズム】
        1. 文字数が一致することを確認 → この時点で、mapping は理論上可能だと言える。
            → 
        2. 半角／全角 などの細かい差を修正しながら、おかしい文字を探していく。
        '''
        # 未知語の扱いは？
        # step through original characters
        char_sun = 0
        for t in context_tokens:
            char_sun += len(t)
        
        acc = ''  # accumulator
        t_idx = 0  # current word loc
        mapping = dict()
        # 文字数が同じなら、そのまま mapping して終わり。
        if len(context) == char_sun:
            for c_idx, char in enumerate(context):
                acc += char
                # 同じ文字数になったら、tmp変数を初期化
                if len(acc) == len(context_tokens[t_idx]):
                    t_start = c_idx - len(acc) + 1
                    for char_loc in range(t_start, c_idx +1):
                        mapping[char_loc] = (acc, t_idx)
                    acc = ''
                    t_idx += 1
            return mapping
        else:
            # print('\n\n[Error](prob1)  context の文字数が違います。  len(context), char_sun = {}, {}'.format(len(context), char_sun))
            # ずれた場所を探す。
            c_loc_s = 0
            for t_idx, c_token in enumerate(context_tokens):
                # 最後 の文字がズレてたら、原因は間違いなくそこ。
                c_loc_e = c_loc_s + len(c_token) -1
                # 超えてたら、エラーで落ちるので break
                if c_loc_e >= len(context):
                    c_loc_e = len(context) -1
                    break
                if context[c_loc_e] != c_token[-1]:
                    # ズレとは関係ないのは続行
                    if  c_token[-1] == u'▁' or\
                            context[c_loc_e] == u'℃':
                        c_loc_s += len(c_token)
                        continue
                    break
                c_loc_s += len(c_token)
            # print('[確認](prob1)  ズレた場所？　：　', t_idx)
            # print('              context[i] ：　「{}」'.format( context[c_loc_s-1:c_loc_e+1] ))
            # print('              tokens[i]  ：　「　{}」'.format( c_token ))
            # print(context)
            # print(context_tokens)
            return None
    
    def is_same_char(self, char1, char2):
        pass

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
        # debug = 0

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
        ### 質問文：Q,  答え：A
        for question_id in tqdm(range(len(dataset)), desc="Preprocessing {}".format(tier)):
            q_json = dataset[question_id]
            # << Q >>
            question = q_json['question'].strip()  # string
            question_tokens = tokenizer.tokenize(question)
            # print('    question_tokens : ', question_tokens)

            # << A >>
            ans_text = q_json['answer']

            ### 問題文：H
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
                # 字数が変わるものを、積極的に変える。
                context = context.replace("…", '...').replace('\n', '')
                context = context.replace("（", '(').replace("）", ')')
                context = context.replace("℃", '°C').replace("Ⅱ", 'II')
                context_tokens = tokenizer.tokenize(context)

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
                    continue  # skip this context example

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
                # 一箇所もなかった場合
                if len(ans_charposi_list) == 0:
                    num_multiansprob += 1
                    continue
                # とりあえず、最後を取ってる。
                ans_start_c, ans_end_c = ans_charposi_list[-1]
                if len(ans_charposi_list) > 1:
                    """
                    複数箇所ある場合： 2-gram 一致率で計算
                    1. 前後の単語(2-gram分)を context_tokens から取得。
                    2. 1.の単語が 質問文(Q) にいくつ含まれるかを計算
                    3. 1.2.をans_charposi_list全てに対して行い、最も多かった単語位置を採用する
                    """
                    num_multiansprob += 1
                    # print('\n[Error](prob2+)  答えが複数箇所存在 : \n      ', ans_text, ans_charposi_list)
                    # print(context)

                # Unicode の 2バイト文字対策？
                if context[ans_start_c:ans_end_c] != ans_text:
                    # Sometimes this is misaligned, mostly because "narrow builds" of Python 2 interpret certain Unicode characters to have length 2 https://stackoverflow.com/questions/29109944/python-returns-length-of-2-for-single-unicode-character-string
                    # We should upgrade to Python 3 next year!
                    num_spanalignprob += 1
                    print('\n[Error](prob2)  Unicode の 2バイト文字対策？ : ', ans_charposi_list)
                    print('        (context[ans_start_c:ans_end_c]) {},      (ans_text) {}'.format(context[ans_start_c:ans_end_c], ans_text))
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
                ans_token_t = "".join(ans_tokens)
                ans_text_t = "".join(ans_text.split())
                if ans_token_t != ans_text_t:
                    num_tokenprob += 1
#                     print('''\n
# [Error](prob3)  最終テスト失敗...。
#         ans_tokens : {}
#         answer txt : {} '''.format("".join(ans_tokens), "".join(ans_text)))
                    #print(context_tokens)
                    '''
                    例）
                        ans_tokens : アメリカの
                        answer txt : アメリカ
                        ans_tokens : 午後10時から
                        answer txt : 午後10時
                        ans_tokens : マンマ・ミーア!』
                        answer txt : マンマ・ミーア!      ← ここまでセーフ（見分ける方法が...。）
                        ans_tokens : は大宝律令
                        answer txt : 大宝律令
                    '''
                    # 前後２文字くらいは OK にすることにした。
                    if ans_token_t[0:2] != ans_text_t[0:2] \
                            or len(ans_token_t) - 2 > len(ans_text_t):
                        continue  # skip this question/answer pair
                    # else:
                    #     print('[確認](prob3)  <- 見逃しました。')

                examples.append((
                    ' '.join(context_tokens),
                    ' '.join(question_tokens),
                    ' '.join(ans_tokens),
                    ' '.join([str(ans_start_w), str(ans_end_w)]), str(is_impossible)))
                num_exs += 1


            # 試し
            # debug += 1
            # if debug == 2000:
            #     sys.exit(0)


        print("[確認] char mapping との数が合わずに失敗した数 : ", num_mappingprob)
        print("Number of (context, question, answer) triples discarded because character-based answer span is unaligned with tokenization: ", num_spanalignprob)
        print("[確認] Tokenize の切れ目の関係で、助詞とががくっついてしまったパターン: ", num_tokenprob)
        print("[確認] 答えの句が複数（０個）ある問題文（※ 除いてはいない。） : ", num_multiansprob)
        print("Processed %i examples of total %i\n" %
            (num_exs, num_exs + num_mappingprob + num_tokenprob + num_spanalignprob))
            
        return examples

