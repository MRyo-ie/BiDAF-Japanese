"""Downloads SQuAD train and dev sets, preprocesses and writes tokenized versions to file"""

import os
#from .ja_qa.ja_ga_builder import JaQA_Builder
from .squad.squad_builder import SQuAD_Data, SQuAD_Builder
from .task_data_builder import TaskData, TaskBuilder


class OptionValues():
    def __init__(self, data_dir, squad_version, do_lowercase):
        self.data_dir = data_dir
        self.squad_version = squad_version
        self.do_lowercase = do_lowercase



def build(task: str, opts: OptionValues):
    task_data:TaskData = None
    task_builder:TaskBuilder = None
    if task == 'squad':
        task_data = SQuAD_Data(opts.data_dir, opts.squad_version)
        task_builder = SQuAD_Builder(task_data, opts.squad_version, opts.do_lowercase)
    elif task == 'ja_qa':
        task_data = SQuAD_Data(opts.data_dir, opts.squad_version)
        task_builder = JaQA_Builder(task_data)

    print('[確認] 出力ディレクトリ： ', opts.data_dir)
    if not os.path.exists(opts.data_dir):
        os.makedirs(opts.data_dir)

    '''
    1. データをダウンロード : data_download()
    2. trainデータ と valデータに分割（-> json x2）: divide_data()
      (総数を数える：total_examples())
    '''
    # download train/val data
    task_data.download_data()
    # read train/val
    task_data.load_data()

    '''
    3. 文中の単語に番号を振る作業？ : get_char_word_loc_mapping()
    4. Tokenize → 
        ・ ~.span ：答えの文を文中から探して、最初と最後の単語が何番目か数える
        ・ ~.context ：問題文
        ・ ~.question ：質問文
        ・ ~.answer ：答え
        を、それぞれのファイルに保存する。
    '''
    # preprocess train set and write to file
    task_builder.preprocess_data()
    print("[finish] data preprocessed!")

