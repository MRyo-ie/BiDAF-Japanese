import configparser
import glob
import os
import sentencepiece as sp
import sys
from urllib.request import urlretrieve



class SentencePieceTrainer():
    def __init__(self, config:configparser, wiki_glob_path, model_dir_abspath):
        self.wiki_glob_path = wiki_glob_path
        self.prefix = os.path.join( model_dir_abspath, config.get('train', 'vocab_size') )
        self.vocab_size = config.get('train', 'vocab_size')

        if not os.path.exists(model_dir_abspath):
            os.mkdir(model_dir_abspath)


    def _get_text_file(self):
        file_list = glob.glob(self.wiki_glob_path)
        files = ",".join(file_list)
        return files


    def train(self):
        files = self._get_text_file()
        command = f'--input={files} --model_prefix={self.prefix} --vocab_size={self.vocab_size}'
        #print('    (SPTrainer)  command : ', command)
        sp.SentencePieceTrainer.Train(command)

