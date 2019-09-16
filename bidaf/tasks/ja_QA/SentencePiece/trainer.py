import configparser
import glob
import os
import sentencepiece as sp
import sys
from urllib.request import urlretrieve



class SentencePieceTrainer():
    def __init__(self, config:configparser, wiki_glob_path):
        self.wiki_glob_path = wiki_glob_path
        self.prefix = os.path.join( os.path.dirname(os.path.abspath(__file__)), config.get('train', 'model_dir'))
        self.vocab_size = config['train']['vocab_size']

    def _get_text_file(self):
        file_list = glob.glob(self.wiki_glob_path)
        files = ",".join(file_list)
        return files


    def train(self):
        files = self._get_text_file()
        command = f'--input={files} --model_prefix={self.prefix} --vocab_size={self.vocab_size}'
        #print('    (SPTrainer)  command : ', command)
        sp.SentencePieceTrainer.Train(command)

