import configparser
import glob
import os
import sentencepiece as sp
import sys
from urllib.request import urlretrieve



class SentencePieceTrainer():
    def __init__(self, config:configparser, wiki_dirpath):
        self.wiki_dirpath = wiki_dirpath
        self.prefix = config['train']['model_dir']
        self.vocab_size = config['train']['vocab_size']

    def _get_text_file(self):
        file_list = glob.glob(f'{self.wiki_dirpath}/**/wiki_')
        files = ",".join(file_list)
        return files


    def train(self):
        files = self._get_text_file()
        command = f'--input={files} --model_prefix={self.prefix} --vocab_size={self.vocab_size}'
        sp.SentencePieceTrainer.Train(command)

