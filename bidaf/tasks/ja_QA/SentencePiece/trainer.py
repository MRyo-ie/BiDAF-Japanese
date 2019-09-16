import configparser
import glob
import os
import sentencepiece as sp
import shutil
import sys
from urllib.request import urlretrieve



class SentencePieceTrainer():
    def __init__(self, config:configparser, wiki_extpath, model_dir_abspath):
        self.wiki_glob_path = os.path.join(wiki_extpath, '**', "wiki_*")
        self.prefix = os.path.join( model_dir_abspath, config.get('train', 'vocab_size') )
        self.vocab_size = config.get('train', 'vocab_size')
        if not os.path.exists(model_dir_abspath):
            os.mkdir(model_dir_abspath)
        
        # Sentence Piece は、パスに スペースが入ってると動かない...。ので、一旦別の場所に移動する。
        self.is_space_in_path = ' ' in model_dir_abspath
        if self.is_space_in_path:
            root_dir = os.path.join('tmp', 'SentencePiece')
            # Wikipedia extractデータ をコピー
            if not os.path.exists(root_dir):
                os.mkdir(root_dir)
            shutil.copytree(wiki_extpath, root_dir)
            self.wiki_glob_path = os.path.join(root_dir, '**', "wiki_*")
            # model ファイルは、通常の場所にもコピーする。という方向で。
            self.prefix = os.path.join( root_dir, config.get('train', 'vocab_size') )
            self.prefix_tmp = os.path.join( model_dir_abspath, config.get('train', 'vocab_size') )


    def _get_text_file(self):
        file_list = glob.glob(self.wiki_glob_path)
        files = ",".join(file_list)
        return files


    def train(self):
        files = self._get_text_file()
        command = f'--input="{files}" --model_prefix={self.prefix} --vocab_size={self.vocab_size}'
        #print('    (SPTrainer)  command : ', command)
        sp.SentencePieceTrainer.Train(command)
        if self.is_space_in_path:
            shutil.copytree(self.prefix, self.prefix_tmp)

