import configparser
import glob
import os
import sentencepiece as sp
import sys
from .SentencePiece.initer import SentencePieceIniter


class TokenizerSP():
    def __init__(self):
        # const.cfg ファイルを読み込み
        self.current_dir = os.path.dirname(os.path.abspath(__file__))
        cfg_path = os.path.join(self.current_dir, 'SentencePiece', 'const.cfg')
        config = configparser.ConfigParser()
        config.read(cfg_path, 'UTF-8')

        # SentencePiece を初期化
        sp_initer = SentencePieceIniter(config)
        self.sp_model = sp_initer.sp_model


    def tokenize(self, sequence):
        """
        日本語の Tokenizer
        ・ Sentence Piece
        ・ mecab
        の両方使えるようにする予定。
        """
        print('日本語Tokenizer!!')
