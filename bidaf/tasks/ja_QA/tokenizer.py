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
        #print(cfg_path)
        config = configparser.ConfigParser()
        config.read(cfg_path, 'UTF-8')
        #print(dict(config.items()))

        self.model_dirpath = config.get('train', 'model_dir')
        # SentencePiece の学習が済んでない場合は、まずそれからやる。
        if not self.is_already_SP_model_learned():
            sp_initer = SentencePieceIniter(config)
            sp_initer.setup_SP()

    # 最初に確認
    def is_already_SP_model_learned(self) -> bool:
        # モデルまで作成済み？
        file_list = glob.glob(self.model_dirpath+'/**/*.model') + glob.glob(self.model_dirpath+'/**/*.vocab')
        print('[確認] SentencePiece のモデル : ', file_list)
        # 2つ揃っている？
        return len(file_list) > 1

    def tokenize(self, sequence):
        """
        日本語の Tokenizer
        ・ Sentence Piece
        ・ mecab
        の両方使えるようにする予定。
        """
        print('日本語Tokenizer!!')
