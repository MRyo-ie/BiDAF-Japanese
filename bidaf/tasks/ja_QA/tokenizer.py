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
        self.sp_model = sp.SentencePieceProcessor()
        self.sp_model.Load(sp_initer.model_fpath)

    def tokenize(self, sequence):
        """
        日本語の Tokenizer
        ・ Sentence Piece
        ・ mecab
        の両方使えるようにする予定。
        """
        tokens = self.sp_model.EncodeAsPieces(sequence)
        # なぜか、最初の文字に 空白記号が入る...。
        # Token的には間違ってるっぽいので削除。
        if tokens[0] == '▁':
            tokens = tokens[1:]
        if tokens[0][0] == '▁':
            tokens[0] = tokens[0][1:]
        return tokens
