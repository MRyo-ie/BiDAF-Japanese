from configparser import ConfigParser
import glob
import os
import subprocess
import sys
from urllib.request import urlretrieve
from .trainer import SentencePieceTrainer


class SentencePieceIniter():
    def __init__(self, config:ConfigParser):
        self.current_dir = os.path.dirname(os.path.abspath(__file__))
        # 定数、パス
        self.wikiExtractor_URL = config['init']['wikiExtractor_URL']
        self.wiki_dump_URL = config['init']['Wiki_dump_URL']
        self.wiki_dirpath = os.path.join(self.current_dir, 'tmp')
        self.wiki_filepath = os.path.join(self.wiki_dirpath, self.wiki_dump_URL.split('/')[-1])

        self.model_dirpath = config['train']['model_dir']  #os.path.join(self.current_dir, 'model')
        self.sp_trainer = SentencePieceTrainer(config, self.wiki_dirpath)


    def download(self):
        # WikiExtractor.py
        wikiExtractor_fpath = os.path.join(self.current_dir, 'WikiExtractor.py')
        if not os.path.exists( wikiExtractor_fpath ):
            print('[・・] WikiExtractor.py をダウンロード中...')
            urlretrieve(self.wikiExtractor_URL, wikiExtractor_fpath, reporthook)
        print('[確認] WikiExtractor.py を確認しました！')

        # Wikipedia データ
        if os.path.exists(self.wiki_filepath):
            # すでにあるなら、飛ばす
            print('[確認] Wikipedia データを確認しました！')
            return
        try:
            print('[・・] Wikipedia データをダウンロード中...。')
            urlretrieve(self.wiki_dump_URL, self.wiki_filepath, reporthook)
        except (Exception, KeyboardInterrupt) as e:
            # ダウンロード中断されたら、とりあえず削除する。
            if os.path.exists( self.wiki_filepath ):
                os.remove( self.wiki_filepath )
                print('\n[Error] ダウンロードが中止されました')
            raise e
    

    def extract(self):
        subprocess.call(['python3', 'WikiExtractor.py', 
                            self.wiki_filepath, "-o="+self.wiki_dirpath])


    def setup_SP(self):
        # wikiextractor, Wikipedia記事 をダウンロード
        self.download()
        self.extract()

        




def reporthook(blocknum, blocksize, totalsize):
    '''
    ダウンロードの時に、プログレスバーを表示するやつ
    '''
    readsofar = blocknum * blocksize
    if totalsize > 0:
        percent = readsofar * 1e2 / totalsize
        s = "\r%5.1f%% %*d / %d" % (
            percent, len(str(totalsize)), readsofar, totalsize)
        sys.stderr.write(s)
        if readsofar >= totalsize:  # near the end
            sys.stderr.write("\n")
    else:  # total size is unknown
        sys.stderr.write("read %d\n" % (readsofar,))

