from configparser import ConfigParser
import glob
import os
import shutil
import subprocess
import sys
from tqdm import tqdm
from urllib.request import urlretrieve
from .trainer import SentencePieceTrainer


class SentencePieceIniter():
    
    def __init__(self, config:ConfigParser):
        self.current_dir = os.path.dirname(os.path.abspath(__file__))
        ### まだモデルを作れてない場合は、全自動で学習。
        model_dir_abspath = os.path.join(self.current_dir, *config.get('train', 'model_dir').split('/'))
        model_fpath = os.path.join(model_dir_abspath, config['train']['model_name'] + '.model')
        vocab_fpath = os.path.join(model_dir_abspath, config['train']['model_name']+'.vocab')
        is_already_SP_model_learned = os.path.exists(model_fpath) and os.path.exists(vocab_fpath)
        if not is_already_SP_model_learned:
            # 変数、パス
            self.wikiExtractor_URL = config['init']['wikiExtractor_URL']
            self.wikiExtractor_py = os.path.join(self.current_dir, 'WikiExtractor.py')
            self.wiki_dump_URL = config['init']['Wiki_dump_URL']
            self.wiki_tmppath = os.path.join(self.current_dir, 'tmp')  # tmp/
            self.wiki_filepath = os.path.join(self.wiki_tmppath, self.wiki_dump_URL.split('/')[-1])  # tmp/jawiki-latest-pages-articles-multistream.xml.bz2
            self.wiki_extpath = os.path.join(self.wiki_tmppath, 'out')  # tmp/out/
            if not os.path.exists(self.wiki_tmppath):
                os.mkdir(self.wiki_tmppath)
            self.sp_trainer = SentencePieceTrainer(config, self.wiki_extpath, model_dir_abspath)
            # 学習開始
            self.setup_SP()
        else:
            ### すでに モデルがあるなら、読み込む。
            self.sp_model = None


    def setup_SP(self):
        ### ダウンロード
        if not os.path.exists(self.wiki_filepath):  # なければ、ダウンロードを実行。
            self.download()
        print('[確認](SP_Initer)  Wikipediaデータを確認しました！')

        ### 分解
        if not os.path.exists(self.wiki_extpath):
            os.mkdir(self.wiki_extpath)
        if len(glob.glob(os.path.join(self.wiki_extpath, '**', "wiki_*"))) < 4:  # なければ、分解を実行。
            self.extract()
        print('[確認](SP_Initer)  Wikipediaデータのextractを確認しました！')

        ### Sentence Piece を学習。
        print('\n[・・](SP_Initer)  Sentence Piece の学習を開始します。')  # tokenizer で確認済み
        self.sp_trainer.train()


    def download(self):
        """ wikiextractor, Wikipediaデータ のダウンロード """
        # WikiExtractor.py
        if not os.path.exists( self.wikiExtractor_py ):
            print('[・・](SP_Initer)  WikiExtractor.py をダウンロード中...')
            with DownloadProgressBar(unit='B', unit_scale=True,
                                    miniters=1, desc='WikiExtractor.py') as t:
                urlretrieve(self.wikiExtractor_URL, self.wikiExtractor_py, reporthook=t.update_to)
        print('[確認](SP_Initer)  WikiExtractor.py を確認しました！')

        # Wikipedia データ
        try:
            print('[・・](SP_Initer)  Wikipedia データをダウンロード中...。')
            with DownloadProgressBar(unit='B', unit_scale=True, miniters=1,
                                    desc='jawiki-latest-pages-articles-multistream.xml.bz2') as t:
                urlretrieve(self.wiki_dump_URL, self.wiki_filepath, reporthook=t.update_to)
        except (Exception, KeyboardInterrupt) as e:
            # ダウンロード中断されたら、とりあえず削除する。
            if os.path.exists( self.wiki_filepath ):
                os.remove( self.wiki_filepath )
                print('\n[Error](SP_Initer)  ダウンロードが中止されました')
            raise e

        print('[確認](SP_Initer)  Wikipedia データを確認しました！')


    def extract(self):
        # logファイル のセットアップ
        wikiExtract_logfile = os.path.join(self.wiki_tmppath, 'extruct_log.txt')
        if os.path.exists(wikiExtract_logfile):
            os.remove(wikiExtract_logfile)
        try:
            print('    self.wiki_tmppath : ', self.wiki_tmppath)
            print('    wikiExtract_logfile : ', wikiExtract_logfile)
            retcode = subprocess.call(['python3', self.wikiExtractor_py, self.wiki_filepath,
                                                    "-o={}".format(self.wiki_extpath),
                                                    '-q',  '-b=500M', '--processes=3',
                                                    "--log_file={}".format(wikiExtract_logfile)])
            print('[確認](SP_Initer)  retcode : ', retcode)
            if not retcode == 0:
                raise ScriptRunningError()
        except:
            # エラーが起きたら、全部削除
            shutil.rmtree(self.wiki_extpath)
            os.remove(wikiExtract_logfile)
            raise ScriptRunningError('[Error](SP_Initer)  Wikipedia の Extract に失敗しました...。\n')




# Exception を投げる
class ScriptRunningError(Exception):
    pass


# def reporthook(blocknum, blocksize, totalsize):
#     '''
#     ダウンロードの時に、プログレスバーを表示するやつ
#     '''
#     readsofar = blocknum * blocksize
#     if totalsize > 0:
#         percent = readsofar * 1e2 / totalsize
#         s = "\r%5.1f%% %*d / %d" % (
#             percent, len(str(totalsize)), readsofar, totalsize)
#         sys.stderr.write(s)
#         if readsofar >= totalsize:  # near the end
#             sys.stderr.write("\n")
#     else:  # total size is unknown
#         sys.stderr.write("read %d\n" % (readsofar,))


class DownloadProgressBar(tqdm):
    def update_to(self, b=1, bsize=1, tsize=None):
        """
        b: int, optional
            Number of blocks just transferred [default: 1].
        bsize: int, optional
            Size of each block (in tqdm units) [default: 1].
        tsize: int, optional
            Total size (in tqdm units). If [default: None] remains unchanged.
        """
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)
