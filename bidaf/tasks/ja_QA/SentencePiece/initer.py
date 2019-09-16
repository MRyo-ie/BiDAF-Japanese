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
        # 共通 path
        self.wikiExtractor_URL = config['init']['wikiExtractor_URL']
        self.wikiExtractor_fpath = os.path.join(self.current_dir, 'WikiExtractor.py')
        self.wiki_dump_URL = config['init']['Wiki_dump_URL']
        self.wiki_tmppath = os.path.join(self.current_dir, 'tmp')  # tmp/
        self.wiki_filepath = os.path.join(self.wiki_tmppath, self.wiki_dump_URL.split('/')[-1])  # tmp/jawiki-latest-pages-articles-multistream.xml.bz2
        self.wiki_extpath = os.path.join(self.wiki_tmppath, 'out')  # tmp/out/
        if not os.path.exists(self.wiki_tmppath):
            os.mkdir(self.wiki_tmppath)

        self.model_dirpath = config['train']['model_dir']  #os.path.join(self.current_dir, 'model')
        self.sp_trainer = SentencePieceTrainer(config, self.wiki_extpath)


    def download(self):
        """ wikiextractor, Wikipediaデータ のダウンロード """
        # WikiExtractor.py
        if not os.path.exists( self.wikiExtractor_fpath ):
            print('[・・] WikiExtractor.py をダウンロード中...')
            urlretrieve(self.wikiExtractor_URL, self.wikiExtractor_fpath, reporthook)
        print('[確認] WikiExtractor.py を確認しました！')

        # Wikipedia データ
        try:
            print('[・・] Wikipedia データをダウンロード中...。')
            urlretrieve(self.wiki_dump_URL, self.wiki_filepath, reporthook)
        except (Exception, KeyboardInterrupt) as e:
            # ダウンロード中断されたら、とりあえず削除する。
            if os.path.exists( self.wiki_filepath ):
                os.remove( self.wiki_filepath )
                print('\n[Error] ダウンロードが中止されました')
            raise e

        print('[確認] Wikipedia データを確認しました！')


    def extract(self):
        try:
            # wikiExtract_logfile = os.path.join(self.wiki_tmppath, 'extruct_log.log')
            # #wikiExtract_elogfile = os.path.join(self.wiki_tmppath, 'extruct_elog.log')
            # with open(wikiExtract_logfile, 'w') as log_f:#, \
            #     #open(wikiExtract_elogfile, 'w') as elog_f:
            #     proc = subprocess.Popen(['python3', self.wikiExtractor_fpath, 
            #                                 self.wiki_filepath, "-o=" + self.wiki_extpath,
            #                                 '--log_file'],
            #                             stdout=log_f, stderr=subprocess.PIPE)
            # stdout, stderr = proc.communicate()
            # retcode = proc.returncode
            # del proc

            retcode = subprocess.call(['python3', self.wikiExtractor_fpath, self.wiki_filepath,
                                                    "-o=" + self.wiki_extpath, '--log_file'])
        except:
            # エラーが起きたら、全部削除
            import shutil
            shutil.rmtree(self.wiki_extpath)
            raise ScriptRunningError('[Error] Wikipedia の Extract に失敗しました...。\n')#,
            #                            stderr)
        if retcode == 1:
            # エラーが起きたら、全部削除
            import shutil
            shutil.rmtree(self.wiki_extpath)
            raise ScriptRunningError('[Error] Wikipedia の Extract に失敗しました...。\n')#,
            #                            stderr)


    def setup_SP(self):
        ### wikiextractor, Wikipediaデータ のダウンロード
        if not os.path.exists(self.wiki_filepath):
            # なければ、ダウンロードを実行。
            self.download()
        print('[確認] Wikipedia データを確認しました！')

        ### Wikipediaデータ を分解
        if not os.path.exists(self.wiki_extpath):
            # なければ、分解を実行。
            self.extract()
        print('[確認] Wikipediaデータを extract しました！')

        ### Sentence Piece を学習。
        print('[・・] Wikipediaデータを extract しました！')
        self.sp_trainer

        
# Exception を投げる
class ScriptRunningError(Exception):
    pass




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

