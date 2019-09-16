import configparser
import os

class ConfigPathUtils():
    def __init__(self, group, root_dir, base_path_item='base_path'):
        """
        root_dir = os.path.dirname(os.path.abspath(__file__))
        """
        # Drive のパスを読み込み： _settings/BiDAF.cfg  から デフォルト設定を読み込み
        self.cfg = configparser.ConfigParser()
        self.cfg.read(os.path.join('_settings', 'SavePaths.cfg'), 'UTF-8')
        #print(dict(self.cfg.items()))
        self.group = group
        self.base_path = os.path.join( root_dir, self._rebuild(base_path_item) )
        # print('    root_dir  : ', root_dir)
        # print('    base_path : ', self.base_path )

    def _rebuild(self, path_item):
        """
        パスをリビルドする。
         ＝ os ごとの差分吸収のために、一旦 '/' で分解して、os.path.join() で再構築する。
        """
        return os.path.join( *self.cfg.get(self.group, path_item).split('/') )

    def get_path(self, dir_key):
        dir_path = os.path.join( self.base_path, self._rebuild(dir_key) )
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
        return dir_path

