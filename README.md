# fork 元
https://github.com/ParikhKadam/bidaf-keras

# ライブラリインストール
conda, pyenv など、仮想環境に入るのを忘れない！
```
$ cndac_BiDAF_test
```

1. pip 関連
    ```
    pip install -r _settings/requirements.txt  
    ```
1. (GPUが動いてないっぽい場合)
   - 確認
      ```
      $ python3
      from tensorflow.python.client import device_lib
      device_lib.list_local_devices()
      ```
   - keras, tensorflow を再インストール
    ```
    pip uninstall tensorflow
    pip uninstall tensorflow-gpu
    pip install tensorflow-gpu==1.13.2
    pip install keras 
    ```



# 【Predict】 テスト（英語版）
## 準備
### Pre-trained モデル を使わせてもらう
https://github.com/ParikhKadam/bidaf-keras#pre-trained-models
1. 以下をダウンロード
   - **Model Name:** [bidaf_50.h5](https://drive.google.com/open?id=10C56f1DSkWbkBBhokJ9szXM44P9T-KfW)
     **Model Configuration:**
       - lowercase: True
       - batch size: 16
       - max passage length: None
       - max question length: None
       - embedding dimension: 400
       - squad version: 1.1

2. bidaf/data/models/ に移動する。
   - ※ 場所を変えたい場合は、
      `_settings/BiDAF.cfg` の `dir_path` に、パスを指定するとできる！


## Python インタプリタ で実行
```
$ python3
>>>
```
事前準備
```
from bidaf import BidirectionalAttentionFlow
bidaf_model = BidirectionalAttentionFlow(400)
bidaf_model.load_bidaf("bidaf/data/models/bidaf_50.h5")
```
テスト！
```
bidaf_model.predict_ans("This is a tree", "What is this?")
#=> {'answer': 'tree'}

bidaf_model.predict_ans("This is a tree which highest in my country.", "What is this?")
#=> {'answer': 'tree which highest in my country'}

bidaf_model.predict_ans("Apollo ran from 1961 to 1972, and was supported by the two-man Gemini program which ran concurrently with it from 1962 to 1966. Gemini missions developed some of the space travel techniques that were necessary for the success of the Apollo missions. Apollo used Saturn family rockets as launch vehicles. Apollo/Saturn vehicles were also used for an Apollo Applications Program, which consisted of Skylab, a space station that supported three manned missions in 1973–74, and the Apollo–Soyuz Test Project, a joint Earth orbit mission with the Soviet Union in 1975.", "What space station supported three manned missions in 1973–1974?")
#=> {'answer': 'skylab'}
```

## bot 形式？
```
$ python3 bot.py

# 問題文の例
Apollo ran from 1961 to 1972, and was supported by the two-man Gemini program which ran concurrently with it from 1962 to 1966. Gemini missions developed some of the space travel techniques that were necessary for the success of the Apollo missions. Apollo used Saturn family rockets as launch vehicles. Apollo/Saturn vehicles were also used for an Apollo Applications Program, which consisted of Skylab, a space station that supported three manned missions in 1973–74, and the Apollo–Soyuz Test Project, a joint Earth orbit mission with the Soviet Union in 1975.

# 質問文の例
What space station supported three manned missions in 1973–1974?
```

## まとめて解かせる方式
準備中



# 【Train】SQuAD（英語版）
## 準備（SQuAD）
- v1.1 の場合
  ```
  $ python3 setup.py
  ```
- v2.0 の場合
  ```
  $ python3 setup.py -sv=2.0
  ```
- オプションで
  - `-l`：`do_lowercase`
  - ` > bidaf/data/tmp/log.txt`

## Python インタプリタ で実行
```
$ python3
>>>
```
```
from bidaf import BidirectionalAttentionFlow
from bidaf.scripts import load_data_generators
bidaf_model = BidirectionalAttentionFlow(400)
#bidaf_model.load_bidaf("bidaf/data/models/eng_model_test.h5") # when you want to resume training
train_generator, validation_generator = load_data_generators(24, 400)
keras_model = bidaf_model.train_model(train_generator, validation_generator=validation_generator)
```

## スクリプト で実行
```
python3 syugyo.py -sv=2.0
```
- オプションで
  - `-l`：`do_lowercase`
  - ログを撮りたい場合
    - 標準出力 ： ` > bidaf/data/tmp/log.txt`
    - エラー出力 ： ` 2> bidaf/data/tmp/log_err.txt`


