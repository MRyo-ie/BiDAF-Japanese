# fork 元
https://github.com/ParikhKadam/bidaf-keras

# ライブラリインストール
conda, pyenv など、仮想環境に入るのを忘れない！
```
$ cndac_BiDAF_test
```
1. keras, tensorflow をインストール
    ```
    pip uninstall tensorflow
    pip uninstall tensorflow-gpu
    pip install tensorflow-gpu==1.13.2
    pip install keras 
    ```
   - 確認
      ```
      $ python3
      from tensorflow.python.client import device_lib
      device_lib.list_local_devices()
      ```
2. pip 関連
    ```
    pip install -r _settings/requirements.txt  
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



# 【Train】（英語版）
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



# その他ログ
## インストール時
```
(tensor2tensor) [e165738@localhost] Study$ > pip install bidaf-keras                                   [  8月30, 2:42 ]
Collecting bidaf-keras
  Downloading https://files.pythonhosted.org/packages/2c/26/31925afe8c3a15f849d748d4b6df06b778ae4d55c0c4834012e91cd64a5f/bidaf_keras-1.0.0-py3-none-any.whl
Collecting keras (from bidaf-keras)
  Downloading https://files.pythonhosted.org/packages/f8/ba/2d058dcf1b85b9c212cc58264c98a4a7dd92c989b798823cc5690d062bb2/Keras-2.2.5-py2.py3-none-any.whl (336kB)
    100% |████████████████████████████████| 337kB 7.6MB/s 
Requirement already satisfied: tqdm in /home/e165738/.conda/envs/tensor2tensor/lib/python3.7/site-packages (from bidaf-keras) (4.31.1)
Requirement already satisfied: nltk in /home/e165738/.conda/envs/tensor2tensor/lib/python3.7/site-packages (from bidaf-keras) (3.4)
Collecting pymagnitude (from bidaf-keras)
  Downloading https://files.pythonhosted.org/packages/0a/a3/b9a34d22ed8c0ed59b00ff55092129641cdfa09d82f9abdc5088051a5b0c/pymagnitude-0.1.120.tar.gz (5.4MB)
    100% |████████████████████████████████| 5.4MB 4.6MB/s 
Requirement already satisfied: h5py in /home/e165738/.conda/envs/tensor2tensor/lib/python3.7/site-packages (from keras->bidaf-keras) (2.9.0)
Requirement already satisfied: six>=1.9.0 in /home/e165738/.conda/envs/tensor2tensor/lib/python3.7/site-packages (from keras->bidaf-keras) (1.12.0)
Requirement already satisfied: pyyaml in /home/e165738/.conda/envs/tensor2tensor/lib/python3.7/site-packages (from keras->bidaf-keras) (5.1)
Requirement already satisfied: keras-applications>=1.0.8 in /home/e165738/.conda/envs/tensor2tensor/lib/python3.7/site-packages (from keras->bidaf-keras) (1.0.8)
Requirement already satisfied: keras-preprocessing>=1.1.0 in /home/e165738/.conda/envs/tensor2tensor/lib/python3.7/site-packages (from keras->bidaf-keras) (1.1.0)
Requirement already satisfied: scipy>=0.14 in /home/e165738/.conda/envs/tensor2tensor/lib/python3.7/site-packages (from keras->bidaf-keras) (1.2.1)
Requirement already satisfied: numpy>=1.9.1 in /home/e165738/.conda/envs/tensor2tensor/lib/python3.7/site-packages (from keras->bidaf-keras) (1.16.2)
Requirement already satisfied: singledispatch in /home/e165738/.conda/envs/tensor2tensor/lib/python3.7/site-packages (from nltk->bidaf-keras) (3.4.0.3)
Building wheels for collected packages: pymagnitude
  Building wheel for pymagnitude (setup.py) ... -ls
done
  Stored in directory: /home/e165738/.cache/pip/wheels/a2/c7/98/cb48b9db35f8d1a7827b764dc36c5515179dc116448a47c8a1
Successfully built pymagnitude
Installing collected packages: keras, pymagnitude, bidaf-keras
Successfully installed bidaf-keras-1.0.0 keras-2.2.5 pymagnitude-0.1.120
```


