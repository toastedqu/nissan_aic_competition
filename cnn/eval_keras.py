# ライブラリのインポート
import numpy as np
from keras.preprocessing.image import load_img, img_to_array
from keras.models import model_from_json
from sklearn.metrics import average_precision_score
import os
import time

# モデル構築 or 読込部 ※以下はモデル構築でなく読込みの例です
model_path = "model.json"
weights_path = "model_weights.hdf5"
json_string = open(model_path).read()
model = model_from_json(json_string)
model.load_weights(weights_path)

# 画像読み込み用の関数の作成
def get_img_array(img_path, size):
    img = load_img(img_path, target_size=size)
    arr = img_to_array(img)
    arr = np.expand_dims(arr, axis=0)
    return arr

# 評価用の数値を格納するリストの用意
y_true = []
y_scores = []
elapsed_times = []

# 画像の path の指定
path = "test/"
classes = ["parking", "street"]
IMG_WIDTH = 1920
IMG_HEIGHT = 1080
img_size = [IMG_WIDTH, IMG_HEIGHT]

# 精度と処理時間の計算
for cls in classes:
    files = os.listdir(path + cls)
    files.sort()
    for file in files:
        img_array = get_img_array(path + cls + "/" + file, size=img_size)
        t1 = time.time() # 処理の開始時間の計測
        pred_prob = model.predict(img_array) # 画像へのモデル適用部
        t2 = time.time() # 処理の終了時間の計測
        elapsed_time = t2 - t1 # 処理時間の計算
        elapsed_times.append(elapsed_time)
        
        # 予測確率の取得
        y_score = pred_prob[0][0]
        y_scores.append(y_score)

        # 正解ラベルの取得
        label = 1 if cls=="parking" else 0
        y_true.append(label)

# 平均処理時間の表示
print(sum(elapsed_times)/len(elapsed_times))

# 平均精度 (Average Precision) の表示
y_true = np.array(y_true)
y_scores = np.array(y_scores)
ap = average_precision_score(y_true,y_scores)
print(ap)