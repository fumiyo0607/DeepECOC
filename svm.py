#必要なライブラリを import 
from sklearn import datasets
import numpy as np
import sklearn
from sklearn import datasets
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# クラスラベルを振り直す用の関数
def fix_data(y,one_class):
    y_fix = []
    for i in y:
        if i == one_class:
            y_fix.append(1)
        else:
            y_fix.append(-1)
        
    return y_fix

# SVMを作成する関数
def cleaate_svm(X,y):
    # データの標準化処理
    sc = StandardScaler()
    sc.fit(X)
    X_std = sc.transform(X)

    # 線形SVMのインスタンスを生成
    model = SVC(kernel='linear', random_state=None)
    # モデルの学習。fit関数で行う。
    model.fit(X_std, y)

    return model