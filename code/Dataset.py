import numpy as np
import datetime
from datetime import datetime as dt

# データセットのクラス
class Dataset:
    N = 0
    dayFrom = datetime.date(1985,10,26)
    data  = np.array([[],[]])
    ind = []

    # コンストラクタ
    def __init__(self, dataset, ind):
        self.N = len(dataset)

        # データセットが2次元配列の場合
        if dataset.ndim == 2 :
            self.data = dataset
            self.ind = ind
            d_temp = dt.strptime(dataset[0,0], '%Y/%m/%d') # 日付を抽出
        # データセットが1次元配列の場合
        else:
            self.data = dataset.reshape(self.N,-1)
            self.ind = [ind]
            d_temp = dt.strptime(dataset[0], '%Y/%m/%d') # 日付を抽出

        self.dayFrom = dt.date(d_temp) # 起点日を格納

    # 指定キーの特徴量の差分を抽出するメソッド
    def extractDifference(self, key):
        k = self.ind.index(key)
        fv_temp = np.array([[0]])

        for i in range(self.N-1):
            fv_temp = np.vstack((fv_temp, np.array([[ self.data[i+1,k] -self.data[i,k]]]) ))

        fv = fv_temp
        keystr = key +'-Diff'
        #self.data = np.hstack((self.data, nf_temp))
        #self.ind.append( key +'-Diff')

        return fv, keystr

    # 指定キーの特徴量の過去numステップ分を特徴量として抽出するメソッド
    def extractPrecede(self, key, num):
        k = self.ind.index(key)
        #fv = np.array([])
        for i in range(self.N):
            fv_temp = np.array([]) # 空の配列を宣言
            for j in range(num):
                if i-j>=0:
                    fv_temp = np.append(fv_temp, self.data[i-j,k])
                else:
                    fv_temp = np.append(fv_temp, 0)
                
            try:
                fv = np.vstack((fv,fv_temp))
            except UnboundLocalError: # fvが未定義のエラーが出たら
                fv = fv_temp

        keystr = []
        for i in range(num):
            keystr.append(key+'_-'+str(i))

        return fv, keystr

    # 特徴量をデータセットに追加
    def append(self, fv, fn):
        self.data = np.hstack((self.data, fv))
        #self.data = np.block([self.data[], fv])
        self.ind.append(fn)
