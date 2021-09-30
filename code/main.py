import pandas as pd
import numpy as np
import datetime
import matplotlib.pyplot as plt
from datetime import datetime as dt

from sklearn import svm, ensemble
#from sklearn.model_selection import train_test_split

# データセットのクラス
class Dataset:
    N = 0
    dayFrom = datetime.date(1985,10,26)
    data  = np.array([[],[]])
    ind = []

    # コンストラクタ
    def __init__(self, dataset, ind):
        #strptime(tstr, '%Y-%m-%d %H:%M:%S')
        d_temp = dt.strptime(dataset[0,0], '%Y/%m/%d') # 日付を抽出
        self.dayFrom = dt.date(d_temp) # 起点日として格納

        self.N = len(dataset)
        self.data = dataset
        self.ind = ind

    # 指定キーの特徴量の差分をデータセットに追加するメソッド
    def addDifference(self, key):
        temp = self.ind.index(key)
        nf_temp = np.array([[0]])

        for i in range(self.N-1):
            nf_temp = np.vstack((nf_temp, np.array([[ self.data[i+1,temp] -self.data[i,temp]]]) ))

        self.data = np.hstack((self.data, nf_temp))
        self.ind.append( key +'-Diff')


# データファイルを読み込み（数値データとして）
rawdata = pd.read_csv('..\data\dataset_japan_20210928.csv', delimiter=',').values

# データセットの準備
ds_temp = rawdata[104:611,0:6] # 有効なデータ切り出し

# Datasetインスタンスを生成
#       日付， 新規感染者数，   重傷者数，    要療養者数，    回復者数，     確認中
ind = ['date','new-con-case','severe-case','req-inpat-case','discha-case','to-be-con']
ds = Dataset(ds_temp, ind)

# 特徴量の準備
ds.addDifference('discha-case')

# 特徴量のプロット作成
x = np.arange(0, ds.N)
fig0 = plt.figure()
ax = fig0.add_subplot(1,1,1)
ax.plot(x, ds.data[:,1], label=ds.ind[1]) # new-con-case
ax.plot(x, ds.data[:,2], label=ds.ind[2]) # severe-case
ax.plot(x, ds.data[:,3], label=ds.ind[3]) # req-inpat-case
#ax.plot(x, ds.data[:,4], label=ds.ind[4]) # discha-case
ax.plot(x, ds.data[:,5], label=ds.ind[5]) # to-be-con
ax.plot(x, ds.data[:,6], label=ds.ind[6]) # discha-case-Diff
plt.legend(loc='best')
plt.show()

# 学習データを作成
