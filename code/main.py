import Dataset

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn import svm, ensemble


# データファイルを読み込み（数値データとして）
rawdata = pd.read_csv('..\data\dataset_japan_20210928.csv', delimiter=',').values

# データセットの準備
ds_orig = rawdata[104:611,0:6] # 有効なデータ切り出し

# Datasetインスタンスを生成
#       日付， 新規感染者数，   重傷者数，    要療養者数，    回復者数，     確認中
ind = ['date','new-con-case','severe-case','req-inpat-case','discha-case','to-be-con']
ds = Dataset.Dataset(ds_orig, ind)

# データセットのプロット作成
x = np.arange(0, ds.N)
fig0 = plt.figure()
ax = fig0.add_subplot(1,1,1)
ax.plot(x, ds.data[:,1], label=ds.ind[1]) # new-con-case
ax.plot(x, ds.data[:,2], label=ds.ind[2]) # severe-case
ax.plot(x, ds.data[:,3], label=ds.ind[3]) # req-inpat-case
#ax.plot(x, ds.data[:,4], label=ds.ind[4]) # discha-case
ax.plot(x, ds.data[:,5], label=ds.ind[5]) # to-be-con
plt.legend(loc='best')
plt.show()


## 特徴量セットの生成
fvs = Dataset.Dataset( ds_orig[:,0], 'date') # 日付のみの空のデータセットを生成

# 特徴量の抽出（指定キーの1ステップ差分），ラベルの生成
fv0, fn0 = ds.extractDifference('discha-case')

# 特徴量の抽出（指定ステップ前（現ステップ含む）までのデータ），ラベルの生成
fv1, fn1 = ds.extractPrecede('severe-case',4)


# 特徴量の決定

