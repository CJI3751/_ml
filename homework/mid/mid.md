# 孤立森林（Isolation Forest）演算法簡介與應用

---

## 一、主題簡介
孤立森林（Isolation Forest）是一種無監督式的機器學習演算法，專門用來進行異常偵測（Anomaly Detection）。與傳統模型不同，它不需要標記資料，只需分析資料分佈的結構，就能有效找出異常點。

## 二、所屬分類
- 人工智慧（AI） ➜ 機器學習（ML） ➜ 無監督學習 ➜ 異常偵測 ➜ Isolation Forest

## 三、原理概述
孤立森林的核心概念是：

- 資料異常點在特徵空間中比較孤立、稀有，因此能以較少的隨機切割次數將它們從資料集中分離出來。
- 使用多棵隨機二元樹（類似隨機森林）來「隔離」資料。
- 被隔離得越快的點，越可能是異常點。

## 四、流程步驟
1. 隨機選取特徵與分割值來建立隔離樹（iTree）
2. 重複建立多棵樹構成孤立森林（iForest）
3. 根據每筆資料在樹中的平均隔離長度計算異常分數（Anomaly Score）
4. 設定閾值來區分正常點與異常點

## 五、實作工具
使用 Python 套件：`scikit-learn` 中的 `IsolationForest` 類別

## 六、簡單程式範例（Python）
```python
from sklearn.ensemble import IsolationForest
from sklearn.datasets import make_blobs
import numpy as np

# 建立資料集
X, _ = make_blobs(n_samples=300, centers=1)
X = np.vstack([X, np.random.uniform(-6, 6, size=(20, 2))])  # 加入異常點

# 模型訓練與預測
clf = IsolationForest(contamination=0.06)
clf.fit(X)
labels = clf.predict(X)  # -1 為異常，1 為正常
```

## 七、優點與限制

| ✅ 優點                         | ⚠️ 限制                                    |
|-------------------------------|---------------------------------------------|
| 不需要標記資料                  | 不適合極高維度資料                          |
| 執行速度快、可擴展              | 可能對資料分佈非常不均的情況較不敏感         |
| 適用於大規模資料集              | 不易解釋「為什麼這筆資料被判為異常」         |

---

## 八、結論

孤立森林（Isolation Forest）是一種有效的無監督式異常偵測演算法，它利用「隨機切割」的方法在不需要標記資料的情況下快速識別離群點。雖然不是深度學習方法，但在大量實務場景中表現穩定、速度快、實作簡單，非常適合在工業應用、金融安全、資安領域部署。
