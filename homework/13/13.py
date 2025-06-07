from sklearn.ensemble import IsolationForest
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt
import numpy as np

# 產生資料：包含一些異常值
X, _ = make_blobs(n_samples=300, centers=1, cluster_std=0.60, random_state=42)
X = np.vstack([X, np.random.uniform(low=-6, high=6, size=(20, 2))])  # 加入 20 筆異常值

# 建立並訓練孤立森林模型
clf = IsolationForest(contamination=0.06, random_state=42)  # 約有 6% 異常
clf.fit(X)

# 預測：1 是正常，-1 是異常
y_pred = clf.predict(X)

# 可視化
plt.scatter(X[:, 0], X[:, 1], c=y_pred, cmap='coolwarm', marker='o')
plt.title("Isolation Forest 異常偵測")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.show()
