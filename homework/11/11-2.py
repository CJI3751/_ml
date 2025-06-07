from sklearn.datasets import make_moons
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# 建立 2D 月亮形狀資料（適合測試分群）
X, _ = make_moons(n_samples=300, noise=0.1, random_state=0)

# KMeans 分群
kmeans = KMeans(n_clusters=2, random_state=0)
labels = kmeans.fit_predict(X)

# 顯示分群結果
plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='Accent')
plt.title("make_moons KMeans 分群")
plt.show()
