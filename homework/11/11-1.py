from sklearn.datasets import load_wine
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# 載入 Wine 資料集（分類紅酒種類）
X, y = load_wine(return_X_y=True)

# 分割資料
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# 建立並訓練邏輯斯迴歸分類器
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# 預測與評估
y_pred = model.predict(X_test)
print("Wine 分類準確率:", accuracy_score(y_test, y_pred))
