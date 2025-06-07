from sklearn.datasets import fetch_california_housing
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# 載入加州房價資料集（回歸）
X, y = fetch_california_housing(return_X_y=True)

# 簡單只用其中一個特徵來做回歸（如房間數）
X = X[:, [0]]  # 第0個特徵：MedInc（收入中位數）

# 分割資料
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# 線性回歸模型
model = LinearRegression()
model.fit(X_train, y_train)

# 預測與評估
y_pred = model.predict(X_test)
print("加州房價回歸 MSE:", mean_squared_error(y_test, y_pred))
