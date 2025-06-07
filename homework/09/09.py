import torch
import torch.nn as nn
import matplotlib.pyplot as plt

# ===== 1. 產生模擬資料 =====
# y = 2x + 3 + noise
torch.manual_seed(0)
X = torch.unsqueeze(torch.linspace(-5, 5, 100), dim=1)  # shape: [100, 1]
y = 2 * X + 3 + 0.5 * torch.randn(X.size())

# ===== 2. 定義模型 =====
model = nn.Linear(1, 1)  # 輸入1維 -> 輸出1維的線性模型

# ===== 3. 損失函數 & 優化器 =====
criterion = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

# ===== 4. 訓練迴圈 =====
epochs = 100
for epoch in range(epochs):
    # 前向傳播
    y_pred = model(X)
    loss = criterion(y_pred, y)

    # 清除舊梯度
    optimizer.zero_grad()

    # 反向傳播
    loss.backward()

    # 更新參數
    optimizer.step()

    # 每10輪印出一次
    if epoch % 10 == 0 or epoch == epochs - 1:
        w = model.weight.item()
        b = model.bias.item()
        print(f"Epoch {epoch:3d}: Loss = {loss.item():.4f}, w = {w:.4f}, b = {b:.4f}")

# ===== 5. 視覺化結果 =====
predicted = model(X).detach()
plt.scatter(X.numpy(), y.numpy(), label='Data')
plt.plot(X.numpy(), predicted.numpy(), color='red', label='Fitted Line')
plt.legend()
plt.title("Linear Regression with PyTorch")
plt.show()
