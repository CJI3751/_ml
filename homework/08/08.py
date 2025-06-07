import torch

# 建立變數，requires_grad=True 代表要追蹤梯度
x = torch.tensor([0.0], requires_grad=True)
y = torch.tensor([0.0], requires_grad=True)
z = torch.tensor([0.0], requires_grad=True)

# 學習率與迭代次數
learning_rate = 0.1
epochs = 100

for epoch in range(epochs):
    # 計算函數值（前向）
    f = x**2 + y**2 + z**2 - 2*x - 4*y - 6*z + 8

    # 清除上一步的梯度
    if x.grad: x.grad.zero_()
    if y.grad: y.grad.zero_()
    if z.grad: z.grad.zero_()

    # 反向傳播（自動計算梯度）
    f.backward()

    # 使用梯度做參數更新
    with torch.no_grad():
        x -= learning_rate * x.grad
        y -= learning_rate * y.grad
        z -= learning_rate * z.grad

    # 輸出過程
    if epoch % 10 == 0 or epoch == epochs - 1:
        print(f"Epoch {epoch:>3}: f = {f.item():.4f}, x = {x.item():.4f}, y = {y.item():.4f}, z = {z.item():.4f}")
