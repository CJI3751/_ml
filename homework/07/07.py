from micrograd.engine import Value

# 初始化變數
x = Value(0.0)
y = Value(0.0)
z = Value(0.0)

# 學習率與訓練輪數
learning_rate = 0.1
for step in range(100):
    # 前向計算
    f = x**2 + y**2 + z**2 - 2*x - 4*y - 6*z + 8

    # 將上一次的梯度清零
    x.grad = y.grad = z.grad = 0.0

    # 反向傳播
    f.backward()

    # 梯度下降步驟
    x.data -= learning_rate * x.grad
    y.data -= learning_rate * y.grad
    z.data -= learning_rate * z.grad

    if step % 10 == 0 or step == 99:
        print(f"Step {step}: f = {f.data:.4f}, x = {x.data:.4f}, y = {y.data:.4f}, z = {z.data:.4f}")

