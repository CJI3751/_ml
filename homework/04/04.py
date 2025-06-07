import numpy as np

# === 七段顯示器輸入對應 ===
seven_segment_truth_table = {
    0:  (1, 1, 1, 1, 1, 1, 0),
    1:  (0, 1, 1, 0, 0, 0, 0),
    2:  (1, 1, 0, 1, 1, 0, 1),
    3:  (1, 1, 1, 1, 0, 0, 1),
    4:  (0, 1, 1, 0, 0, 1, 1),
    5:  (1, 0, 1, 1, 0, 1, 1),
    6:  (1, 0, 1, 1, 1, 1, 1),
    7:  (1, 1, 1, 0, 0, 0, 0),
    8:  (1, 1, 1, 1, 1, 1, 1),
    9:  (1, 1, 1, 1, 0, 1, 1)
}

binary_outputs = {
    i: tuple(map(int, f"{i:04b}")) for i in range(10)
}

#資料轉換
input_vectors = np.array([seven_segment_truth_table[i] for i in range(10)])  # shape: (10, 7)
target_outputs = np.array([binary_outputs[i] for i in range(10)])           # shape: (10, 4)

#損失函數 (均方誤差)
def loss_function(flat_weights):
    w = np.array(flat_weights).reshape(7, 4)
    predictions = input_vectors @ w
    return np.mean((predictions - target_outputs) ** 2)

#預測函數
def predict(segment_input, weights, threshold_func=np.round):
    raw = segment_input @ weights
    return threshold_func(raw).astype(int)

# 梯度下降法實作
def gradientDescendent(loss_func, p, learning_rate=0.01, max_iters=10000, tolerance=1e-6):
    prev_loss = loss_func(p)
    for _ in range(max_iters):
        grad = numerical_gradient(loss_func, p)
        p = [x - learning_rate * dx for x, dx in zip(p, grad)]
        new_loss = loss_func(p)
        if abs(new_loss - prev_loss) < tolerance:
            break
        prev_loss = new_loss
    for i in range(len(p)):
        global flat_weights
        flat_weights[i] = p[i]

def numerical_gradient(f, p, delta=1e-5):
    grad = []
    for i in range(len(p)):
        p_up = p[:]
        p_down = p[:]
        p_up[i] += delta
        p_down[i] -= delta
        df = (f(p_up) - f(p_down)) / (2 * delta)
        grad.append(df)
    return grad

#啟動訓練 
weights_init = np.random.rand(7, 4)
flat_weights = weights_init.flatten().tolist()
gradientDescendent(loss_function, flat_weights)
trained_weights = np.array(flat_weights).reshape(7, 4)

print("預測結果")
for segment in seven_segment_truth_table.values():
    seg_input = np.array(segment)
    pred = predict(seg_input, trained_weights)
    pred_str = ''.join(map(str, pred))
    input_str = ''.join(map(str, segment))
    print(f"Input: {input_str} -> Predicted: {pred_str}")

