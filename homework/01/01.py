import random

def f(x, y, z):
    return x**2 + y**2 + z**2 - 2*x - 4*y - 6*z + 8

def get_neighbors(x, y, z, delta):
    steps = [-delta, 0, delta]
    neighbors = []
    for dx in steps:
        for dy in steps:
            for dz in steps:
                if dx == dy == dz == 0:
                    continue
                neighbors.append((x + dx, y + dy, z + dz))
    return neighbors

def hill_climb(start_x, start_y, start_z, delta=0.1, max_iters=1000):
    x, y, z = start_x, start_y, start_z
    for i in range(max_iters):
        current_val = f(x, y, z)
        neighbors = get_neighbors(x, y, z, delta)
        next_point = min(neighbors, key=lambda p: f(*p))
        next_val = f(*next_point)
        if next_val < current_val:
            x, y, z = next_point
        else:
            break  
    return (x, y, z), f(x, y, z)

start = [random.uniform(-10, 10) for _ in range(3)]
opt_point, opt_val = hill_climb(*start)

print("最小點近似值:", opt_point)
print("函數最小值:", opt_val)

