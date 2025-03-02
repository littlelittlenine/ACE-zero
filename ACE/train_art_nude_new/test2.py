import numpy as np

# 假设 A 是一个已知的 320x768 的满秩矩阵
A = np.random.rand(320, 768)  # 这里使用随机矩阵作为示例

# 进行 SVD 分解
U, Sigma, VT = np.linalg.svd(A)
print(U.shape, Sigma.shape, VT.shape)
# 构造 Sigma 矩阵为 768x768
Sigma_full = np.zeros((768, 768))
Sigma_full[:320, :320] = np.diag(Sigma)

# 计算 B 和 C
B = U
C = VT.T @ Sigma_full

# 验证分解是否正确
Areconstructed = B @ C
if np.allclose(A, Areconstructed):
    print("分解成功，A 和 B C 相等。")
else:
    print("分解失败，A 和 B C 不相等。")


