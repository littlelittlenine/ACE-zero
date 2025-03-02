import torch

def SVD(W_k):
    # 使用 PyTorch 的 SVD 函数
    U, Sigma, VT = torch.linalg.svd(W_k)
    
    # 创建一个对角矩阵
    S_matrix = torch.zeros((320, 768))
    S_matrix[:Sigma.shape[0], :Sigma.shape[0]] = torch.diag(Sigma)
    
    return U @ S_matrix , VT

W_k = torch.randn(320, 768)
W1, W2 = SVD(W_k)
print(W1.shape, W2.shape)
# 使用 SVD 函数重构 W_k
# 检查重构的矩阵是否与原始矩阵接近
is_reconstruction_close = torch.allclose(W_k, W1 @ W2, atol=1e-5)
print(is_reconstruction_close)  

