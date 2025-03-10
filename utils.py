import torch


def calculate_upsilon(C, T, alpha, eps, gamma_c=None, gamma_t=None):
    T_i = torch.sum(T, axis=1)
    # caculate gradient (add eps to avoid log(0))
    if gamma_c != None and gamma_t == None:
        gradient = C + gamma_c * torch.log(torch.unsqueeze(T_i, axis=1).expand(-1, C.shape[1]) + eps)
    elif gamma_c == None and gamma_t != None:
        gradient = C + gamma_t * torch.log(torch.unsqueeze(T_i, axis=1).expand(-1, C.shape[1]) + eps)
    upsilon = T * torch.exp(-alpha * gradient)
    return upsilon

def gaussian_kernel(x, y, sigma=0.01):
    return torch.exp(-(x-y)*(x-y) / (2 * (sigma ** 2))).reshape(1,-1)

def calculate_gaussian_kernel_matrix(device, Y, sigma=1):
    num_samples = Y.shape[0]
    kernel_matrix = torch.zeros((num_samples, num_samples)).to(device)
    for i in range(num_samples):
        kernel_matrix[i] = gaussian_kernel(Y[i], Y, sigma)
    return kernel_matrix
