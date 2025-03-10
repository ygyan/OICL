from dataloder import load_ihdp
from utils import *


def oicl(device, X_t, X_c, Y_t, Y_c, gt_ate, res_file=None, dim_reduce=10, lamb_c=1, lamb_t=1, gamma_c=0.1, gamma_t=0.1, alpha_base=0.01,
         alpha_update_steps=10, decay_rate=0.95, max_iter=2000, abstol=1e-5, eps=1e-3, is_check=True):

    # ------------------------------ step1: Initialization ------------------------------
    n_t, n_c = X_t.shape[0], X_c.shape[0]
    n = n_c + n_t
    X = torch.cat((X_c, X_t), dim=0)

    # initialize optimal plan T(our goal)
    T_c = (torch.ones([n_c, n]) / (n_c * n)).to(device)
    T_t = (torch.ones([n_t, n]) / (n_t * n)).to(device)

    # initialize kernel matrix K_c, K_t
    K_t = calculate_gaussian_kernel_matrix(device, Y_t)
    K_c = calculate_gaussian_kernel_matrix(device, Y_c)

    T_c_old = T_c
    T_t_old = T_t

    # ------------------------------ step2: Training ------------------------------
    for i in range(max_iter):
        # Calculte matrix in Eq.(32)
        mat_c1 = torch.mm(torch.mm(X_c.T, torch.diag(torch.mm(T_c,torch.ones(n, 1).to(device)).squeeze(1))), X_c)
        mat_c2 = torch.mm(torch.mm(X.T, torch.diag(torch.mm(T_c.T, torch.ones(n_c, 1).to(device)).squeeze(1))), X)
        mat_c3 = 2 * torch.mm(torch.mm(X.T, T_c.T), X_c)
        theta_c = mat_c1 + mat_c2 - mat_c3

        mat_t1 = torch.mm(torch.mm(X_t.T, torch.diag(torch.mm(T_t, torch.ones(n, 1).to(device)).squeeze(1))), X_t)
        mat_t2 = torch.mm(torch.mm(X.T, torch.diag(torch.mm(T_t.T, torch.ones(n_t, 1).to(device)).squeeze(1))), X)
        mat_t3 = 2 * torch.mm(torch.mm(X.T, T_t.T), X_t)
        theta_t = mat_t1 + mat_t2 - mat_t3

        diag_c = torch.diag(torch.mm(K_c, torch.ones(n_c, 1).to(device)).squeeze(1)) - K_c
        theta_cc = 2 * lamb_c * torch.mm(torch.mm(X_c.T, diag_c), X_c)

        diag_t = torch.diag(torch.mm(K_t, torch.ones(n_t, 1).to(device)).squeeze(1)) - K_t
        theta_tt = 2 * lamb_t * torch.mm(torch.mm(X_t.T, diag_t), X_t)

        theta = theta_c + theta_t + theta_cc + theta_tt

        # Choose the first dim_reduce's eigenvectors
        eigenval, eigenvec = torch.linalg.eig(theta)
        eigenval = torch.real(eigenval)
        eigenvec = torch.real(eigenvec)
        _, min_indices = torch.topk(eigenval, dim_reduce, largest=False)

        # Update projection matrix P
        P = eigenvec[:,min_indices]

        # Update Subspace of covariates
        PX_c = torch.mm(X_c, P)
        PX_t = torch.mm(X_t, P)
        PX = torch.mm(X, P)

        # Update matrix C_c
        dc = torch.diag(torch.mm(PX_c, PX_c.T)).reshape(n_c,-1)
        d = torch.diag(torch.mm(PX, PX.T)).reshape(-1,n)
        C_c = dc + d  - 2 * torch.mm(PX_c, PX.T)

        # Upadate matrix C_t
        dt = torch.diag(torch.mm(PX_t, PX_t.T)).reshape(n_t, -1)
        C_t = dt + d - 2 * torch.mm(PX_t, PX.T)

        # Update lr
        alpha = alpha_base * (decay_rate ** (i // alpha_update_steps))

        # Update T alternately
        if(i % 2 == 0):
            upsilon_c = calculate_upsilon(C=C_c, T=T_c, gamma_c=gamma_c, alpha=alpha, eps=eps)
            T_c_j = torch.sum(upsilon_c, dim=0)
            T_c_new = upsilon_c / (n * torch.unsqueeze(T_c_j, dim=0).expand(n_c, -1))
            T_t_new = T_t_old
        else:
            upsilon_t = calculate_upsilon(C=C_t, T=T_t, gamma_t=gamma_t, alpha=alpha, eps=eps)
            T_t_j = torch.sum(upsilon_t, dim=0)
            T_t_new = upsilon_t / (n * torch.unsqueeze(T_t_j, dim=0).expand(n_t, -1))
            T_c_new = T_c_old

        # Check the change of T to determine whether to stop the iteration
        current_abstol_c = torch.sum(torch.abs(T_c_new - T_c_old))
        current_abstol_t = torch.sum(torch.abs(T_t_new - T_t_old))
        current_abstol = (current_abstol_t + current_abstol_c)/2
        T_c_old = T_c_new
        T_t_old = T_t_new
        T_c = T_c_new
        T_t = T_t_new
        if current_abstol < abstol:
            break

        # print current result
        if is_check == True:
            w_c = torch.sum(T_c_new, dim=1)
            w_t = torch.sum(T_t_new, dim=1)
            pred_ate = torch.sum(w_t * Y_t) - torch.sum(w_c * Y_c)
            if (i % 50 == 0) | (i+1 == max_iter):
                print("i:{}, gt_ate: {:.4f}, pre_ate: {:.4f}, MAE:{:.4f}, current_abstol:{:.6f}, current_abstol_c:{:.6f}, current_abstol_t:{:.6f}".format(
                    i, gt_ate, pred_ate, abs(pred_ate - gt_ate), current_abstol, current_abstol_c, current_abstol_t))
                if res_file:
                    print("i:{}, gt_ate: {:.4f}, pre_ate: {:.4f}, MAE:{:.4f}, current_abstol:{:.6f}".format(
                        i, gt_ate, pred_ate, abs(pred_ate - gt_ate), current_abstol), file=res_file)

    # ------------------------------ step3: Estimate ATE ------------------------------
    w_c = torch.sum(T_c_new, dim=1)
    w_t = torch.sum(T_t_new, dim=1)
    pred_ate = torch.sum(w_t * Y_t) - torch.sum(w_c * Y_c)
    return pred_ate


def main(device, gamma_c, gamma_t, lamb_c, lamb_t, dim_reduce, alpha_base, alpha_update_steps, decay_rate, abstol, eps, max_iter, res_file, dataset, seed, is_check):

    if dataset == "ihdp":
        X_t, X_c, Y_t, Y_c, gt_ate = load_ihdp(seed=seed)

    X_t, X_c = torch.tensor(X_t, dtype=torch.float32).to(device), torch.tensor(X_c, dtype=torch.float32).to(device)
    Y_t, Y_c = torch.tensor(Y_t, dtype=torch.float32).to(device), torch.tensor(Y_c, dtype=torch.float32).to(device)


    pred_ate = oicl(device, X_t, X_c, Y_t, Y_c, gt_ate, dim_reduce=dim_reduce, lamb_c=lamb_c, lamb_t=lamb_t, gamma_c=gamma_c, gamma_t=gamma_t, alpha_base=alpha_base,
                    alpha_update_steps=alpha_update_steps, decay_rate=decay_rate,
                    abstol=abstol, eps=eps, max_iter=max_iter, res_file=res_file, is_check=is_check)

    print("seed: {}, True_ATE: {:.4f}, Pred_ATE: {:.4f}, Error: {:.4f}".format(seed, gt_ate, pred_ate,
                                                                                 abs(gt_ate - pred_ate)))
    if res_file:
        print("seed: {}, True_ATE: {:.4f}, Pred_ATE: {:.5f}, Error: {:.5f}\n".format(seed, gt_ate, pred_ate,
                                                                                    abs(gt_ate - pred_ate)), file=res_file)


    return abs(gt_ate - pred_ate)