from oicl import *
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # param related to dataset
    parser.add_argument('--dataset', type=str, default="ihdp")

    # params related to objective function
    parser.add_argument('--gamma_c', type=float, default=1e-2, help='strength of negative entropy regularization on control group.')
    parser.add_argument('--gamma_t', type=float, default=1e-2, help='strength of negative entropy regularization on treated group.')
    parser.add_argument('--lamb_c', type=float, default=5, help='strength of factual outcome guidance term on treated group.')
    parser.add_argument('--lamb_t', type=float, default=5e-1, help='strength of factual outcome guidance term on control group.')
    parser.add_argument('--dim_reduce', type=int, default=15, help='the dimension of the subspace')

    # params related to learning rate
    parser.add_argument('--alpha_base', type=float, default=1e-1, help='learning rate.')
    parser.add_argument('--alpha_update_steps', type=int, default=20, help='update frequency of learning rate.')
    parser.add_argument('--decay_rate', type=float, default=0.98, help='decay rate of learning rate.')

    # params related to training
    parser.add_argument('--device', type=str, default='cuda', help='choose cuda or cpu')
    parser.add_argument('--abstol', type=float, default=1e-5, help='early stop condition.')
    parser.add_argument('--eps', type=float, default=1e-5, help='help log() avoid 0.')
    parser.add_argument('--max_iter', type=int, default=1000, help='max_iter.')
    parser.add_argument('--seed', type=int, default=0, help='random seed.')
    parser.add_argument('--is_check', type=str, default='True', help='whether to print result during training')
    parser.add_argument('--res_file', type=str, default=None, help='file to save result')

    args = parser.parse_args()

    device = args.device
    dataset = args.dataset

    gamma_c = args.gamma_c
    gamma_t = args.gamma_t
    lamb_c = args.lamb_c
    lamb_t = args.lamb_t
    dim_reduce = args.dim_reduce

    alpha_base = args.alpha_base
    alpha_update_steps = args.alpha_update_steps
    decay_rate = args.decay_rate

    abstol = args.abstol
    eps = args.eps
    max_iter = args.max_iter
    seed = args.seed
    is_check = args.is_check
    res_file = args.res_file

    # List used to save result of each seed
    MAE = []

    for seed in range(10):
        print("Training in seed : " + str(seed))
        mae = main(device=device, gamma_c=gamma_c, gamma_t=gamma_t, lamb_c=lamb_c, lamb_t=lamb_t, dim_reduce=dim_reduce, alpha_base=alpha_base,
                   alpha_update_steps=alpha_update_steps, decay_rate=decay_rate, abstol=abstol, eps=eps, max_iter=max_iter, res_file=res_file, dataset=dataset, seed=seed, is_check=is_check)
        MAE.append(mae)

    MAE = torch.tensor(MAE)
    print(f"result : {MAE.mean():.4f} +- {MAE.std():.4f}")
