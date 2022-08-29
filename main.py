import os
import argparse
from solver import Solver
from data_loader import get_loader
from torch.backends import cudnn


def str2bool(v):
    return v.lower() in ('true')

def main(config):
    # For fast training.
    cudnn.benchmark = True

    # Create directories if not exist.
    if not os.path.exists(config.log_dir):
        os.makedirs(config.log_dir)
    if not os.path.exists(config.model_save_dir):
        os.makedirs(config.model_save_dir)
    if not os.path.exists(config.sample_dir):
        os.makedirs(config.sample_dir)
    if not os.path.exists(config.result_dir):
        os.makedirs(config.result_dir)

    # Data loader.
    data_loader = None

    if config.dataset in ['Covid']:
        data_loader = get_loader(config.image_dir, config.image_size, config.batch_size,
                                 'Covid', config.mode, config.num_workers)
        
    # Solver for training and testing Fixed-Point GAN.
    solver = Solver(data_loader, config)
    

    if config.mode == 'train':
        if config.dataset in ['Covid']: # add more datasets here
            solver.train()
    elif config.mode == 'test':
        if config.dataset in ['Covid']: # add more datasets here
            solver.test()
    elif config.mode == 'testAUC':
        if config.dataset in ['Covid']:  # add more datasets here
            solver.testAUC()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Model configuration.
    parser.add_argument('--image_size', type=int, default=128, help='image resolution')
    parser.add_argument('--g_conv_dim', type=int, default=64, help='number of conv filters in the first layer of G')
    parser.add_argument('--d_conv_dim', type=int, default=64, help='number of conv filters in the first layer of D')
    parser.add_argument('--g_repeat_num', type=int, default=6, help='number of residual blocks in G')
    parser.add_argument('--d_repeat_num', type=int, default=6, help='number of strided conv layers in D')
    parser.add_argument('--lambda_cls', type=float, default=0, help='weight for domain classification loss')
    parser.add_argument('--lambda_rec', type=float, default=1, help='weight for reconstruction loss')
    parser.add_argument('--lambda_gp', type=float, default=10, help='weight for gradient penalty')
    parser.add_argument('--lambda_id', type=float, default=1, help='weight for identity loss')
    parser.add_argument('--lambda_mask', type=float, default=0.1, help='weight for mask loss')
    parser.add_argument('--lambda_msmall', type=float, default=1, help='weight for mask loss')
    parser.add_argument('--lambda_mzerone', type=float, default=1, help='weight for mask loss')
    parser.add_argument('--mask_loss_mode', type=str, default='sum', help='method for mask loss')
    
    # Training configuration.
    parser.add_argument('--dataset', type=str, default='Covid', choices=['Covid']) # add more datasets here
    parser.add_argument('--batch_size', type=int, default=16, help='mini-batch size')
    parser.add_argument('--num_iters', type=int, default=200000, help='number of total iterations for training D')
    parser.add_argument('--num_iters_decay', type=int, default=100000, help='number of iterations for decaying lr')
    parser.add_argument('--g_lr', type=float, default=0.0001, help='learning rate for G')
    parser.add_argument('--d_lr', type=float, default=0.0001, help='learning rate for D')
    parser.add_argument('--n_critic', type=int, default=2, help='number of D updates per each G update')
    parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for Adam optimizer')
    parser.add_argument('--beta2', type=float, default=0.999, help='beta2 for Adam optimizer')
    parser.add_argument('--resume_iters', type=int, default=None, help='resume training from this step')

    # Test configuration.
    parser.add_argument('--test_iters', type=int, default=200000, help='test model from this step')

    # Miscellaneous.
    parser.add_argument('--num_workers', type=int, default=12)
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'test', 'testAUC'])
    parser.add_argument('--use_tensorboard', type=str2bool, default=True)

    # Directories.
    parser.add_argument('--image_dir', type=str, default='datasets/covid')
    parser.add_argument('--log_dir', type=str, default='covid/logs')
    parser.add_argument('--model_save_dir', type=str, default='covid/models')
    parser.add_argument('--sample_dir', type=str, default='covid/samples')
    parser.add_argument('--result_dir', type=str, default='covid/results')

    # Step size.
    parser.add_argument('--log_step', type=int, default=10)
    parser.add_argument('--sample_step', type=int, default=1000)
    parser.add_argument('--model_save_step', type=int, default=10000)
    parser.add_argument('--lr_update_step', type=int, default=1000)

    config = parser.parse_args()
    print(config)
    main(config)
