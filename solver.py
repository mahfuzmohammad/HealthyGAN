from model import Generator
from model import Discriminator
from torch.autograd import Variable
from torchvision.utils import save_image
import torch
import torch.nn.functional as F
import numpy as np
import os
import time
import datetime
# from sklearn.cluster import KMeans
from tqdm import tqdm
import pandas as pd
from sklearn.metrics import roc_curve, auc, roc_auc_score, classification_report, confusion_matrix


class Solver(object):
    """Solver for training and testing Fixed-Point GAN."""

    def __init__(self, data_loader, config):
        """Initialize configurations."""

        # All config
        self.config = config

        # Data loader.
        self.data_loader = data_loader

        # Model configurations.
        # self.c_dim = config.c_dim
        # self.c2_dim = config.c2_dim
        self.image_size = config.image_size
        self.g_conv_dim = config.g_conv_dim
        self.d_conv_dim = config.d_conv_dim
        self.g_repeat_num = config.g_repeat_num
        self.d_repeat_num = config.d_repeat_num
        self.lambda_cls = config.lambda_cls
        self.lambda_rec = config.lambda_rec
        self.lambda_gp = config.lambda_gp
        self.lambda_id = config.lambda_id
        self.lambda_mask = config.lambda_mask
        self.lambda_msmall = config.lambda_msmall
        self.lambda_mzerone = config.lambda_mzerone
        self.mask_loss_mode = config.mask_loss_mode

        # Training configurations.
        self.dataset = config.dataset
        self.batch_size = config.batch_size
        self.num_iters = config.num_iters
        self.num_iters_decay = config.num_iters_decay
        self.g_lr = config.g_lr
        self.d_lr = config.d_lr
        self.n_critic = config.n_critic
        self.beta1 = config.beta1
        self.beta2 = config.beta2
        self.resume_iters = config.resume_iters
        # self.selected_attrs = config.selected_attrs

        # Test configurations.
        self.test_iters = config.test_iters

        # Miscellaneous.
        self.use_tensorboard = config.use_tensorboard
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Directories.
        self.log_dir = config.log_dir
        self.sample_dir = config.sample_dir
        self.model_save_dir = config.model_save_dir
        self.result_dir = config.result_dir

        # Step size.
        self.log_step = config.log_step
        self.sample_step = config.sample_step
        self.model_save_step = config.model_save_step
        self.lr_update_step = config.lr_update_step

        # Build the model and tensorboard.
        self.build_model()
        if self.use_tensorboard:
            self.build_tensorboard()

    def build_model(self):
        """Create a generator and a discriminator."""
        if self.dataset in ['Covid', 'BRATS', 'Directory']:
            self.G = Generator(self.g_conv_dim, 0, self.g_repeat_num)
            self.D = Discriminator(self.image_size, self.d_conv_dim, 0, self.d_repeat_num)

        self.g_optimizer = torch.optim.Adam(self.G.parameters(), self.g_lr, [self.beta1, self.beta2])
        self.d_optimizer = torch.optim.Adam(self.D.parameters(), self.d_lr, [self.beta1, self.beta2])
        self.print_network(self.G, 'G')
        self.print_network(self.D, 'D')
            
        self.G.to(self.device)
        self.D.to(self.device)

    def recreate_image(self, codebook, labels, w, h):
        """Recreate the (compressed) image from the code book & labels"""
        d = codebook.shape[1]
        image = np.zeros((w, h, d))
        label_idx = 0
        for i in range(w):
            for j in range(h):
                image[i][j] = codebook[labels[label_idx]]
                label_idx += 1
        return image

    def print_network(self, model, name):
        """Print out the network information."""
        num_params = 0
        for p in model.parameters():
            num_params += p.numel()
        print(model)
        print(name)
        print("The number of parameters: {}".format(num_params))

    def restore_model(self, resume_iters):
        """Restore the trained generator and discriminator."""
        print('Loading the trained models from step {}...'.format(resume_iters))
        G_path = os.path.join(self.model_save_dir, '{}-G.ckpt'.format(resume_iters))
        D_path = os.path.join(self.model_save_dir, '{}-D.ckpt'.format(resume_iters))
        self.G.load_state_dict(torch.load(G_path, map_location=lambda storage, loc: storage))

        # if D_path exists, load it
        if os.path.exists(D_path):
            self.D.load_state_dict(torch.load(D_path, map_location=lambda storage, loc: storage))

    def build_tensorboard(self):
        """Build a tensorboard logger."""
        from logger import Logger
        self.logger = Logger(self.log_dir)

    def update_lr(self, g_lr, d_lr):
        """Decay learning rates of the generator and discriminator."""
        for param_group in self.g_optimizer.param_groups:
            param_group['lr'] = g_lr
        for param_group in self.d_optimizer.param_groups:
            param_group['lr'] = d_lr

    def reset_grad(self):
        """Reset the gradient buffers."""
        self.g_optimizer.zero_grad()
        self.d_optimizer.zero_grad()

    def denorm(self, x):
        """Convert the range from [-1, 1] to [0, 1]."""
        out = (x + 1) / 2
        return out.clamp_(0, 1)

    def gradient_penalty(self, y, x):
        """Compute gradient penalty: (L2_norm(dy/dx) - 1)**2."""
        weight = torch.ones(y.size()).to(self.device)
        dydx = torch.autograd.grad(outputs=y,
                                   inputs=x,
                                   grad_outputs=weight,
                                   retain_graph=True,
                                   create_graph=True,
                                   only_inputs=True)[0]

        dydx = dydx.view(dydx.size(0), -1)
        dydx_l2norm = torch.sqrt(torch.sum(dydx**2, dim=1))
        return torch.mean((dydx_l2norm-1)**2)

    def label2onehot(self, labels, dim):
        """Convert label indices to one-hot vectors."""
        batch_size = labels.size(0)
        out = torch.zeros(batch_size, dim)
        out[np.arange(batch_size), labels.long()] = 1
        return out

    # def create_labels(self, c_org, c_dim=5, dataset='Covid', selected_attrs=None):
    #     """Generate target domain labels for debugging and testing."""
    #     # Get hair color indices.
    #     if dataset in ['Covid']:
    #         hair_color_indices = []
    #         for i, attr_name in enumerate(selected_attrs):
    #             if attr_name in ['Black_Hair', 'Blond_Hair', 'Brown_Hair', 'Gray_Hair']:
    #                 hair_color_indices.append(i)
    #
    #     c_trg_list = []
    #     for i in range(c_dim):
    #         if dataset in ['Covid']:
    #             c_trg = c_org.clone()
    #             if i in hair_color_indices:  # Set one hair color to 1 and the rest to 0.
    #                 c_trg[:, i] = 1
    #                 for j in hair_color_indices:
    #                     if j != i:
    #                         c_trg[:, j] = 0
    #             else:
    #                 c_trg[:, i] = (c_trg[:, i] == 0)  # Reverse attribute value.
    #         elif dataset == 'BRATS':
    #             c_trg = c_org.clone()
    #             c_trg[:, i] = (c_trg[:, i] == 0)  # Reverse attribute value.
    #         elif dataset == 'Directory':
    #             c_trg = self.label2onehot(torch.ones(c_org.size(0))*i, c_dim)
    #
    #         c_trg_list.append(c_trg.to(self.device))
    #     return c_trg_list

    def classification_loss(self, logit, target, dataset='Covid'):
        """Compute binary or softmax cross entropy loss."""
        if dataset in ['Covid', 'BRATS']:
            return F.binary_cross_entropy_with_logits(logit, target, size_average=False) / logit.size(0)
        elif dataset == 'Directory':
            return F.cross_entropy(logit, target)

    def mask_zero_one_criterion(self, mask, center=0.5, epsilon=0.01):
        base_loss = 1. / (center + epsilon)
        loss = torch.sum(1 / (torch.abs(mask - center) + epsilon)) / mask.numel()
        return loss - base_loss

    def mask_small_criterion_square(self, mask):
        return (torch.sum(mask) / mask.numel()) ** 2

    def mask_small_criterion_abs(self, mask):
        return torch.abs((torch.sum(mask))) / mask.numel()

    def mask_criterion_TV(self, mask):
        return (torch.sum(torch.abs(mask[:, :, 1:, :]-mask[:, :, :-1, :])) + \
               torch.sum(torch.abs(mask[:, :, :, 1:] - mask[:, :, :, :-1]))) / mask.numel()

    def train(self):
        """Train Fixed-Point GAN within a single dataset."""
        # Set data loader.
        if self.dataset in ['Covid', 'BRATS', 'Directory']:
            data_loader = self.data_loader

        # Fetch fixed inputs for debugging.
        data_iter = iter(data_loader)
        x_fixedA, x_fixedB = next(data_iter)
        x_fixedA = x_fixedA.to(self.device)
        x_fixedB = x_fixedB.to(self.device)
        # c_fixed_list = self.create_labels(c_org, self.c_dim, self.dataset, self.selected_attrs)

        # Learning rate cache for decaying.
        g_lr = self.g_lr
        d_lr = self.d_lr

        # Start training from scratch or resume training.
        start_iters = 0
        if self.resume_iters:
            start_iters = self.resume_iters
            self.restore_model(self.resume_iters)

        # Start training.
        print('Start training...')
        start_time = time.time()
        for i in range(start_iters, self.num_iters):

            # =================================================================================== #
            #                             1. Preprocess input data                                #
            # =================================================================================== #

            # Fetch real images and labels.
            try:
                x_realA, x_realB = next(data_iter)
            except:
                data_iter = iter(data_loader)
                x_realA, x_realB = next(data_iter)

            # # Generate target domain labels randomly.
            # rand_idx = torch.randperm(label_org.size(0))
            # label_trg = label_org[rand_idx]

            # if self.dataset in ['Covid', 'BRATS']:
            #     c_org = label_org.clone()
            #     c_trg = label_trg.clone()
            # elif self.dataset == 'Directory':
            #     c_org = self.label2onehot(label_org, self.c_dim)
            #     c_trg = self.label2onehot(label_trg, self.c_dim)

            x_realA = x_realA.to(self.device)           # Input images.
            x_realB = x_realB.to(self.device)           # Input images.
            # c_org = c_org.to(self.device)             # Original domain labels.
            # c_trg = c_trg.to(self.device)             # Target domain labels.
            # label_org = label_org.to(self.device)     # Labels for computing classification loss.
            # label_trg = label_trg.to(self.device)     # Labels for computing classification loss.

            # =================================================================================== #
            #                             2. Train the discriminator                              #
            # =================================================================================== #

            # Compute loss with real images.
            _, out_src = self.D(x_realB)
            d_loss_real = - torch.mean(out_src)
            # d_loss_cls = self.classification_loss(out_cls, label_org, self.dataset)

            # Compute loss with fake images.
            x_fakeB, mask = self.G(x_realA)
            x_fakeB, mask = torch.tanh(x_fakeB), torch.tanh(mask)
            mask = (mask+1.)/2.
            x_fakeB2 = x_fakeB * mask + x_realA * (1 - mask)
            # _, out_src = self.D(x_fakeB.detach())
            _, out_src2 = self.D(x_fakeB2.detach())
            # d_loss_fake = 0.5 * torch.mean(out_src) + 0.5 * torch.mean(out_src2)
            d_loss_fake =torch.mean(out_src2)

            # Compute loss for gradient penalty.
            alpha = torch.rand(x_realB.size(0), 1, 1, 1).to(self.device)
            # x_hat = (alpha * x_realB.data + (1 - alpha) * x_fakeB.data).requires_grad_(True)
            x_hat2 = (alpha * x_realB.data + (1 - alpha) * x_fakeB2.data).requires_grad_(True)
            # _, out_src = self.D(x_hat)
            _, out_src2 = self.D(x_hat2)
            # d_loss_gp = 0.5 * self.gradient_penalty(out_src, x_hat) + 0.5 * self.gradient_penalty(out_src2, x_hat2)
            d_loss_gp = self.gradient_penalty(out_src2, x_hat2)

            # Backward and optimize.
            # d_loss = d_loss_real + d_loss_fake + self.lambda_cls * d_loss_cls + self.lambda_gp * d_loss_gp
            d_loss = d_loss_real + d_loss_fake + self.lambda_gp * d_loss_gp
            self.reset_grad()
            d_loss.backward()
            self.d_optimizer.step()

            # Logging.
            loss = {}
            loss['D/loss_real'] = d_loss_real.item()
            loss['D/loss_fake'] = d_loss_fake.item()
            # loss['D/loss_cls'] = d_loss_cls.item()
            loss['D/loss_gp'] = d_loss_gp.item()
            
            # =================================================================================== #
            #                               3. Train the generator                                #
            # =================================================================================== #
            
            if (i+1) % self.n_critic == 0:
                # Original-to-target domain.
                x_fakeB, maskOT = self.G(x_realA)
                maskOT_max, maskOT_min = torch.max(maskOT), torch.min(maskOT)
                x_fakeB, maskOT = torch.tanh(x_fakeB), torch.tanh(maskOT)
                maskOT = (maskOT+1.)/2.
                x_fakeB2 = x_fakeB * maskOT + x_realA * (1 - maskOT)

                # _, out_src = self.D(x_fakeB)
                _, out_src2 = self.D(x_fakeB2)
                # g_loss_fake = - 0.5 * torch.mean(out_src) - 0.5 * torch.mean(out_src2)
                g_loss_fake = - torch.mean(out_src2)
                
                x_fakeA = x_realA * maskOT + x_fakeB * (1 - maskOT)
                g_loss_rec = torch.mean(torch.abs(x_realA - x_fakeA))

                maskOT_small_loss = self.mask_small_criterion_square(maskOT)
                maskOT_zo_loss = self.mask_zero_one_criterion(maskOT)

                g_mask_loss_OT = self.lambda_msmall * maskOT_small_loss + self.lambda_mzerone * maskOT_zo_loss
                

                # g_loss_cls = self.classification_loss(out_cls, label_trg, self.dataset)

                # Original-to-original domain.
                x_fakeB, maskOO = self.G(x_realB)
                maskOO_max, maskOO_min = torch.max(maskOO), torch.min(maskOO)
                x_fakeB, maskOO = torch.tanh(x_fakeB), torch.tanh(maskOO)
                maskOO = (maskOO+1.)/2.
                x_fakeB2 = x_fakeB * maskOO + x_realB * (1 - maskOO)

                # _, out_src = self.D(x_fakeB)
                _, out_src2 = self.D(x_fakeB2)
                # g_loss_fake_id = - 0.5 * torch.mean(out_src) - 0.5 * torch.mean(out_src2)
                g_loss_fake_id = - torch.mean(out_src2)
                # g_loss_cls_id = self.classification_loss(out_cls_id, label_org, self.dataset)
                g_loss_id = torch.mean(torch.abs(x_realB - x_fakeB))
                # g_loss_id = 0.5*torch.mean(torch.abs(x_realB - x_fakeB)) + 0.5*torch.mean(torch.abs(x_realB - x_fakeB2))
                # g_mask_loss += 0.5 * torch.sum(maskOO)

                maskOO_small_loss = self.mask_small_criterion_square(maskOO)
                maskOO_zo_loss = self.mask_zero_one_criterion(maskOO)

                g_mask_loss_OO = self.lambda_msmall * maskOO_small_loss + self.lambda_mzerone * maskOO_zo_loss

                # # Target-to-original domain.
                # delta_reconst = self.G(x_fake, c_org)
                # x_reconst = torch.tanh(x_fake + delta_reconst)
                # g_loss_rec = torch.mean(torch.abs(x_real - x_reconst))

                # # Original-to-original domain.
                # delta_reconst_id = self.G(x_fake_id, c_org)
                # x_reconst_id = torch.tanh(x_fake_id + delta_reconst_id)
                # g_loss_rec_id = torch.mean(torch.abs(x_real - x_reconst_id))

                # Backward and optimize.
                # g_loss_same = g_loss_fake_id + self.lambda_rec * g_loss_rec_id + self.lambda_cls * g_loss_cls_id + self.lambda_id * g_loss_id
                # g_loss = g_loss_fake + self.lambda_rec * g_loss_rec + self.lambda_cls * g_loss_cls + g_loss_same

                g_mask_loss = 0.5 * g_mask_loss_OT + 0.5 * g_mask_loss_OO
                g_loss = g_loss_fake + g_loss_fake_id + self.lambda_id * g_loss_id + self.lambda_rec * g_loss_rec + self.lambda_mask * g_mask_loss

                self.reset_grad()
                g_loss.backward()
                self.g_optimizer.step()

                # Logging.
                loss['G/loss_fake'] = g_loss_fake.item()
                loss['G/loss_rec'] = g_loss_rec.item()
                # loss['G/loss_cls'] = g_loss_cls.item()
                loss['G/loss_fake_id'] = g_loss_fake_id.item()
                # loss['G/loss_rec_id'] = g_loss_rec_id.item()
                # loss['G/loss_cls_id'] = g_loss_cls_id.item()
                loss['G/loss_id'] = g_loss_id.item()
                loss['G/loss_mask'] = g_mask_loss.item()


                loss['Mask/OT_min'] = maskOT_min.item()
                loss['Mask/OT_max'] = maskOT_max.item()
                loss['Mask/OT_small'] = maskOT_small_loss.item()
                loss['Mask/OT_zo'] = maskOT_zo_loss.item()


                loss['Mask/OO_min'] = maskOO_min.item()
                loss['Mask/OO_max'] = maskOO_max.item()
                loss['Mask/OO_small'] = maskOO_small_loss.item()
                loss['Mask/OO_zo'] = maskOO_zo_loss.item()

            # =================================================================================== #
            #                                 4. Miscellaneous                                    #
            # =================================================================================== #

            # Print out training information.
            if (i+1) % self.log_step == 0:
                et = time.time() - start_time
                et = str(datetime.timedelta(seconds=et))[:-7]
                log = "Elapsed [{}], Iteration [{}/{}]".format(et, i+1, self.num_iters)
                for tag, value in loss.items():
                    log += ", {}: {:.4f}".format(tag, value)
                print(log)

                if self.use_tensorboard:
                    for tag, value in loss.items():
                        self.logger.scalar_summary(tag, value, i+1)

            # Translate fixed images for debugging.
            if (i+1) % self.sample_step == 0:
                with torch.no_grad():
                    x_fake_list = [x_fixedA]
                    delta1, mask1 = self.G(x_fixedA)
                    delta1 = torch.tanh(delta1)
                    mask1 = torch.sigmoid(mask1)
                    x_fake_list.append(delta1)
                    delta1 = mask1 * delta1 + (1 - mask1) * x_fixedA
                    x_fake_list.append(delta1)
                    x_fake_list.append((mask1.repeat(1, 3, 1, 1) - 0.5) * 2.0)
                    x_fake_list.append(x_fixedB)
                    delta2, mask2 = self.G(x_fixedB)
                    delta2 = torch.tanh(delta2)
                    mask2 = torch.sigmoid(mask2)
                    x_fake_list.append(delta2)
                    delta2 = mask2 * delta2 + (1 - mask2) * x_fixedB
                    x_fake_list.append(delta2)
                    x_fake_list.append((mask2.repeat(1, 3, 1, 1) - 0.5) * 2.0)
                    x_concat = torch.cat(x_fake_list, dim=3)

                    # is_fid_model = is_fid_pytorch.ScoreModel(mode=2, stats_file='metrics/res/stats_pytorch/fid_stats_celeba.npz', cuda=self.device)
                    # is_mean, is_std, fid = is_fid_model.get_score_image_tensor(delta1, batch_size=self.batch_size//2)
                    # print(f'Validation FID = {fid}, mean = {is_mean}, std = {is_std}')

                    sample_path = os.path.join(self.sample_dir, '{}-images.jpg'.format(i+1))
                    save_image(self.denorm(x_concat.data.cpu()), sample_path, nrow=1, padding=0)
                    print('Saved real and fake images into {}...'.format(sample_path))

            # Save model checkpoints.
            if (i+1) % self.model_save_step == 0:
                G_path = os.path.join(self.model_save_dir, '{}-G.ckpt'.format(i+1))
                D_path = os.path.join(self.model_save_dir, '{}-D.ckpt'.format(i+1))
                torch.save(self.G.state_dict(), G_path)
                torch.save(self.D.state_dict(), D_path)
                print('Saved model checkpoints into {}...'.format(self.model_save_dir))

            # Decay learning rates.
            if (i+1) % self.lr_update_step == 0 and (i+1) > (self.num_iters - self.num_iters_decay):
                g_lr -= (self.g_lr / float(self.num_iters_decay))
                d_lr -= (self.d_lr / float(self.num_iters_decay))
                self.update_lr(g_lr, d_lr)
                print ('Decayed learning rates, g_lr: {}, d_lr: {}.'.format(g_lr, d_lr))


    def test(self):
        """Translate images using Fixed-Point GAN trained on a single dataset."""
        # Load the trained generator.
        self.restore_model(self.test_iters)
        
        # Set data loader.
        if self.dataset in ['Covid', 'Directory']:
            data_loader = self.data_loader
        
        with torch.no_grad():
            for i, (x_realA, x_realB) in enumerate(data_loader):

                # Prepare input images and target domain labels.
                x_realA = x_realA.to(self.device)
                x_realB = x_realB.to(self.device)
                # c_trg_list = self.create_labels(c_org, self.c_dim, self.dataset, self.selected_attrs)

                # # code for debugging
                # x_fakeB, maskOT = self.G(x_realA)
                # print("Img val range:", torch.min(x_fakeB), torch.max(x_fakeB))
                # print("Mask val range:", torch.min(maskOT), torch.max(maskOT))
                # exit()

                # Translate images.
                x_fake_list = [x_realA]
                fake, mask = self.G(x_realA)
                fake, mask = torch.tanh(fake), (torch.tanh(mask)+1.)/2.
                x_fake_list.append(fake)
                fake = mask * fake + (1 - mask) * x_realA
                x_fake_list.append(fake)
                x_fake_list.append((mask.repeat(1, 3, 1, 1)-0.5)*2)

                x_fake_list.append(x_realB)
                fake, mask = self.G(x_realB)
                fake, mask = torch.tanh(fake), (torch.tanh(mask)+1.)/2.
                x_fake_list.append(fake)
                fake = mask * fake + (1 - mask) * x_realB
                x_fake_list.append(fake)
                x_fake_list.append((mask.repeat(1, 3, 1, 1)-0.5)*2)

                # Save the translated images.
                x_concat = torch.cat(x_fake_list, dim=3)
                result_path = os.path.join(self.result_dir, '{}-images.jpg'.format(i+1))
                save_image(self.denorm(x_concat.data.cpu()), result_path, nrow=1, padding=0)
                print('Saved real and fake images into {}...'.format(result_path))

    def Find_Optimal_Cutoff(self, target, predicted):
        """ Find the optimal probability cutoff point for a classification model related to event rate
        Parameters
        ----------
        target : Matrix with dependent or target data, where rows are observations

        predicted : Matrix with predicted data, where rows are observations

        Returns
        -------     
        list type, with optimal cutoff value
            
        """
        fpr, tpr, threshold = roc_curve(target, predicted)
        i = np.arange(len(tpr)) 
        roc = pd.DataFrame({'tf' : pd.Series(tpr-(1-fpr), index=i), 'threshold' : pd.Series(threshold, index=i)})
        roc_t = roc.iloc[(roc.tf-0).abs().argsort()[:1]]

        return list(roc_t['threshold'])


    def testAUC(self):
        """Translate images using Fixed-Point GAN trained on a single dataset."""
        # Load the trained generator.
        self.restore_model(self.test_iters)

        from data_loader import get_loader

        meanp = []
        gt = []

        for gtv, modev in enumerate(['hea', 'ano']):
        
            # Set data loader.
            data_loader = get_loader(self.config.image_dir, self.config.image_size, self.config.batch_size,
                                       'TestValid', self.config.mode + modev, self.config.num_workers)
            
            with torch.no_grad():
                for i, x_realA in tqdm(enumerate(data_loader), total=len(data_loader)):
                    # Prepare input images and target domain labels.
                    x_realA = x_realA.to(self.device)
                    
                    gt += [gtv]*x_realA.shape[0]

                    # Translate images.
                    fake, mask = self.G(x_realA)
                    fake, mask = torch.tanh(fake), (torch.tanh(mask)+1.)/2.
                    fake = mask * fake + (1 - mask) * x_realA
                    diff = torch.abs(x_realA - fake)
                    diff /= 2.
                    diff = diff.data.cpu().numpy()
                    meanp += list(np.mean(diff, axis=(1,2,3)))

        thmean = self.Find_Optimal_Cutoff(gt, meanp)[0]

        print(f"Threshold: {thmean}")
        meanpth = (np.array(meanp)>=thmean)

        print(f"Unique: {np.unique(meanpth)}")
        print(f"Classification report:\n{classification_report(gt, meanpth)}\n")

        fpr, tpr, threshold = roc_curve(gt, meanp)
        tn, fp, fn, tp = confusion_matrix(gt, meanpth).ravel()
        specificity = tn / (tn+fp)
        sensitivity = tp / (tp+fn)
        meanauc = auc(fpr, tpr)

        print(f"Model Iter {self.test_iters} AUC: {round(meanauc, 2)}, SEN: {sensitivity}, SPEC: {specificity}")


    # def test2(self):
    #     """Translate images using Fixed-Point GAN trained on a single dataset."""
    #     # Load the trained generator.
    #     self.restore_model(self.test_iters)
    #
    #     # Set data loader.
    #     if self.dataset in ['Covid', 'Directory']:
    #         data_loader = self.data_loader
    #
    #     with torch.no_grad():
    #         for i, (x_realA, x_realB) in tqdm(enumerate(data_loader), total=len(data_loader)):
    #
    #             assert x_realA.shape[0] == 1
    #
    #             # Prepare input images and target domain labels.
    #             x_realA = x_realA.to(self.device)
    #             x_realB = x_realB.to(self.device)
    #             # c_trg_list = self.create_labels(c_org, self.c_dim, self.dataset, self.selected_attrs)
    #
    #             # # code for debugging
    #             # x_fakeB, maskOT = self.G(x_realA)
    #             # print("Img val range:", torch.min(x_fakeB), torch.max(x_fakeB))
    #             # print("Mask val range:", torch.min(maskOT), torch.max(maskOT))
    #             # exit()
    #
    #             # Translate images.
    #             x_fake_list = [x_realA]
    #             fake, mask = self.G(x_realA)
    #             fake, mask = torch.tanh(fake), (torch.tanh(mask)+1.)/2.
    #             # x_fake_list.append(fake)
    #             fake = mask * fake + (1 - mask) * x_realA
    #             x_fake_list.append(fake)
    #             x_fake_list.append(torch.abs(fake - x_realA)-1.)
    #             # x_fake_list.append((mask.repeat(1, 3, 1, 1)-0.5)*2)
    #
    #             diff = torch.abs(fake - x_realA)
    #             diff /= 2.
    #             diff = diff.data.cpu().numpy()
    #             p = np.mean(diff)
    #
    #             # Save the translated images.
    #             x_concat = torch.cat(x_fake_list, dim=3)
    #             result_path = os.path.join(self.result_dir, 'abnormal_{:.5f}_image_{}.jpg'.format(p, i+1))
    #             save_image(self.denorm(x_concat.data.cpu()), result_path, nrow=1, padding=0)
    #             # print('Saved real and fake images into {}...'.format(result_path))
    #
    #
    #
    #             x_fake_list = [x_realB]
    #             fake, mask = self.G(x_realB)
    #             fake, mask = torch.tanh(fake), (torch.tanh(mask)+1.)/2.
    #             # x_fake_list.append(fake)
    #             fake = mask * fake + (1 - mask) * x_realB
    #             x_fake_list.append(fake)
    #             x_fake_list.append(torch.abs(fake - x_realB)-1.)
    #             # x_fake_list.append((mask.repeat(1, 3, 1, 1)-0.5)*2)
    #
    #             diff = torch.abs(fake - x_realB)
    #             diff /= 2.
    #             diff = diff.data.cpu().numpy()
    #             p = np.mean(diff)
    #
    #             # Save the translated images.
    #             x_concat = torch.cat(x_fake_list, dim=3)
    #             result_path = os.path.join(self.result_dir, 'normal_{:.5f}_image_{}.jpg'.format(p, i+1))
    #             save_image(self.denorm(x_concat.data.cpu()), result_path, nrow=1, padding=0)
    #             # print('Saved real and fake images into {}...'.format(result_path))
    #
    #
    #
    # def testA2B(self):
    #     """Translate images using Fixed-Point GAN trained on a single dataset."""
    #     # Load the trained generator.
    #     self.restore_model(self.test_iters)
    #
    #     # Set data loader.
    #     if self.dataset in ['Covid', 'Directory']:
    #         data_loader = self.data_loader
    #
    #     with torch.no_grad():
    #         for i, (x_realA, x_realB) in enumerate(data_loader):
    #
    #             # Prepare input images and target domain labels.
    #             x_realA = x_realA.to(self.device)
    #             x_realB = x_realB.to(self.device)
    #             # c_trg_list = self.create_labels(c_org, self.c_dim, self.dataset, self.selected_attrs)
    #
    #             # # code for debugging
    #             # x_fakeB, maskOT = self.G(x_realA)
    #             # print("Img val range:", torch.min(x_fakeB), torch.max(x_fakeB))
    #             # print("Mask val range:", torch.min(maskOT), torch.max(maskOT))
    #             # exit()
    #
    #             # Translate images.
    #             x_fake_list = []
    #             fake, mask = self.G(x_realA)
    #             fake, mask = torch.tanh(fake), (torch.tanh(mask)+1.)/2.
    #             x_fake_list.append(fake)
    #
    #             # Save the translated images.
    #             x_concat = torch.cat(x_fake_list, dim=3)
    #             result_path = os.path.join(self.result_dir, '{}-images.png'.format(i+1))
    #             save_image(self.denorm(x_concat.data.cpu()), result_path, nrow=1, padding=0)
    #             print('Saved real and fake images into {}...'.format(result_path))
    #
    #
    # def testA2BM(self):
    #     """Translate images using Fixed-Point GAN trained on a single dataset."""
    #     # Load the trained generator.
    #     self.restore_model(self.test_iters)
    #
    #     # Set data loader.
    #     if self.dataset in ['Covid', 'Directory']:
    #         data_loader = self.data_loader
    #
    #     with torch.no_grad():
    #         for i, (x_realA, x_realB) in enumerate(data_loader):
    #
    #             # Prepare input images and target domain labels.
    #             x_realA = x_realA.to(self.device)
    #             x_realB = x_realB.to(self.device)
    #             # c_trg_list = self.create_labels(c_org, self.c_dim, self.dataset, self.selected_attrs)
    #
    #             # # code for debugging
    #             # x_fakeB, maskOT = self.G(x_realA)
    #             # print("Img val range:", torch.min(x_fakeB), torch.max(x_fakeB))
    #             # print("Mask val range:", torch.min(maskOT), torch.max(maskOT))
    #             # exit()
    #
    #             # Translate images.
    #             x_fake_list = []
    #             fake, mask = self.G(x_realA)
    #             fake, mask = torch.tanh(fake), (torch.tanh(mask)+1.)/2.
    #             fake = mask * fake + (1 - mask) * x_realA
    #             x_fake_list.append(fake)
    #
    #             # Save the translated images.
    #             x_concat = torch.cat(x_fake_list, dim=3)
    #             result_path = os.path.join(self.result_dir, '{}-images.png'.format(i+1))
    #             save_image(self.denorm(x_concat.data.cpu()), result_path, nrow=1, padding=0)
    #             print('Saved real and fake images into {}...'.format(result_path))



    # def test_brats(self):
    #     """Translate images using Fixed-Point GAN trained on a single dataset."""
    #     # Load the trained generator.
    #     self.restore_model(self.test_iters)
    #
    #     # Set data loader.
    #     if self.dataset in ['BRATS']:
    #         data_loader = self.data_loader
    #
    #     with torch.no_grad():
    #         for i, (x_real, c_org) in enumerate(data_loader):
    #             x_real = x_real.to(self.device)
    #
    #             c_trg = c_org.clone()
    #             c_trg[:, 0] = 0 # always to healthy
    #             c_trg_list = [c_trg.to(self.device)]
    #
    #             # Translate images.
    #             x_fake_list = [x_real]
    #             for c_trg in c_trg_list:
    #                 delta = self.G(x_real, c_trg)
    #                 delta_org = torch.abs(torch.tanh(delta + x_real) - x_real) - 1.0
    #                 delta_gray = np.mean(delta_org.data.cpu().numpy(), axis=1)
    #                 delta_gray_norm = []
    #
    #                 loc = []
    #                 cls_mul = []
    #
    #                 for indx in range(delta_gray.shape[0]):
    #                     temp = delta_gray[indx, :, :] + 1.0
    #                     tempimg_th = np.percentile(temp, 99)
    #                     tempimg = np.float32(temp >= tempimg_th)
    #                     temploc = np.reshape(tempimg, (self.image_size*self.image_size, 1))
    #
    #                     kmeans = KMeans(n_clusters=2, random_state=0).fit(temploc)
    #                     labels = kmeans.predict(temploc)
    #
    #                     recreated_loc = self.recreate_image(kmeans.cluster_centers_, labels, self.image_size, self.image_size)
    #                     recreated_loc = ((recreated_loc - np.min(recreated_loc)) / (np.max(recreated_loc) - np.min(recreated_loc)))
    #
    #                     loc.append(recreated_loc)
    #                     delta_gray_norm.append( tempimg )
    #
    #
    #                 loc = np.array(loc, dtype=np.float32)[:, :, :, 0]
    #                 delta_gray_norm = np.array(delta_gray_norm)
    #
    #                 loc = (loc * 2.0) - 1.0
    #                 delta_gray_norm = (delta_gray_norm * 2.0) - 1.0
    #
    #                 x_fake_list.append( torch.from_numpy(np.repeat(delta_gray[:, np.newaxis, :, :], 3, axis=1)).to(self.device) ) # difference map
    #                 x_fake_list.append( torch.from_numpy(np.repeat(delta_gray_norm[:, np.newaxis, :, :], 3, axis=1)).to(self.device) ) # localization thershold
    #                 x_fake_list.append( torch.from_numpy(np.repeat(loc[:, np.newaxis, :, :], 3, axis=1)).to(self.device) ) # localization kmeans
    #                 x_fake_list.append( torch.tanh(delta + x_real) ) # generated image
    #
    #             # Save the translated images.
    #             x_concat = torch.cat(x_fake_list, dim=3)
    #             result_path = os.path.join(self.result_dir, '{}-images.jpg'.format(i+1))
    #             save_image(self.denorm(x_concat.data.cpu()), result_path, nrow=1, padding=0)
    #             print('Saved real and fake images into {}...'.format(result_path))
