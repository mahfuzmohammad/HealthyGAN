from model import Generator
from model import Discriminator
from torchvision.utils import save_image
import torch
import torch.nn.functional as F
import numpy as np
import os
import time
import datetime
from tqdm import tqdm
import pandas as pd
from sklearn.metrics import roc_curve, auc, classification_report, confusion_matrix


class Solver(object):
    """Solver for training and testing HealthyGAN."""

    def __init__(self, data_loader, config):
        """Initialize configurations."""

        # All config
        self.config = config

        # Data loader.
        self.data_loader = data_loader

        # Model configurations.
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
        """Train HealthyGAN within a single dataset."""
        # Set data loader.
        if self.dataset in ['Covid']:
            data_loader = self.data_loader

        # Fetch fixed inputs for debugging.
        data_iter = iter(data_loader)
        x_fixedA, x_fixedB = next(data_iter)
        x_fixedA = x_fixedA.to(self.device)
        x_fixedB = x_fixedB.to(self.device)

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

            x_realA = x_realA.to(self.device)           # Input images.
            x_realB = x_realB.to(self.device)           # Input images.

            # =================================================================================== #
            #                             2. Train the discriminator                              #
            # =================================================================================== #

            # Compute loss with real images.
            _, out_src = self.D(x_realB)
            d_loss_real = - torch.mean(out_src)

            # Compute loss with fake images.
            x_fakeB, mask = self.G(x_realA)
            x_fakeB, mask = torch.tanh(x_fakeB), torch.tanh(mask)
            mask = (mask+1.)/2.
            x_fakeB2 = x_fakeB * mask + x_realA * (1 - mask)
            _, out_src2 = self.D(x_fakeB2.detach())
            d_loss_fake =torch.mean(out_src2)

            # Compute loss for gradient penalty.
            alpha = torch.rand(x_realB.size(0), 1, 1, 1).to(self.device)
            x_hat2 = (alpha * x_realB.data + (1 - alpha) * x_fakeB2.data).requires_grad_(True)
            _, out_src2 = self.D(x_hat2)
            d_loss_gp = self.gradient_penalty(out_src2, x_hat2)

            # Backward and optimize.
            d_loss = d_loss_real + d_loss_fake + self.lambda_gp * d_loss_gp
            self.reset_grad()
            d_loss.backward()
            self.d_optimizer.step()

            # Logging.
            loss = {}
            loss['D/loss_real'] = d_loss_real.item()
            loss['D/loss_fake'] = d_loss_fake.item()
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

                _, out_src2 = self.D(x_fakeB2)
                g_loss_fake = - torch.mean(out_src2)
                
                x_fakeA = x_realA * maskOT + x_fakeB * (1 - maskOT)
                g_loss_rec = torch.mean(torch.abs(x_realA - x_fakeA))

                maskOT_small_loss = self.mask_small_criterion_square(maskOT)
                maskOT_zo_loss = self.mask_zero_one_criterion(maskOT)

                g_mask_loss_OT = self.lambda_msmall * maskOT_small_loss + self.lambda_mzerone * maskOT_zo_loss

                # Original-to-original domain.
                x_fakeB, maskOO = self.G(x_realB)
                maskOO_max, maskOO_min = torch.max(maskOO), torch.min(maskOO)
                x_fakeB, maskOO = torch.tanh(x_fakeB), torch.tanh(maskOO)
                maskOO = (maskOO+1.)/2.
                x_fakeB2 = x_fakeB * maskOO + x_realB * (1 - maskOO)

                _, out_src2 = self.D(x_fakeB2)
                g_loss_fake_id = - torch.mean(out_src2)
                g_loss_id = torch.mean(torch.abs(x_realB - x_fakeB))

                maskOO_small_loss = self.mask_small_criterion_square(maskOO)
                maskOO_zo_loss = self.mask_zero_one_criterion(maskOO)
                g_mask_loss_OO = self.lambda_msmall * maskOO_small_loss + self.lambda_mzerone * maskOO_zo_loss

                g_mask_loss = 0.5 * g_mask_loss_OT + 0.5 * g_mask_loss_OO
                g_loss = g_loss_fake + g_loss_fake_id + self.lambda_id * g_loss_id + self.lambda_rec * g_loss_rec + self.lambda_mask * g_mask_loss

                self.reset_grad()
                g_loss.backward()
                self.g_optimizer.step()

                # Logging.
                loss['G/loss_fake'] = g_loss_fake.item()
                loss['G/loss_rec'] = g_loss_rec.item()
                loss['G/loss_fake_id'] = g_loss_fake_id.item()
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
        """Translate images using HealthyGAN trained on a single dataset."""
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
        """Translate images using HealthyGAN trained on a single dataset."""
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
