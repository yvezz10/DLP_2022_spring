import math
from operator import pos
import imageio
import numpy as np
import matplotlib.pyplot as plt
import torch
from PIL import Image, ImageDraw
from scipy import signal
from skimage.metrics import peak_signal_noise_ratio as psnr_metric
from skimage.metrics import structural_similarity as ssim_metric
from torch.autograd import Variable
from torchvision import transforms
from torchvision.utils import save_image
import torch.nn as nn

import csv
import os


def kl_criterion(mu, logvar, args):
  # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
  KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
  KLD /= args.batch_size  
  return KLD
    
def eval_seq(gt, pred):
    T = len(gt)
    bs = gt[0].shape[0]
    ssim = np.zeros((bs, T))
    psnr = np.zeros((bs, T))
    mse = np.zeros((bs, T))
    for i in range(bs):
        for t in range(T):
            origin = gt[t][i]
            predict = pred[t][i]
            for c in range(origin.shape[0]):
                ssim[i, t] += ssim_metric(origin[c], predict[c]) 
                psnr[i, t] += psnr_metric(origin[c], predict[c])
            ssim[i, t] /= origin.shape[0]
            psnr[i, t] /= origin.shape[0]
            mse[i, t] = mse_metric(origin, predict)

    return mse, ssim, psnr

def mse_metric(x1, x2):
    err = np.sum((x1 - x2) ** 2)
    err /= float(x1.shape[0] * x1.shape[1] * x1.shape[2])
    return err

# ssim function used in Babaeizadeh et al. (2017), Fin et al. (2016), etc.
def finn_eval_seq(gt, pred):
    T = len(gt)
    bs = gt[0].shape[0]
    ssim = np.zeros((bs, T))
    psnr = np.zeros((bs, T))
    mse = np.zeros((bs, T))
    for i in range(bs):
        for t in range(T):
            origin = gt[t][i].detach().cpu().numpy()
            predict = pred[t][i].detach().cpu().numpy()
            for c in range(origin.shape[0]):
                res = finn_ssim(origin[c], predict[c]).mean()
                if math.isnan(res):
                    ssim[i, t] += -1
                else:
                    ssim[i, t] += res
                psnr[i, t] += finn_psnr(origin[c], predict[c])
            ssim[i, t] /= origin.shape[0]
            psnr[i, t] /= origin.shape[0]
            mse[i, t] = mse_metric(origin, predict)

    return mse, ssim, psnr

def finn_psnr(x, y, data_range=1.):
    mse = ((x - y)**2).mean()
    return 20 * math.log10(data_range) - 10 * math.log10(mse)

def fspecial_gauss(size, sigma):
    x, y = np.mgrid[-size // 2 + 1:size // 2 + 1, -size // 2 + 1:size // 2 + 1]
    g = np.exp(-((x ** 2 + y ** 2) / (2.0 * sigma ** 2)))
    return g / g.sum()

def finn_ssim(img1, img2, data_range=1., cs_map=False):
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)

    size = 11
    sigma = 1.5
    window = fspecial_gauss(size, sigma)

    K1 = 0.01
    K2 = 0.03

    C1 = (K1 * data_range) ** 2
    C2 = (K2 * data_range) ** 2
    mu1 = signal.fftconvolve(img1, window, mode='valid')
    mu2 = signal.fftconvolve(img2, window, mode='valid')
    mu1_sq = mu1*mu1
    mu2_sq = mu2*mu2
    mu1_mu2 = mu1*mu2
    sigma1_sq = signal.fftconvolve(img1*img1, window, mode='valid') - mu1_sq
    sigma2_sq = signal.fftconvolve(img2*img2, window, mode='valid') - mu2_sq
    sigma12 = signal.fftconvolve(img1*img2, window, mode='valid') - mu1_mu2

    if cs_map:
        return (((2 * mu1_mu2 + C1) * (2 * sigma12 + C2))/((mu1_sq + mu2_sq + C1) *
                    (sigma1_sq + sigma2_sq + C2)), 
                (2.0 * sigma12 + C2) / (sigma1_sq + sigma2_sq + C2))
    else:
        return ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) *
                    (sigma1_sq + sigma2_sq + C2))

def init_weights(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1 or classname.find('Linear') != -1:
        m.weight.data.normal_(0.0, 0.02)
        m.bias.data.fill_(0)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


def pred(validate_seq, validate_cond, modules, args, device, priority = True):
    
    with torch.no_grad():
        y_pred = torch.tensor([]).to(device)

        first_img = validate_seq[:,0]
        first_img = first_img.unsqueeze(0)
        y_pred = torch.cat((y_pred, first_img), 0)

        for i in range(1, args.n_past + args.n_future):
            ground_truth_img = validate_seq[:,i]
            previous_img = validate_seq[:, i-1]
            gt_latent = modules['encoder'](ground_truth_img)
            pr_latent = modules['encoder'](previous_img)

            if  i< args.n_past:
                h = pr_latent[0]
                skip = pr_latent[1]
            else:
                h = pr_latent[0]

            if priority:
                z_t, _, _ = modules['posterior'](gt_latent[0])
            else:
                z_t = torch.Tensor(args.batch_size, args.z_dim).normal_().float().to(device)

            if i<= args.n_past:
                g_pred = modules['frame_predictor'](torch.cat([h, z_t, validate_cond[:,i]], 1))

            else:
                g_pred = modules['frame_predictor'](torch.cat([h_pred, z_t, validate_cond[:,i]], 1))
        

            x_pred = modules['decoder']([g_pred, skip])
            h_pred = modules['encoder'](x_pred)[0]

            x_pred = x_pred.unsqueeze(0)
            y_pred = torch.cat((y_pred, x_pred), 0)
            #[12, batch, 3, 64, 64]
    
    return y_pred

def save_gif_and_jpg(filename, inputs, duration = 0.5):
    # inputs shape [12,3,64,64]
    images = []
    for tensor in inputs:
        img = tensor*255
        npImg = img.cpu().numpy().astype(np.uint8)
        npImg = np.transpose(npImg, (1,2,0))
        images.append(npImg)
    filename_gif = filename+'.gif'
    filename_jpg = filename+'.jpg'
    imageio.mimsave(filename_gif, images, duration = duration)
    images_seq = np.concatenate(images, 1)
    imageio.imwrite(filename_jpg, images_seq)

    
def plot_pred(validate_seq, validate_cond, modules, epoch, args, device, mode, priority: bool):
    y_pred = pred(validate_seq, validate_cond, modules, args, device, priority)

    to_plot_gt = validate_seq[0].squeeze(0)[:(args.n_past+args.n_future)]
    to_plot_pred = y_pred[:,0].squeeze(1)

    prior = '_with_prior' if priority else '_without_prior'
    
    pred_mode = 'val/' if mode =='validate' else 'test/'

    filename_gif_gt = './gen_gif/'+pred_mode +str(epoch)+ prior+ '_gt'
    filename_gif_pred = './gen_gif/'+pred_mode +str(epoch)+ prior+ '_pred'

    if not os.path.isdir('./gen_gif/'+pred_mode):
        os.makedirs('./gen_gif/'+pred_mode)
    save_gif_and_jpg(filename=filename_gif_pred, inputs = to_plot_pred)
    save_gif_and_jpg(filename=filename_gif_gt, inputs= to_plot_gt)


class record_param:

    def __init__(self, folder):
        """record [loss, mse, kld, beta, tfr], [psnr]"""
        self.record_path = folder
        if not os.path.isdir(self.record_path):
            os.makedirs(self.record_path)
        
        if not os.path.isdir(os.path.join(self.record_path,'loss.csv')):
            with open(os.path.join(folder,'loss.csv'), "w") as f:
                writer = csv.writer(f)
                header = ['epoch','loss', 'mse', 'kld', 'beta', 'tfr']
                writer.writerow(header)

        if not os.path.isdir(os.path.join(self.record_path,'psnr.csv')):
            with open(os.path.join(folder,'psnr.csv'), "w") as f:
                writer = csv.writer(f)
                header = ['epoch', 'ave_psnr']
                writer.writerow(header)

    def write_loss(self, loss):
        """loss = ['epoch', 'loss', 'mse', 'kld', 'beta', 'tfr']"""
        with open(os.path.join(self.record_path,'loss.csv'), 'a+') as f:
            writer = csv.writer(f)
            writer.writerow(loss)

    def write_psnr(self, psnr):
        with open(os.path.join(self.record_path,'psnr.csv'), "a+") as f:
            writer = csv.writer(f)
            writer.writerow(psnr)