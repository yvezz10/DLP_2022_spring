import itertools
import os
import random
from socket import AddressFamily

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from tqdm import tqdm

from args import parse_args
from dataset import bair_robot_pushing_dataset
from models.lstm import gaussian_lstm, lstm
from models.vgg_64 import vgg_decoder, vgg_encoder
from utils import init_weights, finn_eval_seq, kl_criterion, pred, plot_pred, record_param

torch.backends.cudnn.benchmark = True
mse_criterion = nn.MSELoss()

def train(x, cond, modules, optimizer, kl_anneal, args, teacher_forcing):

    modules['frame_predictor'].zero_grad()
    modules['posterior'].zero_grad()
    modules['encoder'].zero_grad()
    modules['decoder'].zero_grad()

    # initialize the hidden state.
    modules['frame_predictor'].hidden = modules['frame_predictor'].init_hidden()
    modules['posterior'].hidden = modules['posterior'].init_hidden()
    mse = 0
    kld = 0

    for i in range(1, args.n_past + args.n_future +5):
        ground_truth_img = x[:,i]
        previous_img = x[:, i-1]
        gt_latent = modules['encoder'](ground_truth_img)
        pr_latent = modules['encoder'](previous_img)

        use_teacher_forcing = True if random.random() < teacher_forcing else False
        
        ##encode to h
        if args.last_frame_skip or i< args.n_past:
            h = pr_latent[0]
            skip = pr_latent[1]
        else:
            h = pr_latent[0]
        
        if i <=args.n_past:
            h_pred = h
        
        ##get priority
        z_t, mu, logvar = modules['posterior'](gt_latent[0])

        ##LSTM output
        if use_teacher_forcing:
            g_pred = modules['frame_predictor'](torch.cat([h, z_t, cond[:,i]], 1))
        else:
            g_pred = modules['frame_predictor'](torch.cat([h_pred, z_t, cond[:,i]], 1))

        x_pred = modules['decoder']([g_pred, skip])
        h_pred = modules['encoder'](x_pred)[0]

        mse += mse_criterion(x_pred, x[:,i])
        kld += kl_criterion(mu, logvar, args)

    beta = kl_anneal.get_beta()
    loss = mse + kld * beta
    loss.backward()

    optimizer.step()

    return loss.detach().cpu().numpy() / (args.n_past + args.n_future +5), mse.detach().cpu().numpy() / (args.n_past + args.n_future+5), kld.detach().cpu().numpy() / (args.n_future + args.n_past+5), beta

class kl_annealing():
    def __init__(self, args):
        super().__init__()
        self.kl_anneal_cyclical = args.kl_anneal_cyclical
        self.beta = 0
        self.target_iter = args.niter*args.epoch_size/args.kl_anneal_cycle
        self.step = args.kl_anneal_ratio/self.target_iter
        self.current_iter = 0
    
    def update(self):
        if self.kl_anneal_cyclical:
            self.current_iter +=1
            self.beta += self.step
            self.beta = min([self.beta,1])
            if self.current_iter % self.target_iter == 0:
                self.beta = 0

        else:
            self.beta += self.step
            self.beta = min([self.beta,1])
    
    def get_beta(self):
        self.update()
        return self.beta


def main():
    args = parse_args()
    learning_rate = args.lr
    if args.cuda:
        assert torch.cuda.is_available(), 'CUDA is not available.'
        device = 'cuda:0'
    else:
        device = 'cpu'
    
    assert args.n_past + args.n_future <= 30 and args.n_eval <= 30
    assert 0 <= args.tfr and args.tfr <= 1
    assert 0 <= args.tfr_start_decay_epoch 
    assert 0 <= args.tfr_decay_step and args.tfr_decay_step <= 1

    test_flag = True if args.testing else False

    if args.model_dir != '':
        # load model and continue training from checkpoint
        saved_model = torch.load('%s/model.pth' % args.model_dir)
        optimizer = args.optimizer
        model_dir = args.model_dir
        niter = args.niter
        args = saved_model['args']
        args.optimizer = optimizer
        args.model_dir = model_dir
        args.log_dir = '%s/continued' % args.log_dir
        start_epoch = saved_model['last_epoch']
    else:
        name = 'rnn_size=%d-predictor-posterior-rnn_layers=%d-%d-n_past=%d-n_future=%d-lr=%.4f-g_dim=%d-z_dim=%d-last_frame_skip=%s-beta=%.7f'\
            % (args.rnn_size, args.predictor_rnn_layers, args.posterior_rnn_layers, args.n_past, args.n_future, args.lr, args.g_dim, args.z_dim, args.last_frame_skip, args.beta)

        args.log_dir = '%s/%s' % (args.log_dir, name)
        niter = args.niter
        start_epoch = 0

    os.makedirs(args.log_dir, exist_ok=True)
    os.makedirs('%s/gen/' % args.log_dir, exist_ok=True)

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    if os.path.exists('./{}/train_record.txt'.format(args.log_dir)):
        os.remove('./{}/train_record.txt'.format(args.log_dir))
    
    print(args)

    with open('./{}/train_record.txt'.format(args.log_dir), 'a') as train_record:
        train_record.write('args: {}\n'.format(args))

    # ------------ build the models  --------------

    if args.model_dir != '':
        frame_predictor = saved_model['frame_predictor']
        posterior = saved_model['posterior']
    else:
        frame_predictor = lstm(args.g_dim+args.z_dim+args.condition_len, args.g_dim, args.rnn_size, args.predictor_rnn_layers, args.batch_size, device)
        posterior = gaussian_lstm(args.g_dim, args.z_dim, args.rnn_size, args.posterior_rnn_layers, args.batch_size, device)
        frame_predictor.apply(init_weights)
        posterior.apply(init_weights)
            
    if args.model_dir != '':
        decoder = saved_model['decoder']
        encoder = saved_model['encoder']
    else:
        encoder = vgg_encoder(args.g_dim)
        decoder = vgg_decoder(args.g_dim)
        encoder.apply(init_weights)
        decoder.apply(init_weights)
    
    # --------- transfer to device ------------------------------------
    frame_predictor.to(device)
    posterior.to(device)
    encoder.to(device)
    decoder.to(device)

    # --------- load a dataset ------------------------------------
    train_data = bair_robot_pushing_dataset(args, 'train')
    validate_data = bair_robot_pushing_dataset(args, 'validate')
    test_data = bair_robot_pushing_dataset(args, 'test')

    train_loader = DataLoader(train_data,
                            num_workers=args.num_workers,
                            batch_size=args.batch_size,
                            shuffle=True,
                            drop_last=True,
                            pin_memory=True)
    train_iterator = iter(train_loader)

    validate_loader = DataLoader(validate_data,
                            num_workers=args.num_workers,
                            batch_size=args.batch_size,
                            shuffle=True,
                            drop_last=True,
                            pin_memory=True)

    validate_iterator = iter(validate_loader)

    test_loader = DataLoader(test_data,
                            num_workers=args.num_workers,
                            batch_size=args.batch_size,
                            shuffle=True,
                            drop_last=True,
                            pin_memory=True)

    test_iterator = iter(test_loader)

    # ---------------- optimizers ----------------
    if args.optimizer == 'adam':
        args.optimizer = optim.Adam
    elif args.optimizer == 'rmsprop':
        args.optimizer = optim.RMSprop
    elif args.optimizer == 'sgd':
        args.optimizer = optim.SGD
    else:
        raise ValueError('Unknown optimizer: %s' % args.optimizer)

    params = list(frame_predictor.parameters()) + list(posterior.parameters()) + list(encoder.parameters()) + list(decoder.parameters())
    optimizer = args.optimizer(params, lr=learning_rate, betas=(args.beta1, 0.999))

    kl_anneal = kl_annealing(args)
    
    modules = {
        'frame_predictor': frame_predictor,
        'posterior': posterior,
        'encoder': encoder,
        'decoder': decoder,
    }
    # --------- testing loop ------------------------------------
    if test_flag :

        frame_predictor.eval()
        encoder.eval()
        decoder.eval()
        posterior.eval()

        ##test psnr
        psnr_list = []
        for _ in range(len(test_data) // args.batch_size):
            try:
                test_seq, test_cond = next(test_iterator)
            except StopIteration:
                test_iterator = iter(test_loader)
                test_seq, test_cond = next(test_iterator)
                
            test_seq = test_seq.to(device)
            test_cond = test_cond.to(device)

            pred_seq = pred(test_seq, test_cond, modules, args, device, priority=False)

            test_seq = torch.transpose(test_seq, 0, 1)
            test_seq = test_seq[:args.n_past+args.n_future]

            _, _, psnr = finn_eval_seq(test_seq[args.n_past:], pred_seq[args.n_past:])
            psnr_list.append(psnr)

        ave_psnr = np.mean(np.concatenate(psnr_list))
        print("====================== average test psnr = {:.5f} ========================".format(ave_psnr))

        ##generate gif and frames jpeg
        try:
            test_seq, test_cond = next(test_iterator)
        except StopIteration:
            test_iterator = iter(test_loader)
            test_seq, test_cond = next(test_iterator)

        test_seq = test_seq.to(device)
        test_cond = test_cond.to(device)
        plot_pred(test_seq, test_cond, modules, 0, args, device, mode = 'test', priority=False)

        return

    # --------- training loop ------------------------------------
    loss_record = record_param('./loss_record')

    progress = tqdm(total=args.niter)
    best_val_psnr = 0
    tfr = args.tfr
    for epoch in range(start_epoch, start_epoch + niter):
        loss_record_list = []
        frame_predictor.train()
        posterior.train()
        encoder.train()
        decoder.train()

        epoch_loss = 0
        epoch_mse = 0
        epoch_kld = 0
        
        for _ in range(args.epoch_size):
            try:
                seq, cond = next(train_iterator)
            except StopIteration:
                train_iterator = iter(train_loader)
                seq, cond = next(train_iterator)
            
            seq = seq.to(device)
            cond = cond.to(device)
            
            loss, mse, kld, beta = train(seq, cond, modules, optimizer, kl_anneal, args, tfr)
            epoch_loss += loss
            epoch_mse += mse
            epoch_kld += kld
        
        if epoch >= args.tfr_start_decay_epoch:
            tfr = tfr - args.tfr_decay_step
            if tfr< args.tfr_lower_bound:
                tfr = args.tfr_lower_bound

        progress.update(1)
        with open('./{}/train_record.txt'.format(args.log_dir), 'a') as train_record:
            train_record.write(('[epoch: %02d] loss: %.5f | mse loss: %.5f | kld loss: %.5f\n' % (epoch, epoch_loss  / args.epoch_size, epoch_mse / args.epoch_size, epoch_kld / args.epoch_size)))
        

        loss_record_list = [epoch, epoch_loss, epoch_mse, epoch_kld, beta, tfr]
        loss_record.write_loss(loss_record_list)

        frame_predictor.eval()
        encoder.eval()
        decoder.eval()
        posterior.eval()

        if (epoch) % 5 == 0:
            psnr_list = []
            for _ in range(len(validate_data) // args.batch_size):
                try:
                    validate_seq, validate_cond = next(validate_iterator)
                except StopIteration:
                    validate_iterator = iter(validate_loader)
                    validate_seq, validate_cond = next(validate_iterator)
                
                validate_seq = validate_seq.to(device)
                validate_cond = validate_cond.to(device)

                pred_seq = pred(validate_seq, validate_cond, modules, args, device)

                validate_seq = torch.transpose(validate_seq, 0, 1)
                validate_seq = validate_seq[:args.n_past+args.n_future]

                _, _, psnr = finn_eval_seq(validate_seq[args.n_past:], pred_seq[args.n_past:])
                psnr_list.append(psnr)

            ave_psnr = np.mean(np.concatenate(psnr_list))

            psnr_record_list = [epoch, ave_psnr]
            loss_record.write_psnr(psnr_record_list)


            with open('./{}/train_record.txt'.format(args.log_dir), 'a') as train_record:
                train_record.write(('====================== validate psnr = {:.5f} ========================\n'.format(ave_psnr)))

            if ave_psnr > best_val_psnr :
                if ave_psnr > best_val_psnr:
                    best_val_psnr = ave_psnr
                # save the model
                torch.save({
                    'encoder': encoder,
                    'decoder': decoder,
                    'frame_predictor': frame_predictor,
                    'posterior': posterior,
                    'args': args,
                    'last_epoch': epoch},
                    '%s/model.pth' % args.log_dir)

        if epoch % 20 == 0:
            try:
                validate_seq, validate_cond = next(validate_iterator)
            except StopIteration:
                validate_iterator = iter(validate_loader)
                validate_seq, validate_cond = next(validate_iterator)

            validate_seq = validate_seq.to(device)
            validate_cond = validate_cond.to(device)

            plot_pred(validate_seq, validate_cond, modules, epoch, args, device, mode = 'validate', priority=True)

if __name__ == '__main__':
    main()
        
