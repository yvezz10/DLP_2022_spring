import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--lr', default=0.001, type=float, help='learning rate')
    parser.add_argument('--beta1', default=0.9, type=float, help='momentum term for adam')
    parser.add_argument('--batch_size', default=64, type=int, help='batch size')
    parser.add_argument('--num_classes', default=4, type=int, help='classification classes')
    parser.add_argument('--log_dir', default='./logs', help='base directory to save logs')
    parser.add_argument('--model_dir', default='', help='base directory to save logs')
    parser.add_argument('--data_root', default='./data', help='root directory for data')
    parser.add_argument('--optimizer', default='adam', help='optimizer to train with')
    parser.add_argument('--latent_dim', type=int, default=2048, help='latent dimention of encoder')   
    parser.add_argument('--epoch', type=int, default=10, help='epoch size')
    parser.add_argument('--seed', default=1, type=int, help='manual seed')
    parser.add_argument('--num_workers', type=int, default=4, help='number of data loading threads')
    parser.add_argument('--cuda', default=False, action='store_true')
    parser.add_argument('--testing', default=False, action='store_true') 
    args = parser.parse_args()
    return args
