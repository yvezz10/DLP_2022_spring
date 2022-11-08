import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_shape', default=64, type=int, help='image shape')
    parser.add_argument('--batch_size', default=32, type=int, help='batch size')
    parser.add_argument('--lr', default=0.00005, type=float, help='learning rate')
    parser.add_argument('--beta1', default=0.9, type=float, help='momentum term for adam')
    parser.add_argument('--clip_value', default=0.01, type=float, help='clip value of weight in WGAN')
    parser.add_argument('--mode', default='train', type=str, help='train or test mode')
    parser.add_argument('--latent_dim', default=64, type=int, help='latent dimension')
    parser.add_argument('--classes', default=24, type=int, help='classificaion classes')
    parser.add_argument('--emb_dim', default=24, type=int, help='embedding vector dimesion')
    parser.add_argument('--num_workers', default=4, type=int, help='number of workers')
    parser.add_argument('--epochs', default=1000, type=int, help='training epochs')
    parser.add_argument('--iter', default=10, type=int, help='iteration')
    parser.add_argument('--weight_dir', default='./weight', type=str, help='weight direction')
    parser.add_argument('--img_dir', default='./image', type=str, help='image saved direction')
    parser.add_argument('--log_dir', default='./log', type=str, help='log direction')
    parser.add_argument('--model_dir', default='', type=str, help='model saved direction')
    parser.add_argument('--last_e', default=0, type=int, help='last training epoch if continued')
    args = parser.parse_args()
    return args