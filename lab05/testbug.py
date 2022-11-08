from args import parse_args
from train_fixed_prior import kl_annealing
import matplotlib.pyplot as plt

args = parse_args()
kl_annel = kl_annealing(args)
betas = []

for i in range(args.niter):
    for j in range(args.epoch_size):
        beta = kl_annel.get_beta()
        betas.append(beta)

plt.plot(betas)
plt.show()