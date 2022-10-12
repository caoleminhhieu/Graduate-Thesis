import os
import pprint
import random
import warnings
import torch
import numpy as np
from action import Trainer, Tester, Inference

from config import getConfig
warnings.filterwarnings('ignore')
args = getConfig()


def main(args):
    print('<---- Training Params ---->')
    pprint.pprint(args)

    # Random Seed
    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if use multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    if args.action == 'train':
        Trainer(args)

    elif args.action == 'test':

        datasets = ['ECSSD', 'DUTS-TE', 'DUT-O']
        # datasets = ['ECSSD']

        for dataset in datasets:
            args.dataset = dataset
            test_loss, test_mae, test_maxf, test_avgf, test_s_m = Tester(
                args).test()

            print(f'Test Loss:{test_loss:.3f} | MAX_F:{test_maxf:.4f} '
                  f'| AVG_F:{test_avgf:.4f} | MAE:{test_mae:.4f} | S_Measure:{test_s_m:.4f}')
    else:

        print('<----- Initializing inference mode ----->')
        Inference(args).test()


if __name__ == '__main__':
    main(args)
