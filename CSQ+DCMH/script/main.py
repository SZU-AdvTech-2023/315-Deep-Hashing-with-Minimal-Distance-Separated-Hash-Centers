from torchcmh.run import run
import torch

torch.backends.cudnn.benchmark = True
'''
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"'''

if __name__ == '__main__':
    run()

