import os
import numpy as np
import errno
import torchvision.utils as vutils
from torch.utils.tensorboard import SummaryWriter
from IPython import display
from matplotlib import pyplot as plt
import torch
from datetime import datetime

'''
    TensorBoard Data will be stored in './runs' path
'''


class Logger:

    def __init__(self, model_name):
        self.model_name = model_name
        now = datetime.now().strftime("%d/%m %H:%M")
        
        comment = f"{self.model_name} {now}"
        
        logdir = f'runs/{comment}'

        self.writer = SummaryWriter(log_dir=logdir,comment=comment)

    def log_reward(self,episode,reward):
        self.writer.add_scalar('reward', reward, episode)
    
    def log_duration(self,episode,duration):
        self.writer.add_scalar('duration', duration, episode)
        
    def log_scalar(self,episode,scalar,name):
        self.writer.add_scalar(name,scalar,episode)

    def close(self):
        self.writer.close()
