import matplotlib.pyplot as plt
import torch.nn as nn
import torch
import numpy as np
from dataProcess import getTargetSample
def bnv(adv,org,alabel=2):
    # adv=adv[0]
    # org=org[0]
    plt.plot(adv[0][0][0:200],lw=1,label='adv')
    # plt.plot(org[0][0][0:200],lw=1,label='clean')
    plt.plot(getTargetSample(alabel)[0:200],lw=1,label='target_clean')
    # print(adv)
    # print(nn.BatchNorm1d(1)(adv)[0][0])
    plt.plot(nn.BatchNorm1d(1)(adv)[0][0][0:200].detach().numpy(), lw=1, label='adv_bn')
    # plt.plot(nn.BatchNorm1d(1)(org)[0][0][0:200].detach().numpy(), lw=1, label='clean_bn')
    plt.plot(nn.BatchNorm1d(1)(torch.tensor(np.array([[getTargetSample(alabel)]]),dtype=torch.float32))[0][0][0:200].detach().numpy(), lw=1, label='target_clean_bn')
    plt.legend()
    # plt.title()
    plt.show()

