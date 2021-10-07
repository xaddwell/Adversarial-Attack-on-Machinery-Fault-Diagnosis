import torchattacks.attacks as ta
import argparse
import os
from datetime import datetime
from utils import train
import numpy as np
from tqdm import tqdm

class attackParams():
    def __init__(self):
        self.params={'fgsm_eps':0.07,'fgsm_minBound':0,'fgsm_maxBound':1,
                     'pgd_eps':0.3,'pgd_alpha':2 / 255, 'pgd_steps':40, 'pgd_random_start':True,'pgd_minBound':0,'pgd_maxBound':1,
                     'deepfool_steps':50, 'deepfool_overshoot':0.02,'deepfool_minBound':0,'deepfool_maxBound':1,
                     'cw_c':1e1, 'cw_kappa':1, 'cw_steps':200, 'cw_lr':0.01,'cw_minBound':0,'cw_maxBound':1}
    def __getitem__(self, item):
        return self.params[item]

class ls():
    def __init__(self,attackMethod,attackModel,targeted,targetedClass=9):
        """
        :param attackMethod:
        :param attackModel:
        :param targeted:
        :param targetedClass: int 9 是正常的类,其他的全为不正常
        """
        self.attackMethod=attackMethod
        self.attackModel=attackModel
        self.targeted=targeted
        self.targetedClass=targetedClass

args = None

def parse_args(temp):
    parser = argparse.ArgumentParser(description='AdvGenerate')
    parser.add_argument('--attack',type=bool,choices=[True,False],default=True,help='attack or not')
    parser.add_argument('--attackMethod', type=str,choices=['fgsm','pgd','deepfool','cw'],default=temp.attackMethod, help='the adv_examples generated method')
    parser.add_argument('--attackModel', type=str, choices=['wdcnn','resnet','lenet','alexnet','cnn1d','bilstm','attresnet'],default=temp.attackModel, help='the name of the attacked model')
    parser.add_argument('--targeted',type=bool,default=temp.targeted,help="targeted attack or not")
    parser.add_argument('--targetClass',type=int,choices=[0,1,2,3,4,5,6,7,8,9],default=temp.targetedClass,help='the targeted class')
    parser.add_argument('--data_dir', type=str, default=r"Data\0HP", help='the directory of the data')
    parser.add_argument('--data_name',type=str,choices=['DE','FE','BA'],default='DE',help='the type of the data')
    parser.add_argument('--sampleLength',type=int,default=2048,help="the number of the sampling point")
    parser.add_argument('--sampleNumber',type=int,default=1000,help='the number of the samples in the total dataset')
    parser.add_argument('--isNormal',type=bool,default=True,help='do normalization or not')
    parser.add_argument('--datasetRatio',type=list,default=[0.6,0.2,0.2],help='the corresponding ratio of train val test')
    parser.add_argument('--enc',type=bool,default=True,help='do data enc or not')
    parser.add_argument('--enc_step',type=int,default=28,help='the enc_step')
    parser.add_argument('--cuda_device', type=str, default='0', help='cpu or gpu')
    parser.add_argument('--printInterval',type=int,default=100,help='the interval to print info')
    parser.add_argument('--batch_size', type=int, default=64, help='batchsize of the training process')
    parser.add_argument('--num_workers', type=int, default=0, help='the number of training process')
    parser.add_argument('--optim', type=str, choices=['sgd', 'adam'], default='adam', help='the optimizer')
    parser.add_argument('--params', type=attackParams, default=attackParams(), help="the hyper parameters of the attack method")
    parser.add_argument('--lr', type=float, default=0.001, help='the initial learning rate')
    parser.add_argument('--momentum', type=float, default=0.9, help='the momentum for sgd')
    parser.add_argument('--weight_decay', type=float, default=1e-5, help='the weight decay')
    parser.add_argument('--lr_scheduler', type=str, choices=['step', 'exp', 'stepLR', 'fix'], default='fix', help='the learning rate schedule')
    parser.add_argument('--gamma', type=float, default=0.1, help='learning rate scheduler parameter for step and exp')
    parser.add_argument('--steps', type=str, default='9', help='the learning rate decay for step and stepLR')
    parser.add_argument('--epochs', type=int, default=100, help='the number of epoch')
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    ATK=True
    targeted=True
    #['wdcnn','resnet','lenet','alexnet','cnn1d','bilstm','attresnet','alexnet_batchnorm','wdcnn_NoBN','resnet_NoBN','lenet_NoBN','cnn1d_NoBN','bilstm_NoBN']
    attackModel=['alexnet_batchnorm','alexnet']
    #['fgsm','pgd','deepfool','cw']
    attackMethod=['pgd','fgsm']
    args = parse_args(ls(attackMethod=attackMethod, attackModel=attackModel, targeted=targeted,targetedClass=2))
    os.environ['CUDA_VISIBLE_DEVICES'] = args.cuda_device.strip()
    f = open("log/"+"targeted={}-{}-{}1.txt".format(targeted,attackModel,attackMethod), "w")  # 打开文件以便写入
    trainer=train(args,file=f)
    trainer.setup()
    if ATK==False:
        trainer.train()
        trainer.plotLoss()
    else:
        trainer.attack()
        trainer.plotatkSR()

    f.close  # 关闭文件