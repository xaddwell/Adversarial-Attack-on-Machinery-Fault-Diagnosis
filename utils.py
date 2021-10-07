import os
import logging as lg
import time
import warnings
import torch
from torch import nn
from torch import optim
import models
import numpy as np
from datasets import Datasets
import attackMethod as am
import matplotlib.pyplot as plt
from tqdm import tqdm
import json
from confusionMatrix import confusionMatrix
from MyAttackModel import *
from distortionMeasure import batchDistortion
from bnVisualization import bnv
import seaborn as sns

class file(object):
    def __init__(self,name=None,path='tempFile',temp=None,Save=True,args=None):
        self.name=name
        self.path=path
        self.args=args
        if Save:
            temp=self.reshape(temp)
            self.saveAsfile(temp)

    def reshape(self,temp=None):
        if temp:
            a={}
            for modelName in self.args.attackModel:
                for phase in ["_train","_val","_test"]:
                    if phase=="_test":
                        a[modelName+phase]=temp[modelName][phase[1:]]
                    else:
                        a[modelName+phase]=[temp[modelName][str(epoch)][phase[1:]] for epoch in range(self.args.epochs)]
            temp=a
            return temp

    def saveAsfile(self,temp=None):
        temp=json.dumps(temp)
        f = open("tempFile/"+self.name+".json", "w")
        f.write(temp)
        f.close()


class train(object):
    def __init__(self,args,file):
        self.args=args
        self.file=file
    def setup(self):
        args=self.args

        if torch.cuda.is_available():
            self.device=torch.device("cuda")
            self.device_count=torch.cuda.device_count()
            lg.info('{} gpu available'.format(self.device_count))
            assert args.batch_size%self.device_count==0,"batch size should be divided by device count"
        else:
            warnings.warn("no gpu")
            self.device=torch.device("cpu")
            self.device_count=1
            lg.info('{} cpu available'.format(self.device_count))

        self.Dataset=Datasets(d_path=args.data_dir,length=args.sampleLength,number=args.sampleNumber,normal=args.isNormal,
                         rate=args.datasetRatio,enc=args.enc,enc_step=args.enc_step,data_type=args.data_name)

        self.datasets = {}
        self.datasets['train'],self.datasets['val'],self.datasets['test'] = self.Dataset.datasets

        self.dataloaders = {x: torch.utils.data.DataLoader(self.datasets[x], batch_size=args.batch_size,
                                                           shuffle=(True if x == 'train' else False),num_workers=args.num_workers,
                                                           pin_memory=(True if self.device == 'cuda' else False)) for x in ['train', 'val','test']}

        if args.lr_scheduler == 'step':
            steps = [int(step) for step in args.steps.split(',')]
            self.lr_scheduler = optim.lr_scheduler.MultiStepLR(self.optimizer, steps, gamma=args.gamma)
        elif args.lr_scheduler == 'exp':
            self.lr_scheduler = optim.lr_scheduler.ExponentialLR(self.optimizer, args.gamma)
        elif args.lr_scheduler == 'stepLR':
            steps = int(args.steps)
            self.lr_scheduler = optim.lr_scheduler.StepLR(self.optimizer, steps, args.gamma)
        elif args.lr_scheduler == 'fix':
            self.lr_scheduler = None
        else:
            raise Exception("no such lr_scheduler choice")

        self.criterion=nn.CrossEntropyLoss()
    def train(self):
        args = self.args
        epoch_acc = {}
        epoch_loss = {}
        num = {}
        sumT={}
        for modelName in args.attackModel:
            epoch_acc[modelName]={}
            epoch_loss[modelName]={}
            num[modelName]={}
            sumT[modelName]={}
            for epoch in range(args.epochs):
                epoch_acc[modelName][str(epoch)]={}
                epoch_loss[modelName][str(epoch)] = {}
                num[modelName][str(epoch)] = {}
                sumT[modelName][str(epoch)] = {}
                for phase in ['train', 'val']:
                    epoch_acc[modelName][str(epoch)][phase]=0
                    epoch_loss[modelName][str(epoch)][phase] = 0
                    num[modelName][str(epoch)][phase] = 0
                    sumT[modelName][str(epoch)][phase]=0

        for id,modelName in enumerate(args.attackModel):
            model=getattr(models, modelName)(in_channels=self.Dataset.inputchannel,n_class=self.Dataset.num_classes)

            if self.device_count > 1:
                self.model = torch.nn.DataParallel(model)
            if args.optim == "sgd":
                self.optimizer = optim.SGD(model.parameters(), lr=args.lr,
                                           momentum=args.momentum, weight_decay=args.weight_decay())
            elif args.optim == "adam":
                self.optimizer = optim.Adam(model.parameters(), lr=args.lr,
                                            weight_decay=args.weight_decay)
            else:
                raise Exception("no such optimizer choice")

            model.to(self.device)
            best_val_acc=0 #用来保存模型最高精度

            for epoch in range(args.epochs):
                lg.info(' '*5+'Epoch {}/{}'.format(epoch+1,args.epochs))
                if self.lr_scheduler is not  None:
                    lg.info('current lr {}'.format(self.lr_scheduler.get_lr()))
                else:
                    lg.info('current lr {}'.format(args.lr))

                for phase in ['train','val']:
                    start_time = time.time()
                    if phase == 'train' :
                        model.train()
                    else:
                        model.eval()

                    for batch_idx, (inputs, labels) in enumerate(self.dataloaders[phase]):
                        inputs =  torch.tensor(inputs,dtype=torch.float32).to(self.device)
                        labels = torch.tensor(labels,dtype=torch.float32).to(self.device)
                        with torch.set_grad_enabled(phase=='train'):
                            logits = model(inputs)
                            loss = self.criterion(logits, labels.long())
                            pred = logits.argmax(dim=1)
                            correct = torch.eq(pred, labels).float().sum().item()
                            loss_temp = loss.item() * inputs.size(0)

                            epoch_loss[modelName][str(epoch)][phase] += loss_temp
                            epoch_acc[modelName][str(epoch)][phase] += correct
                            num[modelName][str(epoch)][phase]+=len(labels)

                            # Calculate the training information
                            if phase == 'train':
                                self.optimizer.zero_grad()
                                loss.backward()
                                self.optimizer.step()

                    sumT[modelName][str(epoch)][phase]=time.time()-start_time

                epoch_loss[modelName][str(epoch)]['val'] = epoch_loss[modelName][str(epoch)]['val'] / num[modelName][str(epoch)]['val']
                epoch_acc[modelName][str(epoch)]['val'] = epoch_acc[modelName][str(epoch)]['val'] / num[modelName][str(epoch)]['val']
                epoch_loss[modelName][str(epoch)]['train'] = epoch_loss[modelName][str(epoch)]['train'] / num[modelName][str(epoch)]['train']
                epoch_acc[modelName][str(epoch)]['train'] = epoch_acc[modelName][str(epoch)]['train'] / num[modelName][str(epoch)]['train']

                print('Epoch: {} {}-Loss: {:.4f} {}-Acc: {:.4f} {}-Loss: {:.4f} {}-Acc: {:.4f} SumtimeCost {:.4f} sec'.format(
                        epoch + 1, 'train', epoch_loss[modelName][str(epoch)]['train'], 'train', epoch_acc[modelName][str(epoch)]['train'],
                    'val', epoch_loss[modelName][str(epoch)]['val'], 'val',epoch_acc[modelName][str(epoch)]['val'], time.time() - start_time), file=self.file)

                print('Epoch: {} {}-Loss: {:.4f} {}-Acc: {:.4f} {}-Loss: {:.4f} {}-Acc: {:.4f} SumtimeCost {:.4f} sec'.format(
                        epoch + 1, 'train', epoch_loss[modelName][str(epoch)]['train'], 'train',
                        epoch_acc[modelName][str(epoch)]['train'],
                        'val', epoch_loss[modelName][str(epoch)]['val'], 'val', epoch_acc[modelName][str(epoch)]['val'],
                        time.time() - start_time))

                if epoch_acc[modelName][str(epoch)]['val']>=best_val_acc:
                    best_val_acc=epoch_acc[modelName][str(epoch)]['val']
                    torch.save(model.state_dict(), r'modelSave/' + modelName + '.pth')

                if self.lr_scheduler is not None:
                    self.lr_scheduler.step()

            test_epoch_loss=0
            test_epoch_acc=0
            test_num=0

            for batch_idx, (inputs, labels) in tqdm(enumerate(self.dataloaders['test'])):
                inputs = torch.tensor(inputs, dtype=torch.float32).to(self.device)
                labels = torch.tensor(labels, dtype=torch.float32).to(self.device)
                # print(np.shape(inputs))
                logits = model(inputs)
                loss = self.criterion(logits, labels.long())
                pred = logits.argmax(dim=1)
                correct = torch.eq(pred, labels).float().sum().item()
                loss_temp = loss.item() * inputs.size(0)
                test_epoch_loss += loss_temp
                test_epoch_acc += correct
                test_num += len(inputs)

            print('{} Test loss {} acc {} '.format(modelName,test_epoch_loss/test_num,test_epoch_acc/test_num), file=self.file)
            epoch_loss[modelName]["test"]=test_epoch_loss/test_num
            epoch_acc[modelName]["test"] = test_epoch_acc / test_num

        tempSave1=file(name=modelName+"epoch_loss",temp=epoch_loss,args=args)
        tempSave2=file(name=modelName+"epoch_acc",temp=epoch_acc,args=args)
    #攻击函数
    def attack(self):
        args = self.args
        self.distortion,self.atkSR,self.SR,CM,adv_correct,sumT={},{},{},{},{},{}
        targetClass = args.targetClass
        #初始化
        for name in args.attackMethod:
            self.SR[name] = {}
            self.distortion[name]={}
            adv_correct[name]={}
            sumT[name]= {}
            self.atkSR[name]= {}
            CM[name]={}
            for modelName in args.attackModel:
                CM[name][modelName]=confusionMatrix(modelName=modelName,attackMethod=name,targeted=args.targeted,targetClass=targetClass)
                self.SR[name][modelName] = []
                adv_correct[name][modelName]=0
                sumT[name][modelName] = 0
                self.atkSR[name][modelName] = 0
                self.distortion[name][modelName] = []

        for batch_idx, (inputs, labels) in enumerate(self.dataloaders['test']):
            for modelName in args.attackModel:
                model = getattr(models, modelName)(in_channels=self.Dataset.inputchannel,n_class=self.Dataset.num_classes)
                model.load_state_dict(torch.load(r'modelSave/' + modelName + '.pth'))
                model.to(self.device)
                #bilstm时必须采用False,不然可能会报错
                if modelName == 'bilstm':
                    torch.backends.cudnn.enabled = False
                else:
                    torch.backends.cudnn.enabled = True

                # 定义目标类的labels
                targetLabels = torch.tensor([targetClass] * len(labels), dtype=torch.float32).to(self.device)
                inputs = torch.tensor(inputs, dtype=torch.float32).to(self.device)
                labels = torch.tensor(labels, dtype=torch.float32).to(self.device)
                print("originModelAcc:",torch.eq(model(inputs).argmax(dim=1), labels).float().sum().item() / len(labels))
                for name in args.attackMethod:
                    t1 = time.time()
                    #具体的某个攻击函数
                    atk = am.attack().get(name, model, args.params)
                    #目标攻击
                    if args.targeted:
                        if 'target' not in atk._supported_mode:
                            #设置为目标攻击模型模式
                            atk._supported_mode.append('targeted')
                            atk._targeted =True
                        # 以lamda形式输入
                        atk.set_mode_targeted_by_function(lambda images, labels: targetLabels.long())
                    else:
                        #非目标攻击则移除targeted选项
                        if 'target' in atk._supported_mode:
                            atk._supported_mode.remove('targeted')
                            atk._targeted=False

                    #得到对抗样本
                    adv= atk(inputs, labels.long())
                    #对抗样本输出
                    adv_logits = model(adv)
                    # 得到预测输出
                    adv_pred = adv_logits.argmax(dim=1)
                    #混淆矩阵数据输入
                    CM[name][modelName].set(labels.cpu(), adv_pred.cpu())
                    #能量损失度量
                    # print(adv.cpu())
                    distortion=batchDistortion(inputs.cpu(), adv.cpu())
                    self.distortion[name][modelName].append(distortion)
                    #bn可视化
                    # bnv(adv.cpu(),inputs.cpu(),alabel=2)
                    # print(labels,adv_pred)
                    #统计和目标标签正确的个数
                    if args.targeted:
                        add= torch.eq(adv_pred,targetLabels).float().sum().item()
                        # print("targeted")
                    else:
                        add = torch.eq(adv_pred, labels).float().sum().item()
                        # print("untargeted")
                    adv_correct[name][modelName]+=add
                    sr=add / len(labels) if args.targeted else 1 -add / len(labels)
                    self.SR[name][modelName].append(sr)
                    # print(self.SR[name][modelName])
                    target='target' if args.targeted else 'untargeted'
                    print('{}-{}-{}-Acc: {:.4f} distortion:{}'.format
                          (name, modelName, target,sr,distortion))
                    sumT[name][modelName] += time.time() - t1
                    # if modelName=='alexnet':
                    #     exit(0)
                    # 首先将要保存的tensor从计算图中分离出来，这里用到detach函数
                    # enc = adv.detach().cpu().numpy()
                    # np.savez("advs/"+target+"/"+name+"/"+modelName+"/batch-{}".format(batch_idx), enc)
                    ## np.savez_compressed("/test/enc_{}".format(batch_idx), codecs)  # 也可以用进行压缩存储

                print()


        for name in args.attackMethod:
            for modelName in args.attackModel:
                CM[name][modelName].show()
                self.atkSR[name][modelName] = adv_correct[name][modelName] / 2000 if args.targeted else 1 - adv_correct[name][modelName] / 2000
                print('{}-{}-{}-Acc: {:.4f} AttackTime {:.4f} sec Distortion:{} Best: {}'.format
                      (name,modelName,'Targeted' if args.targeted else 'Untargeted',self.atkSR[name][modelName],
                       sumT[name][modelName], np.mean(self.distortion[name][modelName]),np.max(self.SR[name][modelName])),file=self.file)
                print('{}-{}-{}-Acc: {:.4f} AttackTime {:.4f} sec Distortion:{}'.format
                      (name, modelName, 'Targeted' if args.targeted else 'Untargeted', self.atkSR[name][modelName],
                       sumT[name][modelName],np.mean(self.distortion[name][modelName])))

    def plotatkSR(self):
        args=self.args
        target='Target Attack' if args.targeted else 'Untargeted Attack'
        sns.set_style("darkgrid")
        temp=np.array([self.SR['fgsm']['alexnet'],self.SR['fgsm']['alexnet_batchnorm'],
                       self.SR['pgd']['alexnet'], self.SR['pgd']['alexnet_batchnorm']])
        np.save('defense/'+target+".npy",temp)
        # sns.boxplot(data=temp,notch=True)#,palette=sns.color_palette('pastel'),notch=True)
        sns.violinplot(data=temp,palette="Set3")

        sns.swarmplot(data=temp,color="w", alpha=.4)
        # plt.plot(self.SR[name][modelName],lw=1,marker='s',label=name+"-"+str(target)+"-"+modelName)
        plt.legend()
        plt.xlabel("Batchs(64/batch)",{'size':10})
        plt.ylabel("Attack Success Rate(%)",{'size':10})
        plt.xticks(fontsize=10)
        plt.yticks(fontsize=10)
        plt.tight_layout()
        plt.savefig("DefenseVisualization/"+str(args.attackMethod)+target+str(args.attackModel)+".png",dpi=600)
        plt.show()

    def plotLoss(self):
        args=self.args

        trainLoss={}
        valLoss={}
        trainAcc = {}
        valAcc = {}

        with open("tempFile/epoch_acc.json", 'r') as acc:
            epoch_acc = json.load(acc)
        with open("tempFile/epoch_loss.json", 'r') as loss:
            epoch_loss = json.load(loss)

        for modelName in args.attackModel:
            trainLoss[modelName]=epoch_loss[modelName+"_train"]
            valLoss[modelName] = epoch_loss[modelName+"_val"]
            trainAcc[modelName] = epoch_acc[modelName+"_train"]
            valAcc[modelName] = epoch_acc[modelName+"_val"]

        fig1 = plt.figure(figsize=(6, 3))  # 设置图大小 figsize=(6,3)
        for modelName in args.attackModel:
            plt.plot(trainLoss[modelName], label=modelName+'trainLoss')
        plt.legend(loc='best')
        fig2 = plt.figure(figsize=(6, 3))  # 设置图大小 figsize=(6,3)
        for modelName in args.attackModel:
            plt.plot(valLoss[modelName], label=modelName + 'valLoss')
        plt.legend(loc='best')
        fig3 = plt.figure(figsize=(6, 3))  # 设置图大小 figsize=(6,3)
        for modelName in args.attackModel:
            plt.plot(trainAcc[modelName], label=modelName + 'trainAcc')
        plt.legend(loc='best')
        fig4 = plt.figure(figsize=(6, 3))  # 设置图大小 figsize=(6,3)
        for modelName in args.attackModel:
            plt.plot(valAcc[modelName], label=modelName + 'valAcc')
        plt.legend(loc='best')
        plt.show()