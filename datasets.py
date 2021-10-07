from dataProcess import prepro
import numpy as np
import pickle

class Datasets(object):
    def __init__(self,d_path, length=864, number=1000, normal=True, rate=[0.6, 0.2, 0.2], enc=True, enc_step=28,data_type="DE"):
        datasets=prepro(d_path=d_path, length=length, number=number, normal=normal, rate=rate, enc=enc, enc_step=enc_step,data_type=data_type)
        self.inputchannel=1
        self.num_classes=10
        self.datasets=processData(datasets)

def processData(datasets):
    trX,trY,vX,vY,teX,teY=datasets
    train=[]
    test=[]
    val=[]
    for id,y in enumerate(trY):
        train.append((np.asarray([trX[id]]),trY[id]))
    for id,y in enumerate(vY):
        val.append((np.asarray([vX[id]]),vY[id]))
    for id,y in enumerate(teY):
        test.append((np.asarray([teX[id]]),teY[id]))
    return train,val,test


