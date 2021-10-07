import numpy as np
import itertools
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
lgd=['B007','B014','B021','IR007','IR014','IR021','OR007','OR014','OR021','Normal']

def plot_confusion_matrix(cm, classes,normalize=False,title='Confusion matrix',cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")


    plt.ylabel('Origin label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    # plt.savefig("confusionMat/" + title + ".png", dpi=600)
    plt.savefig("confusionMat/" + title + "_temp.png", dpi=600)
    plt.show()


class confusionMatrix(object):
    def __init__(self,modelName,attackMethod,targeted,targetClass):
        self.modelName=modelName
        self.attackMethod=attackMethod
        self.targeted=targeted
        self.targetClass=targetClass
        self.matrix=np.zeros((10,10))

    def set(self,a,b):
        for i in range(len(a)):
            self.matrix[int(a[i])][int(b[i])]+=1

    def show(self):
        # plt.rcParams['figure.figsize'] = (6.0, 5.0)  # 设置figure_size尺寸
        # plt.rcParams['image.interpolation'] = 'nearest'  # 设置 interpolation style
        plt.rcParams['image.cmap'] = 'gray'  # 设置 颜色 style
        # figsize(12.5, 4) # 设置 figsize
        plt.rcParams['savefig.dpi'] = 600  # 图片像素
        # plt.rcParams['figure.dpi'] = 160  # 分辨率
        # 默认的像素：[6.0,4.0]，分辨率为100，图片尺寸为 600&400
        # 指定dpi=200，图片尺寸为 1200*800
        # 指定dpi=300，图片尺寸为 1800*1200
        # 设置figsize可以在不改变分辨率情况下改变比例
        plt.figure(figsize=(7,7),dpi=120)
        # plt.subplots_adjust(left=0.093, right=0.979, top=0.948, bottom=0.310)
        target=str("Targeted("+str(lgd[self.targetClass])+") " if self.targeted else "Untargeted ")
        plot_confusion_matrix(self.matrix.astype(int), classes=lgd, normalize=False,
                              title=target+str(self.attackMethod)+"-"+str(self.modelName))







