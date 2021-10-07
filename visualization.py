import datasets as dt
import numpy as np
import matplotlib.pyplot as plt


lgd=['B007','B014','B021','IR007','IR014','IR021','OR007','OR014','OR021','Normal']



def PMCS():
    train, val, test = dt.Datasets(d_path=r"Data\0HP").datasets
    inputs, labels = [], []
    for item in train:
        if item[1] not in labels:
            inputs.append(item[0][0])
            labels.append(item[1])
        if len(labels) >= 10:
            break

    fig=plt.figure(figsize=(8,6),dpi=150)
    # p=[]
    for id, item in enumerate(inputs):
        # plt.subplot(5, 2, id + 1)
        plt.plot(item[700:],label="class "+str(lgd[id]))
    # plt.legend([[p[0],p[1],p[2],p[3],p[4]],[p[5],p[6],p[7],p[8],p[9]]],
    #            [[lgd[0],lgd[1],lgd[2],lgd[3],lgd[4]],[lgd[5],lgd[6],lgd[7],lgd[8],lgd[9]]],loc='upper right', fontsize=10)
    # plt.legend([p[0], p[1], p[2], p[3], p[4]],[lgd[0],lgd[1],lgd[2],lgd[3],lgd[4]],loc='upper right', fontsize=10)
    # plt.gca.
    plt.xlabel("Time(1/12kHZ)",{'size':15})
    plt.ylabel("Intensity",{'size':15})
    plt.grid()
    plt.legend(loc='upper center', ncol= 2)
    plt.tight_layout()
    plt.savefig('afterProcess.png', dpi=600)
    plt.show()

def OMCS():
    train, val, test = dt.Datasets(d_path=r"Data\0HP",normal=False).datasets
    inputs, labels = [], []
    for item in train:
        if item[1] not in labels:
            inputs.append(item[0][0])
            labels.append(item[1])
        if len(labels) >= 10:
            break

    f1,f2= plt.subplots(2, 5,dpi=150)
    for id, item in enumerate(inputs):
        ax=plt.subplot(2,5, id + 1)
        plt.plot(item[400:],linewidth=0.5)
        ax.set_title("class " + str(lgd[id]),fontsize=6)
        plt.tick_params(labelsize=5)
        # plt.legend(loc='upper right', fontsize=5)
    f1.suptitle("Origin motor current signal")
    plt.savefig('beforeProcess.png', dpi=600)
    plt.show()

PMCS()

