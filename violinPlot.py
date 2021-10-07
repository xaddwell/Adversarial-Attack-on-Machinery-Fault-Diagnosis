import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import matplotlib.ticker as ticker

data1=np.load(r"Target_Attack.npy")
data2=np.load(r"Untargeted_Attack.npy")
temp=np.array([np.append(data2[i],data1[i]) for i in range(len(data1))]).transpose()
# method1=[1]*32
# method2=[2]*32
# target1=[1]*32
# target2=[0]*32
# model1=[1]*32
# model2=[0]*32

# index=0
# d1,d2,d3,d4=[],[],[],[]
# data=[]
# for t in [target2,target1]:
#     for md in [method1,method2]:
#         for ml in [model2,model1]:
#             d1.extend(temp[index])
#             d2.extend(t)
#             d3.extend(md)
#             d4.extend(ml)
#             index+=1
#
# data=np.array([d1,d2,d3,d4]).transpose()
# for i in data:
#     print(i)
# temp=pd.DataFrame(data)
# temp.columns=['SuccessRate','Targeted','Method','Model']
# print(temp)
# # sns.boxplot(data=temp,notch=True)#,palette=sns.color_palette('pastel'),notch=True)
# sns.violinplot(x="Targeted",y="SuccessRate",data=temp)
# sns.swarmplot(data=temp,color="w", alpha=.9)
# plt.plot(self.SR[name][modelName],lw=1,marker='s',label=name+"-"+str(target)+"-"+modelName)
# plt.legend()
# plt.xlabel("Batchs(64/batch)",{'size':10})
# plt.ylabel("Attack Success Rate(%)",{'size':10})
# plt.xticks(fontsize=10)
# plt.yticks(fontsize=10)
# plt.tight_layout()
# plt.savefig("DefenseVisualization/"+"temp.png",dpi=600)

fig=plt.figure(dpi=210)
ax1 = plt.subplot(2,2,1)
plt.plot(temp[0],marker="s",markersize=2,lw=0.8,label='origin')
plt.plot(temp[1],marker="v",markersize=2,lw=0.8,label='defense')
plt.title("FGSM-Untargeted",{'size':8})
ax1.yaxis.set_major_locator(ticker.MultipleLocator(0.1))
ax1.xaxis.set_major_locator(ticker.MultipleLocator(5))
ax1.grid(axis='y',c='gray',ls='--',lw=0.4)
plt.xticks(fontsize=6)
plt.yticks(fontsize=6)
plt.ylabel('Attack Success Rate(%)',{'size':6})
plt.xlabel('Batchs(64/batch)',{'size':6})


ax2 = plt.subplot(2,2,2)
plt.plot(temp[2],marker="s",markersize=2,lw=0.8)
plt.plot(temp[3],marker="v",markersize=2,lw=0.8)
plt.title("PGD-Untargeted",{'size':8})
ax2.yaxis.set_major_locator(ticker.MultipleLocator(0.035))
ax2.xaxis.set_major_locator(ticker.MultipleLocator(5))
ax2.grid(axis='y',c='gray',ls='--',lw=0.4)
plt.xticks(fontsize=6)
plt.yticks(fontsize=6)
plt.ylabel('Attack Success Rate(%)',{'size':6})
plt.xlabel('Batchs(64/batch)',{'size':6})


ax3 = plt.subplot(2,2,3)
plt.plot(temp[4],marker="s",markersize=2,lw=0.8)
plt.plot(temp[5],marker="v",markersize=2,lw=0.8)
plt.title("FGSM-Targeted",{'size':8})
ax3.yaxis.set_major_locator(ticker.MultipleLocator(0.05))
ax3.xaxis.set_major_locator(ticker.MultipleLocator(5))
ax3.grid(axis='y',c='gray',ls='--',lw=0.4)
plt.xticks(fontsize=6)
plt.yticks(fontsize=6)
plt.ylabel('Attack Success Rate(%)',{'size':6})
plt.xlabel('Batchs(64/batch)',{'size':6})


ax4 = plt.subplot(2,2,4)
plt.plot(temp[6],marker="s",markersize=2,lw=0.8)
plt.plot(temp[7],marker="v",markersize=2,lw=0.8)
plt.title("PGD-Targeted",{'size':8})
ax4.yaxis.set_major_locator(ticker.MultipleLocator(0.2))
ax4.xaxis.set_major_locator(ticker.MultipleLocator(5))
ax4.grid(axis='y',c='gray',ls='--',lw=0.4)
plt.xticks(fontsize=6)
plt.yticks(fontsize=6)
plt.ylabel('Attack Success Rate(%)',{'size':6})
plt.xlabel('Batchs(64/batch)',{'size':6})

plt.tight_layout()
# lines, labels = fig.axes[-1].get_legend_handles_labels()
# fig.legend(loc='upper center')
plt.savefig("defense/pic.png",dpi=1000)
plt.show()


