import torch
import torch.nn as nn
import numpy as np
from ..attack import Attack


class LFGSM(Attack):
    def __init__(self, model, eps=[0.007]*10):
        super().__init__("FGSM", model)
        self.eps = eps
        self._supported_mode = ['default', 'targeted']

    def forward(self, images, labels):
        images = images.clone().detach().to(self.device)
        labels = labels.clone().detach().to(self.device)

        if self._targeted:
            target_labels = self._get_target_label(images, labels)

        loss = nn.CrossEntropyLoss()
        images.requires_grad = True
        outputs = self.model(images)
        eps=[self.eps[i] for i in labels]
        # Calculate loss
        if self._targeted:
            cost = -loss(outputs, target_labels)
        else:
            cost = loss(outputs, labels)
        # Update adversarial images
        grad = torch.autograd.grad(cost, images,retain_graph=False, create_graph=False)[0]
        print(grad)
        temp=torch.tensor([item.cpu().numpy()*eps[labels[id]]
                           for id,item in enumerate(grad.sign())]).to(self.device)

        adv_images = images + temp
        adv_images = torch.clamp(adv_images, min=0, max=1).detach()
        sign=[float(item.cpu().numpy().sum()) for item in grad]
        tag=torch.zeros(1,10)
        for id,num in enumerate(sign):
            if num>0:
                tag[0][labels[id]]+=1 if num>0 else -1
        tag=[1 if num>0 else -1 for num in tag[0]]
        self.eps=(1-np.array(tag)*0.01)*np.array(self.eps)
        # print(self.eps)

        return adv_images,self.eps
