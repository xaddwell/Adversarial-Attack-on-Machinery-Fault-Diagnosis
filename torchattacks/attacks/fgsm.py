import torch
import torch.nn as nn
from ..attack import Attack
class FGSM(Attack):
    def __init__(self, model, eps=0.007,minBound=0,maxBound=1):
        super().__init__("FGSM", model)
        self.min=minBound
        self.max=maxBound
        self.eps = eps*(maxBound-minBound)
        self._supported_mode = ['default', 'targeted']

    def forward(self, images, labels):
        r"""
        Overridden.
        """
        images = images.clone().detach().to(self.device)
        labels = labels.clone().detach().to(self.device)

        if self._targeted:
            target_labels = self._get_target_label(images, labels)

        loss = nn.CrossEntropyLoss()

        images.requires_grad = True
        outputs = self.model(images)

        # Calculate loss
        if self._targeted:
            cost = -loss(outputs, target_labels)
        else:
            cost = loss(outputs, labels)

        # Update adversarial images
        grad = torch.autograd.grad(cost, images,
                                   retain_graph=False, create_graph=False)[0]

        adv_images = images + self.eps*grad.sign()
        adv_images = torch.clamp(adv_images, min=self.min, max=self.max).detach()

        return adv_images
