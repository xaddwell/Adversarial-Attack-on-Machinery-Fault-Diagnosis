import torch
import torch.nn as nn
from ..attack import Attack
class LPGD(Attack):
    def __init__(self, model, eps=0.3,lr=0.05,
                 alpha=2/255, steps=40, random_start=True):
        super().__init__("PGD", model)
        self.lr=lr
        self.eps = eps
        self.alpha = alpha
        self.steps = steps
        self.random_start = random_start
        self._supported_mode = ['default', 'targeted']

    def forward(self, images, labels):
        images = images.clone().detach().to(self.device)
        labels = labels.clone().detach().to(self.device)
        if self._targeted:
            target_labels = self._get_target_label(images, labels)
        loss = nn.CrossEntropyLoss()
        adv_images = images.clone().detach()
        if self.random_start:
            # Starting at a uniformly random point
            adv_images = adv_images + torch.empty_like(adv_images).uniform_(-self.eps, self.eps)
            adv_images = torch.clamp(adv_images, min=0, max=1).detach()
        for _ in range(self.steps):
            eps = torch.tensor(self.eps, requires_grad=True).to(self.device)
            adv_images.requires_grad = True
            outputs = self.model(adv_images)
            # Calculate loss
            if self._targeted:
                cost = -loss(outputs, target_labels)
            else:
                cost = loss(outputs, labels)
            # Update adversarial images
            grad = torch.autograd.grad(cost, adv_images,retain_graph=False, create_graph=False)[0]
            adv_images = adv_images.detach() + self.alpha*grad.sign()
            delta = torch.clamp(adv_images - images, min=-self.eps, max=self.eps)
            adv_images = torch.clamp(images + delta, min=0, max=1).detach()
            eps = (1-grad.sign()*self.lr)*eps.detach()
            self.eps=torch.float(eps.detach())
        return adv_images
