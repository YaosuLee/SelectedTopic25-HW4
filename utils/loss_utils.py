import torch
import torch.nn as nn
from torch.nn.functional import mse_loss
from torchvision.models import vgg16
import torchvision.transforms as T

class GANLoss(nn.Module):
    def __init__(self, use_lsgan=True, target_real_label=1.0, target_fake_label=0.0,
                     tensor=torch.FloatTensor):
        super(GANLoss, self).__init__()
        self.real_label = target_real_label
        self.fake_label = target_fake_label
        self.real_label_var = None
        self.fake_label_var = None
        self.Tensor = tensor
        if use_lsgan:
            self.loss = nn.MSELoss()
        else:
            self.loss = nn.BCELoss()

    def get_target_tensor(self, input, target_is_real):
        target_tensor = None
        if target_is_real:
            create_label = ((self.real_label_var is None) or(self.real_label_var.numel() != input.numel()))
            # pdb.set_trace()
            if create_label:
                real_tensor = self.Tensor(input.size()).fill_(self.real_label)
                # self.real_label_var = Variable(real_tensor, requires_grad=False)
                # self.real_label_var = torch.Tensor(real_tensor)
                self.real_label_var = real_tensor
            target_tensor = self.real_label_var
        else:
            # pdb.set_trace()
            create_label = ((self.fake_label_var is None) or (self.fake_label_var.numel() != input.numel()))
            if create_label:
                fake_tensor = self.Tensor(input.size()).fill_(self.fake_label)
                # self.fake_label_var = Variable(fake_tensor, requires_grad=False)
                # self.fake_label_var = torch.Tensor(fake_tensor)
                self.fake_label_var = fake_tensor
            target_tensor = self.fake_label_var
        return target_tensor

    def __call__(self, input, target_is_real):
        target_tensor = self.get_target_tensor(input, target_is_real)
        # pdb.set_trace()
        return self.loss(input, target_tensor)

class CharbonnierLoss(nn.Module):
    """Charbonnier Loss (a differentiable variant of L1 loss)"""
    def __init__(self, epsilon=1e-3):
        super(CharbonnierLoss, self).__init__()
        self.epsilon = epsilon

    def forward(self, x, y):
        return torch.mean(torch.sqrt((x - y) ** 2 + self.epsilon ** 2))


class PerceptualLoss(nn.Module):
    def __init__(self, device):
        super().__init__()
        vgg = vgg16(pretrained=True).features[:16].eval().to(device)
        for param in vgg.parameters():
            param.requires_grad = False
        self.vgg = vgg
        self.transform = T.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    def forward(self, x, y):
        # Rescale to 3 channels if needed
        if x.shape[1] == 1:
            x = x.repeat(1, 3, 1, 1)
            y = y.repeat(1, 3, 1, 1)
        x = self.transform(x)
        y = self.transform(y)
        return nn.functional.l1_loss(self.vgg(x), self.vgg(y))

