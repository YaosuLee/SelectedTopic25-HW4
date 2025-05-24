import wandb
import torchvision
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from utils.dataset_utils import PromptTrainDataset
from utils.loss_utils import PerceptualLoss, CharbonnierLoss
from net.model import PromptIR
from utils.schedulers import LinearWarmupCosineAnnealingLR
from options import options as opt
import lightning.pytorch as pl
from lightning.pytorch.loggers import WandbLogger,TensorBoardLogger
from lightning.pytorch.callbacks import ModelCheckpoint
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

def compute_psnr_ssim(pred, target):
    assert pred.shape == target.shape
    pred = torch.clamp(pred, 0, 1)
    target = torch.clamp(target, 0, 1)
    pred = pred.detach().cpu().numpy()
    target = target.detach().cpu().numpy()
    psnr_total, ssim_total = 0, 0
    for i in range(len(pred)):
        p = pred[i].transpose(1, 2, 0)
        t = target[i].transpose(1, 2, 0)
        psnr_total += peak_signal_noise_ratio(t, p, data_range=1.0)
        ssim_total += structural_similarity(t, p, data_range=1.0, win_size=5, channel_axis=-1)
    return psnr_total / len(pred), ssim_total / len(pred)

class PromptIRModel(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.net = PromptIR(decoder=True)
        self.loss_fn  = nn.L1Loss()
        # self.edge_loss_fn = edge_loss
        self.charb_loss = CharbonnierLoss()
        self.percep_loss = PerceptualLoss(self.device)
    def forward(self,x):
        return self.net(x)
    
    def training_step(self, batch, batch_idx):
        # training_step defines the train loop.
        # it is independent of forward
        ([clean_name, de_id], degrad_patch, clean_patch) = batch
        restored = self.net(degrad_patch)

        # l1 = self.loss_fn(restored, clean_patch)
        loss_charb = self.charb_loss(restored, clean_patch)
        loss_percep = self.percep_loss(restored, clean_patch)
        
        loss = loss_charb + 0.1 * loss_percep

        # Logging to TensorBoard (if installed) by default
        self.log("loss_total", loss, on_step=True, on_epoch=True, sync_dist=True)
        self.log("charbonnier_loss", loss_charb, on_step=True, on_epoch=True, sync_dist=True)
        self.log("perceptual_loss", loss_percep, on_step=True, on_epoch=True, sync_dist=True)
        
        # Log PSNR and SSIM
        psnr, ssim = compute_psnr_ssim(restored, clean_patch)
        self.log("psnr", psnr, on_step=True, on_epoch=True, sync_dist=True)
        self.log("ssim", ssim, on_step=True, on_epoch=True, sync_dist=True)

        # Log images every N steps or only first few batches
        if self.logger and batch_idx % 6000 == 0:
            # Convert to grid
            grid = torchvision.utils.make_grid(torch.cat([
                degrad_patch[:4],  # input
                restored[:4],      # output
                clean_patch[:4],   # ground truth
            ], dim=0), nrow=4, normalize=True, scale_each=True)

            # Log to wandb
            self.logger.experiment.log({
                "Sample Input/Output/GT": wandb.Image(grid),
                "global_step": self.global_step,
                "epoch": self.current_epoch
            })
        
        return loss
    
    def lr_scheduler_step(self,scheduler,metric):
        scheduler.step()
        lr = scheduler.get_last_lr()
    
    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), lr=2e-4)
        scheduler = LinearWarmupCosineAnnealingLR(optimizer=optimizer,warmup_epochs=15,max_epochs=150)

        return [optimizer],[scheduler]

def main():
    print("Options")
    print(opt)
    if opt.wblogger is not None:
        logger  = WandbLogger(project=opt.wblogger, name=opt.wandb_name)
    else:
        logger = TensorBoardLogger(save_dir = "logs/")

    ##### Dataset and DataLoader #####
    trainset = PromptTrainDataset(opt)
    checkpoint_callback = ModelCheckpoint(
        monitor="psnr",
        mode="max", 
        save_top_k=1,
        save_last=False, 
        dirpath=opt.ckpt_dir,
        filename=opt.ckpt_name 
    )

    trainloader = DataLoader(trainset, batch_size=opt.batch_size, pin_memory=True, shuffle=True,
                             drop_last=True, num_workers=opt.num_workers)
    
    ##### Model #####
    model = PromptIRModel()
    
    ##### Training #####
    trainer = pl.Trainer(max_epochs=opt.epochs, accelerator="gpu", devices=opt.num_gpus, \
                         strategy="ddp_find_unused_parameters_true", logger=logger, \
                         callbacks=[checkpoint_callback])
    trainer.fit(model=model, train_dataloaders=trainloader)

    ##### After training: print best PSNR #####
    print(f"Best PSNR: {checkpoint_callback.best_model_score.item():.4f}")
    print(f"Best model path: {checkpoint_callback.best_model_path}")
    
if __name__ == '__main__':
    main()
