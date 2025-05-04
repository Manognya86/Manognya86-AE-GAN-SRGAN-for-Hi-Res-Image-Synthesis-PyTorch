import os
import argparse
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.datasets as dset
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.utils import save_image
import logging
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
import shutil
from aegan_model import _netG1, _netD1, _netG2, _netRS, _RefinerD
from refiner_model import _RefinerG

class AEGANTrainer:
    def __init__(self, config):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        os.makedirs(config.outf, exist_ok=True)
        logging.basicConfig(
            filename=f'{config.outf}/training.log',
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.writer = SummaryWriter(log_dir=config.outf)

        # Initialize models
        self.netG1 = _netG1(config.nz, config.ngf, config.batch_size).to(self.device)
        self.netD1 = _netD1(config.ndf, config.nz).to(self.device)
        self.netG2 = _netG2(config.nz, config.ngf, config.nc, config.batch_size).to(self.device)
        self.netRS = _netRS(config.nc, config.ndf).to(self.device)
        self.RefinerG = _RefinerG(config.nc, config.ngf).to(self.device)
        self.RefinerD = _RefinerD(config.nc, config.ndf).to(self.device)

        # Optimizers
        self.optimizerG1 = optim.Adam(self.netG1.parameters(), lr=config.lr, betas=(config.beta1, 0.999))
        self.optimizerD1 = optim.Adam(self.netD1.parameters(), lr=config.lr, betas=(config.beta1, 0.999))
        self.optimizerG2 = optim.Adam(self.netG2.parameters(), lr=config.lrRS, betas=(config.beta1, 0.999))
        self.optimizerRS = optim.Adam(self.netRS.parameters(), lr=config.lrRS, betas=(config.beta1, 0.999))
        self.optimizerRefinerG = optim.Adam(self.RefinerG.parameters(), lr=config.lr, betas=(config.beta1, 0.999))
        self.optimizerRefinerD = optim.Adam(self.RefinerD.parameters(), lr=config.lr, betas=(config.beta1, 0.999))

        # Data loading
        transform = transforms.Compose([
            transforms.Resize(64),
            transforms.CenterCrop(64),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])
        self.dataset = dset.ImageFolder(root=config.dataroot, transform=transform)
        self.dataloader = DataLoader(self.dataset, batch_size=config.batch_size,
                                   shuffle=True, num_workers=min(config.workers, 2))
        
        self.fixed_noise = torch.randn(config.batch_size, config.nz, 1, 1, device=self.device)

    def train_stage1(self):
        logging.info("Starting Stage 1 training")
        for epoch in range(self.config.niter_stage1):
            for i, (real_images, _) in enumerate(self.dataloader):
                real_images = real_images.to(self.device)
                
                self.optimizerRS.zero_grad()
                self.optimizerG2.zero_grad()
                
                latent = self.netRS(real_images)
                reconstructed = self.netG2(latent)
                loss = F.l1_loss(reconstructed, real_images)
                
                loss.backward()
                self.optimizerRS.step()
                self.optimizerG2.step()

                if i % 50 == 0:
                    self.writer.add_scalar('Stage1/Loss', loss.item(), epoch*len(self.dataloader)+i)
                    logging.info(f"[Stage1][{epoch}/{self.config.niter_stage1}][{i}/{len(self.dataloader)}] Loss: {loss.item():.4f}")
                    
                    with torch.no_grad():
                        fake = self.netG2(self.netRS(real_images))
                    save_image(real_images, f"{self.config.outf}/real_samples.png", normalize=True)
                    save_image(fake, f"{self.config.outf}/fake_samples_epoch_{epoch}.png", normalize=True)

            if epoch % 10 == 0:
                self._save_checkpoint(epoch, "stage1")

        torch.save(self.netG2.state_dict(), f"{self.config.outf}/step1_netG2_last.pth")
        torch.save(self.netRS.state_dict(), f"{self.config.outf}/step1_netRS_last.pth")
        logging.info("Stage 1 training completed")

    def train_stage2(self):
        logging.info("Starting Stage 2 training")
        scaler = torch.cuda.amp.GradScaler()
        
        for epoch in range(self.config.niter_stage2):
            for i, (real_images, _) in enumerate(self.dataloader):
                real_images = real_images.to(self.device)
                batch_size = real_images.size(0)
                noise = torch.randn(batch_size, self.config.nz, 1, 1, device=self.device)
                
                chunk_size = min(16, batch_size)
                for chunk in range(0, batch_size, chunk_size):
                    chunk_end = min(chunk + chunk_size, batch_size)
                    chunk_real = real_images[chunk:chunk_end]
                    chunk_noise = noise[chunk:chunk_end]
                    
                    # Train D1
                    self.optimizerD1.zero_grad()
                    with torch.amp.autocast(device_type='cuda', dtype=torch.float16):
                        latent_real = self.netRS(chunk_real.detach())
                        output_real = self.netD1(latent_real)
                        errD_real = F.binary_cross_entropy_with_logits(output_real, torch.ones_like(output_real))
                        
                        fake_latent = self.netG1(chunk_noise.detach())
                        output_fake = self.netD1(fake_latent)
                        errD_fake = F.binary_cross_entropy_with_logits(output_fake, torch.zeros_like(output_fake))
                        errD = errD_real + errD_fake
                    
                    scaler.scale(errD).backward()
                    scaler.step(self.optimizerD1)
                    
                    # Train G1
                    self.optimizerG1.zero_grad()
                    with torch.amp.autocast(device_type='cuda', dtype=torch.float16):
                        fake_latent = self.netG1(chunk_noise)
                        output = self.netD1(fake_latent)
                        errG = F.binary_cross_entropy_with_logits(output, torch.ones_like(output))
                    
                    scaler.scale(errG).backward()
                    scaler.step(self.optimizerG1)
                    
                    # Train RefinerD
                    self.optimizerRefinerD.zero_grad()
                    with torch.amp.autocast(device_type='cuda', dtype=torch.float16):
                        output_real_ref = self.RefinerD(chunk_real)
                        errD_real_ref = F.binary_cross_entropy_with_logits(output_real_ref, torch.ones_like(output_real_ref))
                        
                        fake_images = self.netG2(self.netG1(chunk_noise.detach()))
                        refined = self.RefinerG(fake_images.detach())
                        output_fake_ref = self.RefinerD(refined)
                        errD_fake_ref = F.binary_cross_entropy_with_logits(output_fake_ref, torch.zeros_like(output_fake_ref))
                        errD_ref = errD_real_ref + errD_fake_ref
                    
                    scaler.scale(errD_ref).backward()
                    scaler.step(self.optimizerRefinerD)
                    
                    # Train RefinerG
                    self.optimizerRefinerG.zero_grad()
                    with torch.amp.autocast(device_type='cuda', dtype=torch.float16):
                        fake_images = self.netG2(self.netG1(chunk_noise))
                        refined = self.RefinerG(fake_images)
                        output_ref = self.RefinerD(refined)
                        errG_ref = F.binary_cross_entropy_with_logits(output_ref, torch.ones_like(output_ref)) + \
                                  self.config.lamb * F.l1_loss(refined, fake_images)
                    
                    scaler.scale(errG_ref).backward()
                    scaler.step(self.optimizerRefinerG)
                    
                    scaler.update()

                if i % 50 == 0:
                    logging.info(f"[Stage2][{epoch}/{self.config.niter_stage2}][{i}/{len(self.dataloader)}] "
                                f"D1: {errD.item():.4f} G1: {errG.item():.4f} "
                                f"RefD: {errD_ref.item():.4f} RefG: {errG_ref.item():.4f}")
                    self.writer.add_scalars('Stage2/Losses', {
                        'D1': errD.item(),
                        'G1': errG.item(),
                        'RefD': errD_ref.item(),
                        'RefG': errG_ref.item()
                    }, epoch*len(self.dataloader)+i)
                    
                    with torch.no_grad():
                        fake = self.netG2(self.netG1(noise[:8]))
                        refined = self.RefinerG(fake)
                    save_image(refined, f"{self.config.outf}/refined_samples_epoch_{epoch}.png", normalize=True)

            if epoch % 10 == 0:
                self._save_checkpoint(epoch, "stage2")

        torch.save(self.netG1.state_dict(), f"{self.config.outf}/netG1_final.pth")
        torch.save(self.RefinerG.state_dict(), f"{self.config.outf}/RefinerG_final.pth")
        logging.info("Stage 2 training completed")

    def _save_checkpoint(self, epoch, stage):
        """Save intermediate checkpoints"""
        checkpoint_dir = os.path.join(self.config.outf, "checkpoints")
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        torch.save({
            'epoch': epoch,
            'netG1': self.netG1.state_dict(),
            'netD1': self.netD1.state_dict(),
            'netG2': self.netG2.state_dict(),
            'netRS': self.netRS.state_dict(),
            'RefinerG': self.RefinerG.state_dict(),
            'RefinerD': self.RefinerD.state_dict(),
            'optimizerG1': self.optimizerG1.state_dict(),
            'optimizerD1': self.optimizerD1.state_dict(),
            'optimizerG2': self.optimizerG2.state_dict(),
            'optimizerRS': self.optimizerRS.state_dict(),
            'optimizerRefinerG': self.optimizerRefinerG.state_dict(),
            'optimizerRefinerD': self.optimizerRefinerD.state_dict(),
        }, os.path.join(checkpoint_dir, f"checkpoint_{stage}_epoch_{epoch}.pth"))

def create_new_version(out_dir):
    """Package trained models into versioned directory"""
    version = datetime.now().strftime("%Y%m%d-%H%M%S")
    model_dir = os.path.join("models", version)
    os.makedirs(model_dir, exist_ok=True)
    
    try:
        # Copy final models
        shutil.copy(os.path.join(out_dir, "step1_netG2_last.pth"), model_dir)
        shutil.copy(os.path.join(out_dir, "step1_netRS_last.pth"), model_dir)
        shutil.copy(os.path.join(out_dir, "netG1_final.pth"), model_dir)
        shutil.copy(os.path.join(out_dir, "RefinerG_final.pth"), model_dir)
        
        # Update latest symlink
        latest_path = os.path.join("models", "latest")
        if os.path.exists(latest_path):
            os.remove(latest_path)
        os.symlink(version, latest_path)
        print(f"Created new model version: {version}")
        return True
    except Exception as e:
        print(f"Version creation failed: {str(e)}")
        return False

def rollback_model():
    """Revert to previous model version"""
    models_dir = "models"
    if not os.path.exists(models_dir):
        print("No models directory found")
        return False

    versions = [d for d in os.listdir(models_dir) 
               if os.path.isdir(os.path.join(models_dir, d)) and d != "latest"]
    versions.sort(reverse=True)
    
    if len(versions) < 2:
        print("No previous version available for rollback")
        return False
    
    previous_version = versions[1]
    latest_path = os.path.join(models_dir, "latest")
    
    try:
        if os.path.exists(latest_path):
            os.remove(latest_path)
        os.symlink(previous_version, latest_path)
        print(f"Successfully rolled back to version: {previous_version}")
        return True
    except Exception as e:
        print(f"Rollback failed: {str(e)}")
        return False

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataroot', required=True)
    parser.add_argument('--workers', type=int, default=2)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--nz', type=int, default=100)
    parser.add_argument('--ngf', type=int, default=32)
    parser.add_argument('--ndf', type=int, default=32)
    parser.add_argument('--nc', type=int, default=3)
    parser.add_argument('--niter_stage1', type=int, default=50)
    parser.add_argument('--niter_stage2', type=int, default=200)
    parser.add_argument('--lr', type=float, default=0.0002)
    parser.add_argument('--lrRS', type=float, default=0.00001)
    parser.add_argument('--beta1', type=float, default=0.5)
    parser.add_argument('--outf', default='./pickle')
    parser.add_argument('--lamb', type=int, default=100)
    parser.add_argument('--manualSeed', type=int)
    parser.add_argument('--rollback', action='store_true', help='Rollback to previous model version')
    
    args = parser.parse_args()
    
    if args.rollback:
        rollback_model()
        return

    if args.manualSeed is None:
        args.manualSeed = random.randint(1, 10000)
    random.seed(args.manualSeed)
    torch.manual_seed(args.manualSeed)
    
    trainer = AEGANTrainer(args)
    print("Starting Stage 1 training...")
    trainer.train_stage1()
    print("Starting Stage 2 training...")
    trainer.train_stage2()
    print("Packaging trained models...")
    create_new_version(args.outf)
    print("Training complete! New model version deployed.")

if __name__ == "__main__":
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
    main()