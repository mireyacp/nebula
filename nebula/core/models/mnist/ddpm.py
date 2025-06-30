import os
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.utils import save_image

from torch.optim.lr_scheduler import OneCycleLR

# Nebula
from nebula.core.models.nebulamodel import NebulaModel
from nebula.core.models.mnist.unet import Unet
from nebula.core.models.mnist.utils import ExponentialMovingAverage
from nebula.core.models.mnist.model import MNISTDiffusion

class MNISTDiffusionModel(NebulaModel):
    FINAL_SAMPLES_DIR = "finalSamples"
    GENERATED_SAMPLES_DIR = "samples"
    METRICS_IMAGES_DIR = "metrics"
    
    def __init__(
        self,
        input_channels=1,
        num_classes=10, 
        learning_rate=1e-3,
        confusion_matrix=None,
        seed=None, 
        # Diffusion
        batch_size=128, 
        epochs=150, 
        n_samples=36,
        model_base_dim=128,
        timesteps=1000, #Steps for diffusion proccess
        model_ema_steps=10,
        model_ema_decay=0.995,
        dim_mults=[2, 4],
        image_size=28,
        metrics = None,
        n_final_samples = 10000,
    ):
        super().__init__(input_channels, num_classes, learning_rate, metrics, confusion_matrix, seed)

        self.learning_rate = learning_rate
        self.criterion = nn.MSELoss(reduction='mean')
        self.timesteps = timesteps
        self.batch_size = batch_size
        self.epochs = epochs
        self.model_ema_steps = model_ema_steps 
        self.model_ema_decay = model_ema_decay
        self.n_samples = n_samples
        self.glo_steps = 0
        self.no_clip = False
        self.n_final_samples = n_final_samples
        self.generation_steps = 1000
        self.final_generated = False

        self.model = MNISTDiffusion(
            timesteps=self.timesteps,
            image_size=image_size,
            in_channels=self.input_channels,
            base_dim=model_base_dim,
            dim_mults=dim_mults,
        ).to(self.device)

        adjust = 1 * self.batch_size * self.model_ema_steps / self.epochs
        alpha = 1.0 - self.model_ema_decay
        alpha = min(1.0, alpha * adjust)
        self.model_ema = ExponentialMovingAverage(self.model, device=self.device, decay=1.0 - alpha)
    
    def forward(self, x, noise):
        #Return prediction    
        pred = self.model(x, noise)

        return pred
    
    def step(self, batch, batch_idx, phase):
        #Train/Test/Validation step
        image, target = batch
        noise = torch.randn_like(image).to(self.device)
        image = image.to(self.device)
        y_pred = self.forward(image, noise)
        loss = self.criterion(y_pred, noise)

        if phase == "Train":
            self.scheduler.step()

        if phase == "Train" and self.glo_steps % self.model_ema_steps == 0:
            self.model_ema.update_parameters(self.model)

        self.model_ema.eval()
        x = image[:self.n_samples]
        x_reconstructed, x_noisy = self.model_ema.module.reconstruct(
            x,
            noise[:self.n_samples],
            device=self.device
        )
        x = (x + 1) / 2  # [-1, 1] -> [0, 1]
        x = x.clamp(0.0, 1.0)

        if self.round == self.epochs - 1 and phase == "Train" and not self.final_generated:
            self.save_generated_images(self.n_final_samples, True, self.no_clip, phase)
            self.final_generated = True

        if self.glo_steps % self.generation_steps == 0:
            self.save_generated_images(self.n_samples, False, self.no_clip, phase)
            self.save_metrics_images(x, x_reconstructed, x_noisy, x.size(0), phase)

        if phase == "Train":
            self.model_ema.train()
       
        self.process_metrics(phase, x_reconstructed, x, loss)

        self.glo_steps += 1
        self._current_loss = loss
        return loss

    def configure_optimizers(self):
        """
        Set optimizer
        """
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.learning_rate)
        
        return self.optimizer 
    
    def save_generated_images(self, n_samples, final, no_clip, phase):
        """
        Save generated and real images in the specified paths.
        
        Args:
            n_samples (int): Number of samples to save.
            final (bool): True to generate final samples.
            no_clip (bool): Whether to disable clipped reverse diffusion.
            phase (str): Phase identifier (e.g., "train", "val", etc.).
        """
        if final:
            final_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), self.FINAL_SAMPLES_DIR)
            os.makedirs(final_path, exist_ok=True)

            for i in range(0, n_samples, self.batch_size):
                current_batch_size = min(self.batch_size, n_samples - i)
                
                # Generate batch
                samples_batch = self.model_ema.module.sampling(
                    current_batch_size,
                    clipped_reverse_diffusion=not no_clip,
                    device=self.device
                )
                samples_batch = samples_batch.clamp(0, 1)  # [0,1] 

                for j in range(current_batch_size):
                    img = samples_batch[j].detach().cpu()
                    save_image(img, os.path.join(final_path, f"final_samples_{i + j:05d}.png"))
                    del img

                del samples_batch
        else:
            path = os.path.join(os.path.dirname(os.path.abspath(__file__)), self.GENERATED_SAMPLES_DIR)
            os.makedirs(path, exist_ok=True)
            
            samples = self.model_ema.module.sampling(
                n_samples,
                clipped_reverse_diffusion=not no_clip,
                device=self.device
            )
            # Save generated samples
            samples = samples.clamp(0, 1)  # [0,1] 
            save_image(samples, os.path.join(path, f"{phase}_steps_{self.glo_steps:08d}.png"), nrow=int(math.sqrt(n_samples)))
            
    
    def save_metrics_images(self, real_images, reconstructed_images, noisy_images, n_samples, phase):
        path = os.path.join(os.path.dirname(os.path.abspath(__file__)), self.METRICS_IMAGES_DIR)
        os.makedirs(path, exist_ok=True)

        step_str = f"{self.glo_steps:08d}"

        real_images_path = os.path.join(path, f'{phase}_real_step{step_str}.png')
        save_image(real_images[:n_samples], real_images_path, nrow=int(math.sqrt(n_samples)))

        reconstructed_images_path = os.path.join(path, f'{phase}_reconstructed_step{step_str}.png')
        save_image(reconstructed_images[:n_samples], reconstructed_images_path, nrow=int(math.sqrt(n_samples)))

        noisy_images_path = os.path.join(path, f'{phase}_noisy_step{step_str}.png')
        save_image(noisy_images[:n_samples], noisy_images_path, nrow=int(math.sqrt(n_samples)))

    def on_train_start(self):
        super().on_train_start()
        dataloader = self.trainer.train_dataloader
        l = len(dataloader)
        self.scheduler = OneCycleLR(self.optimizer, self.learning_rate,
                                    total_steps = self.epochs * l,
                                    pct_start=0.25,anneal_strategy='cos')
