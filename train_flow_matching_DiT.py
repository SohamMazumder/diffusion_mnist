#!/usr/bin/env python3
"""
Training script for Flow Matching with Diffusion Transformers on MNIST
Based on Flow Matching algorithm (https://arxiv.org/abs/2210.02747)
"""

import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.utils import save_image, make_grid
import matplotlib.pyplot as plt
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime

from DiT import DiT

# Configuration
class Config:
    # Model parameters
    img_size = 28  # MNIST is 28x28
    in_channels = 1  # MNIST is grayscale
    patch_size = 2
    hidden_size = 384  # Reduced from original DiT for MNIST
    depth = 6  # Number of transformer blocks
    num_heads = 6
    time_emb_dim = 128
    
    # Training parameters
    batch_size = 128
    learning_rate = 1e-4
    num_epochs = 30
    weight_decay = 1e-5
    
    # Data and output directories
    data_dir = './data'
    output_dir = './outputs'
    checkpoint_dir = './checkpoints'
    
    # Flow matching parameters
    sigma_min = 0.002
    sigma_max = 80.0


def create_dirs(config):
    """Create necessary directories for outputs and checkpoints"""
    os.makedirs(config.output_dir, exist_ok=True)
    os.makedirs(config.checkpoint_dir, exist_ok=True)


def get_mnist_dataloaders(config):
    """Create MNIST dataloaders"""
    transform = transforms.Compose([
        transforms.Resize(config.img_size),
        transforms.ToTensor(),
        # Normalize to [-1, 1]
        transforms.Normalize((0.5,), (0.5,))
    ])
    
    train_dataset = datasets.MNIST(
        root=config.data_dir,
        train=True,
        download=True,
        transform=transform
    )
    
    test_dataset = datasets.MNIST(
        root=config.data_dir,
        train=False,
        download=True,
        transform=transform
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    return train_loader, test_loader


def sample_time(batch_size, eps=1e-5):
    """Sample uniform time steps in [eps, 1]"""
    return torch.rand(batch_size, device='cuda') * (1 - eps) + eps


def get_flow_matching_noise(x_0, x_1, t):
    """
    Compute the flow matching noise vector field
    Args:
        x_0: Starting point (Gaussian noise)
        x_1: Target point (real data)
        t: Time steps in [0, 1]
    """
    # Linear interpolation between noise and data
    x_t = t.view(-1, 1, 1, 1) * x_1 + (1 - t.view(-1, 1, 1, 1)) * x_0
    
    # The target vector field (ground truth) is simply (x_1 - x_0)
    # This represents the direction from noise to data
    vector_field = x_1 - x_0
    
    return x_t, vector_field


def train_epoch(model, train_loader, optimizer, device, epoch, config):
    """Train the model for one epoch"""
    model.train()
    total_loss = 0
    
    progress_bar = tqdm(train_loader)
    progress_bar.set_description(f"Epoch {epoch}")
    
    for batch_idx, (real_data, _) in enumerate(progress_bar):
        batch_size = real_data.size(0)
        real_data = real_data.to(device)
        
        # Sample Gaussian noise
        noise = torch.randn_like(real_data)
        
        # Sample time steps
        t = sample_time(batch_size)
        
        # Get flow matching inputs and targets
        x_t, vector_field = get_flow_matching_noise(noise, real_data, t)
        
        # Scale time steps for the model (from [0,1] to [0,1000])
        t_scaled = (t * 1000).to(device)
        
        # Forward pass
        optimizer.zero_grad()
        pred_noise = model(x_t.to(device), t_scaled)
        
        # Flow matching loss: predict the vector field
        loss = F.mse_loss(pred_noise, vector_field.to(device))
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        
        # Update progress bar
        progress_bar.set_postfix(loss=loss.item())
    
    return total_loss / len(train_loader)


def evaluate(model, test_loader, device, epoch, config):
    """Evaluate the model on the test set"""
    model.eval()
    total_loss = 0
    
    with torch.no_grad():
        for batch_idx, (real_data, _) in enumerate(test_loader):
            batch_size = real_data.size(0)
            real_data = real_data.to(device)
            
            # Sample Gaussian noise
            noise = torch.randn_like(real_data)
            
            # Sample time steps
            t = sample_time(batch_size)
            
            # Get flow matching inputs and targets
            x_t, vector_field = get_flow_matching_noise(noise, real_data, t)
            
            # Scale time steps for the model (from [0,1] to [0,1000])
            t_scaled = (t * 1000).to(device)
            
            # Forward pass
            pred_noise = model(x_t.to(device), t_scaled)
            
            # Flow matching loss
            loss = F.mse_loss(pred_noise, vector_field.to(device))
            
            total_loss += loss.item()
    
    avg_loss = total_loss / len(test_loader)
    print(f"Validation Loss: {avg_loss:.6f}")
    
    return avg_loss


def generate_samples(model, config, device, num_samples=16, nrow=4):
    """Generate samples using the trained model"""
    model.eval()
    
    with torch.no_grad():
        # Start with random noise
        x = torch.randn(num_samples, config.in_channels, config.img_size, config.img_size, device=device)
        
        # Flow-based generation with Euler solver
        num_steps = 100  # Number of integration steps
        
        for i in tqdm(range(num_steps, 0, -1), desc="Generating samples"):
            # Normalized time step from 0 to 1
            t = torch.ones(num_samples, device=device) * i / num_steps
            
            # Scale time steps for the model (from [0,1] to [0,1000])
            t_scaled = (t * 1000).to(device)
            
            # Predict vector field
            vector_field = model(x, t_scaled)
            
            # Euler step
            x = x + vector_field * (1 / num_steps)
            
        # Save the generated samples
        save_path = os.path.join(config.output_dir, f"samples_epoch.png")
        save_image(x * 0.5 + 0.5, save_path, nrow=nrow)
        
        return x


def main():
    config = Config()
    create_dirs(config)
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create TensorBoard writer
    log_dir = f"logs/mnist_DiT-" + datetime.now().strftime("%Y%m%d-%H%M%S")
    os.makedirs(log_dir, exist_ok=True)
    writer = SummaryWriter(log_dir)
    
    # Get dataloaders
    train_loader, test_loader = get_mnist_dataloaders(config)
    
    # Initialize the model
    model = DiT(
        img_size=config.img_size,
        patch_size=config.patch_size,
        in_channels=config.in_channels,
        hidden_size=config.hidden_size,
        depth=config.depth,
        num_heads=config.num_heads,
        time_emb_dim=config.time_emb_dim
    ).to(device)
    
    # Initialize optimizer
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay
    )
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, 
        T_max=config.num_epochs
    )
    
    # Training loop
    train_losses = []
    val_losses = []
    
    for epoch in range(1, config.num_epochs + 1):
        # Train
        train_loss = train_epoch(model, train_loader, optimizer, device, epoch, config)
        train_losses.append(train_loss)
        
        # Evaluate
        val_loss = evaluate(model, test_loader, device, epoch, config)
        val_losses.append(val_loss)
        
        # Log losses to TensorBoard
        writer.add_scalar("train/loss", train_loss, epoch)
        writer.add_scalar("val/loss", val_loss, epoch)
        
        # Generate samples
        if epoch % 5 == 0 or epoch == config.num_epochs:
            samples = generate_samples(model, config, device)
            
            # Log samples to TensorBoard
            writer.add_image("samples", make_grid(samples * 0.5 + 0.5, nrow=4), epoch)
            
            # Save checkpoint
            checkpoint_path = os.path.join(config.checkpoint_dir, f"model_epoch_{epoch}.pt")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss
            }, checkpoint_path)
            
            # Also save checkpoint to log_dir
            torch.save(model.state_dict(), os.path.join(log_dir, f"model-{epoch:03d}.pth"))
        
        # Step the scheduler
        scheduler.step()
    
    # Plot loss curves
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.savefig(os.path.join(config.output_dir, 'loss_curves.png'))
    
    # Close TensorBoard writer
    writer.close()
    
    print("Training complete!")


if __name__ == "__main__":
    main() 