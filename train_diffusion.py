""" Simple MNIST Diffusion Model, based on https://github.com/MaximeVandegar/Papers-in-100-Lines-of-Code/tree/main/Denoising_Diffusion_Probabilistic_Models

"""
import imageio
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torchvision import datasets, transforms, utils
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
import argparse

import os
from datetime import datetime
from common import utils as cu
from model import UNet
from scheduler import LinearNoiseScheduler


def train(args):
    device = args.device
    torch.manual_seed(1)

    model = UNet(in_channels=1)
    if args.checkpoint:
        print(f"Loading checkpoint from {args.checkpoint}")
        model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    model = model.to(device)

    log_dir = f"logs/mnist_diff-{model.__class__.__name__}-" + datetime.now().strftime("%Y%m%d-%H%M%S")
    os.makedirs(log_dir, exist_ok=True)
    writer = SummaryWriter(log_dir)

    batch_size = 64

    # Create the noise scheduler
    scheduler = LinearNoiseScheduler(num_timesteps=1000, device=device)
    schedule_plot = scheduler.plot()
    writer.add_figure("diffusion_schedule", plt.gcf(), 0)
    plt.close(schedule_plot)

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((32, 32)),
        lambda x: x * 2 - 1  # Scale to [-1, 1]
    ])
    mnist = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    loader = DataLoader(mnist, batch_size=batch_size, shuffle=True, num_workers=4)

    optimizer = torch.optim.Adam(model.parameters(), lr=2e-5)
    criterion = torch.nn.MSELoss()

    print(f"Number of parameters: {cu.count_parameters(optimizer)}")

    # Training
    epochs = 100
    for epoch_idx in range(epochs):
        model.train()
        epoch_loss = []
        for im, _ in tqdm(loader):
            optimizer.zero_grad()
            im = im.float().to(device)

            # Sample random noise
            noise = torch.randn_like(im).to(device)

            # Sample timestep
            t = torch.randint(0, 1000, (im.shape[0],)).to(device)

            # Add noise to images according to timestep
            noisy_im = scheduler.add_noise(im, noise, t)
            noise_pred = model(noisy_im, t)

            loss = criterion(noise_pred, noise)
            loss.backward()
            optimizer.step()

            epoch_loss.append(loss.item())

        writer.add_scalar("train/loss", np.mean(epoch_loss), epoch_idx)
        print(f"Epoch {epoch_idx + 1}/{epochs}, Loss: {np.mean(epoch_loss):.4f}")

        # _, x0_pred = scheduler.sample_prev_timestep(noise, noise_pred, torch.as_tensor(0.).to(device))
        # img_list =
        # writer.add_image("train/x0_pred", utils.make_grid(cu.range_2_1(x0_pred), nrow=4), epoch_idx)

        if epoch_idx % 5 == 0:
            with torch.no_grad():
                model.eval()
                samples = sample(model, scheduler, device)
                writer.add_image("samples", utils.make_grid(cu.range_2_1(samples[-1]), nrow=4), epoch_idx)
                video_writer = imageio.get_writer(os.path.join(log_dir, f"epoch-{epoch_idx + 1:03d}.mp4"), mode='I', fps=10, codec='libx264', quality=7)
                for frame in samples:
                    frame = cu.interpolate(frame[0:1], 256, antialias=True)[0]
                    frame = cu.range_2_255(frame).round().clamp(0, 255).to(torch.uint8).movedim(0, -1).cpu().numpy()
                    video_writer.append_data(frame)
                video_writer.close()

            # Save the model
            torch.save(model.state_dict(), os.path.join(log_dir, f"model-{epoch_idx + 1:03d}.pth"))

    writer.close()


@torch.no_grad()
def sample(model, scheduler, device):
    r"""
    Sample stepwise by going backward one timestep at a time.
    We save the x0 predictions
    """
    xt = torch.randn((8, 1, 32, 32), device=device)
    inference_timesteps = 1000
    samples = []
    for i in tqdm(reversed(range(inference_timesteps))):
        # Get prediction of noise
        noise_pred = model(xt, torch.as_tensor(i).unsqueeze(0).to(device))

        # Use scheduler to get x0 and xt-1
        xt, x0_pred = scheduler.sample_prev_timestep(xt, noise_pred, torch.as_tensor(i).to(device))

        ims = torch.clamp(xt, -1., 1.).detach().cpu()
        samples.append(ims)
    return samples


if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument('--device', type=str, default='cuda:0', help='Device to use for training')
    args.add_argument('--checkpoint', default=None, help='Path to checkpoint to load')
    args = args.parse_args()

    train(args)
