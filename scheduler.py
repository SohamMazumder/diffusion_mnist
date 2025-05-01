import torch


class LinearNoiseScheduler:
    def __init__(self, num_timesteps, beta_start=1e-4, beta_end=0.0):
        self.num_timesteps = num_timesteps
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.betas = torch.linspace(beta_start, beta_end, num_timesteps)
        self.alphas = 1 - self.betas
        self.alpha_cumprod = torch.cumprod(self.alphas, dim=0)
        self.sqrt_alpha_cumprod = torch.sqrt(self.alpha_cumprod)
        self.sqrt_one_minus_alpha_cumprod = torch.sqrt(1 - self.alpha_cumprod)

    def add_noise(self, original_image, noise, t):
        """
        Adds noise to the original image based on the timestep t.
        Args:
            original_image (torch.Tensor): The original image tensor.
            noise (torch.Tensor): The noise tensor.
            t (int): The current timestep.
        Returns:
            torch.Tensor: The noisy image.
        """
        batch_size = original_image.shape[0]
        sqrt_alpha_cumprod_t = self.sqrt_alpha_cumprod[t].view(batch_size, 1, 1, 1)
        sqrt_one_minus_alpha_cumprod_t = self.sqrt_one_minus_alpha_cumprod[t].view(batch_size, 1, 1, 1)

        noisy_image = (sqrt_alpha_cumprod_t * original_image) + (sqrt_one_minus_alpha_cumprod_t * noise)
        return noisy_image

    def sample_prev_timestep(self, xt, noise_pred, t):
        """
        Samples the previous timestep from the current noisy image and noise prediction.
        Args:
            xt (torch.Tensor): The current noisy image.
            noise_pred (torch.Tensor): The predicted noise.
            t (int): The current timestep.
        Returns:
            torch.Tensor: The sampled previous image.
        """
        x0 = (xt - (self.sqrt_one_minus_alpha_cumprod[t] * noise_pred)) / self.sqrt_alpha_cumprod[t]
        x0 = x0.clamp(-1, 1)  # Ensure the values are within the valid range

        mean = xt - ((self.betas[t] * noise_pred)/self.sqrt_one_minus_alpha_cumprod[t])
        mean = mean / torch.sqrt(self.alphas[t])

        if t == 0:
            return mean, x0
        else:
            variance = (self.betas[t] * (1 - self.alpha_cumprod[t-1]))/(1 - self.alpha_cumprod[t])
            sigma = torch.sqrt(variance)
            z = torch.randn(xt.shape, device=xt.device)

        return mean + sigma*z, x0
