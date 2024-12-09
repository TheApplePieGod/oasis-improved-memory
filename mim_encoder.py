import torch
import torch.nn.functional as F
from vae import AutoencoderKL

class MiM_Encode(AutoencoderKL):
    def __init__(self, quantize_steps=10, input_width=16, input_height=16, **kwargs):
        self.downsampled_size = (input_width, input_height)
        self.quantize_steps = quantize_steps
        super().__init__(input_width=input_width, input_height=input_height, **kwargs)

    def autoencode(self, input, sample_posterior=True):
        q_input = self.quantize_image(input)
        super().autoencode(q_input, sample_posterior)

    def quantize_image(self, frame):
        _, _, H, W = frame.shape
        # new_size = (int(H * downsample), int(W * downsample))
        frame = F.interpolate(frame, self.downsampled_size, mode='bilinear', align_corners=False)
        
        step_size = 1 / self.quantize_steps
        frame = torch.round(frame / step_size) * step_size
        return frame
