import torch
import torch.nn as nn
import torch.nn.functional as F
from math import log2

class E_Conv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, gain=2):
        super().__init__()
        self.convolutional = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.a = (gain / (in_channels * (kernel_size ** 2))) ** 0.5
        self.biases = self.convolutional.bias
        self.convolutional.bias = None

        nn.init.normal_(self.convolutional.weight)
        nn.init.zeros_(self.biases)

    def forward(self, x):
        return self.convolutional(x * self.a) + self.biases.view(1, self.biases.shape[0], 1, 1)

class Pixel_Normalization(nn.Module):
    def __init__(self):
        super().__init__()
        self.epsilon = 1e-8

    def forward(self, x):
        return x / torch.sqrt(torch.mean(x ** 2, dim=1, keepdim=True) + self.epsilon)

class Convolutional_Block(nn.Module):
    def __init__(self, in_channels, out_channels, pixel_norm=True):
        super().__init__()
        self.convolutional_layer_1 = E_Conv2d(in_channels, out_channels)
        self.convolutional_layer_2 = E_Conv2d(out_channels, out_channels)
        self.activation = nn.LeakyReLU(0.2)
        self.normalization = Pixel_Normalization()
        self.pixel_norm = pixel_norm

    def forward(self, x):
        x = self.activation(self.convolutional_layer_1(x))
        x = self.normalization(x) if self.pixel_norm else x
        x = self.activation(self.convolutional_layer_2(x))
        x = self.normalization(x) if self.pixel_norm else x
        return x

class Generator(nn.Module):
    def __init__(self, z_dim, in_channels, img_channels=3):
        super().__init__()
        self.first = nn.Sequential(
            Pixel_Normalization(),
            nn.ConvTranspose2d(z_dim, in_channels, 4, 1, 0),
            nn.LeakyReLU(0.2),
            E_Conv2d(in_channels, in_channels),
            nn.LeakyReLU(0.2),
            Pixel_Normalization(),
        )
        self.rgb_i = E_Conv2d(in_channels, img_channels, kernel_size=1, padding=0)
        self.progressive_blocks = nn.ModuleList([])
        self.rgb_layers = nn.ModuleList([self.rgb_i])
        ratios = [1, 1, 1, 1, 1/2, 1/4, 1/8, 1/16, 1/32]
        for i in range(len(ratios) - 1):
            input_channels = int(in_channels * ratios[i])
            output_channels = int(in_channels * ratios[i + 1])
            self.progressive_blocks.append(Convolutional_Block(input_channels, output_channels))
            self.rgb_layers.append(E_Conv2d(output_channels, img_channels, kernel_size=1, padding=0))

        
    def fade_layer(self, a, upscaled, output):
        return torch.tanh(a * output + (1 - a) * upscaled)

    def forward(self, x, a, steps):
        output = self.first(x)

        if steps == 0:
            return self.rgb_i(output)

        for step in range(steps):
            upscaled = F.interpolate(output, scale_factor=2, mode="nearest")
            output = self.progressive_blocks[step](upscaled)

        last_upscale = self.rgb_layers[steps - 1](upscaled)
        last_output = self.rgb_layers[steps](output)
        return self.fade_layer(a, last_upscale, last_output)

class Discriminator(nn.Module):
    def __init__(self, z_dim, in_channels, img_channels=3):
        super().__init__()
        self.activation = nn.LeakyReLU(0.2)
        self.progressive_blocks = nn.ModuleList([])
        self.rgb_layers = nn.ModuleList([])
        ratios = [1, 1, 1, 1, 1/2, 1/4, 1/8, 1/16, 1/32]
        for i in range(len(ratios) - 1, 0, -1):
            input_channels = int(in_channels * ratios[i])
            output_channels = int(in_channels * ratios[i - 1])
            self.progressive_blocks.append(Convolutional_Block(input_channels, output_channels, pixel_norm=False))
            self.rgb_layers.append(E_Conv2d(img_channels, input_channels, kernel_size=1, padding=0))
        self.rgb_i = E_Conv2d(img_channels, in_channels, kernel_size=1, padding=0)
        self.rgb_layers.append(self.rgb_i)
        self.downscale = nn.AvgPool2d(kernel_size=2, stride=2)
        self.last = nn.Sequential(
            E_Conv2d(in_channels + 1, in_channels),
            nn.LeakyReLU(0.2),
            E_Conv2d(in_channels, in_channels, kernel_size=4, padding=0),
            nn.LeakyReLU(0.2),
            nn.Flatten(),
            nn.Linear(in_channels, 1),
        )
        self.last_2 = nn.Sequential(
            E_Conv2d(in_channels, in_channels),
            nn.LeakyReLU(0.2),
            E_Conv2d(in_channels, in_channels, kernel_size=4, padding=0),
            nn.LeakyReLU(0.2),
            nn.Flatten(),
            nn.Linear(in_channels, 1),
        )

    def fade_layer(self, a, downscaled, output):
        return a * output + (1 - a) * downscaled

    def minibatch_std(self, x):
        stats = (torch.std(x, dim=0).mean().repeat(x.shape[0], 1, x.shape[2], x.shape[3]))
        return torch.cat([x, stats], dim=1)

    def forward(self, x, a, steps):
        step = len(self.progressive_blocks) - steps
        output = self.activation(self.rgb_layers[step](x))

        if steps == 0:
            ouput = self.minibatch_std(output)
            return self.last_2(output).view(output.shape[0], -1)

        downscaled = self.activation(self.rgb_layers[step + 1](self.downscale(x)))
        output = self.downscale(self.progressive_blocks[step](output))
        output = self.fade_layer(a, downscaled, output)

        for i in range(step + 1, len(self.progressive_blocks)):
            output = self.progressive_blocks[i](output)
            output = self.downscale(output)

        output = self.minibatch_std(output)
        return self.last(output).view(output.shape[0], -1)

