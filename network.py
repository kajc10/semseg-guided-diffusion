'''
    Code for network modules is from  https://github.com/dome272/Diffusion-Models-pytorch/tree/main
'''

import torch
import torch.nn as nn
import torch.nn.functional as F


class EMA:
    def __init__(self, beta):
        super().__init__()
        self.beta = beta
        self.step = 0

    def update_model_average(self, ma_model, current_model):
        for current_params, ma_params in zip(current_model.parameters(), ma_model.parameters()):
            old_weight, up_weight = ma_params.data, current_params.data
            ma_params.data = self.update_average(old_weight, up_weight)

    def update_average(self, old, new):
        if old is None:
            return new
        return old * self.beta + (1 - self.beta) * new

    def step_ema(self, ema_model, model, step_start_ema=2000):
        if self.step < step_start_ema:
            self.reset_parameters(ema_model, model)
            self.step += 1
            return
        self.update_model_average(ema_model, model)
        self.step += 1

    def reset_parameters(self, ema_model, model):
        ema_model.load_state_dict(model.state_dict())

        

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None, residual=False):
        super().__init__()
        self.residual = residual
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(1, mid_channels),  #=LayerNorm?
            nn.GELU(),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(1, out_channels),
        )

    def forward(self, x):
        if self.residual:
            return F.gelu(x + self.double_conv(x))
        else:
            return self.double_conv(x)


class Down(nn.Module):
    def __init__(self, in_channels, out_channels, emb_dim=256):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, in_channels, residual=True),
            DoubleConv(in_channels, out_channels),
        )

        self.emb_layer = nn.Sequential(
            nn.SiLU(),
            nn.Linear(
                emb_dim,
                out_channels
            ),
        )

    def forward(self, x, t):
        device = x.device  # Get the device from the input tensor
        t = t.to(device)
        x = self.maxpool_conv(x)
        emb = self.emb_layer(t).to(device).unsqueeze(2).unsqueeze(3).expand(-1, -1, x.shape[2], x.shape[3])
        return x + emb
    

class Up(nn.Module):
    def __init__(self, in_channels, out_channels, emb_dim=256):
        super().__init__()

        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.conv = nn.Sequential(
            DoubleConv(in_channels, in_channels, residual=True),
            DoubleConv(in_channels, out_channels, in_channels // 2),
        )

        self.emb_layer = nn.Sequential(
            nn.SiLU(),
            nn.Linear(
                emb_dim,
                out_channels
            ),
        )

    def forward(self, x, skip_x, t):
        x = self.up(x)
        x = torch.cat([skip_x, x], dim=1)
        x = self.conv(x)
        device = x.device 
        t = t.to(device)
        emb = self.emb_layer(t).to(device).unsqueeze(2).unsqueeze(3).expand(-1, -1, x.shape[2], x.shape[3])
        return x + emb



class SelfAttention(nn.Module):
    def __init__(self, channels):
        super(SelfAttention, self).__init__()
        self.channels = channels        
        self.mha = nn.MultiheadAttention(channels, 4, batch_first=True)
        self.ln = nn.LayerNorm([channels])
        self.ff_self = nn.Sequential(
            nn.LayerNorm([channels]),
            nn.Linear(channels, channels),
            nn.GELU(),
            nn.Linear(channels, channels),
        )

    def forward(self, x):
        size = x.shape[-1]
        x = x.view(-1, self.channels, size * size).swapaxes(1, 2)
        x_ln = self.ln(x)
        attention_value, _ = self.mha(x_ln, x_ln, x_ln)
        attention_value = attention_value + x
        attention_value = self.ff_self(attention_value) + attention_value
        return attention_value.swapaxes(2, 1).view(-1, self.channels, size, size)


class SemsegUNet(nn.Module):
    def __init__(self, c_in=3, c_out=3, hidden_dim=128, emb_dim=256, num_classes=19):
        super().__init__()
        hidden_dim = int(hidden_dim)
        print('Initializing SemsegUNet with onehot encodings as conditioner')

        self.emb_dim = emb_dim
        self.num_classes = num_classes

        self.mask_conv = DoubleConv(num_classes, hidden_dim)
        
        self.inc = DoubleConv(c_in, hidden_dim)
        self.down1 = Down(hidden_dim, hidden_dim*2)
        self.down2 = Down(hidden_dim*2, hidden_dim*4)
        self.down3 = Down(hidden_dim*4, hidden_dim*4)

        self.bot1 = DoubleConv(hidden_dim*4, hidden_dim*8)
        self.bot2 = DoubleConv(hidden_dim*8, hidden_dim*8)
        self.bot3 = DoubleConv(hidden_dim*8, hidden_dim*4)

        self.up1 = Up(hidden_dim*8, hidden_dim*2)
        self.up2 = Up(hidden_dim*4, hidden_dim)
        self.up3 = Up(hidden_dim*2, hidden_dim)
        self.outc = nn.Conv2d(hidden_dim, c_out, kernel_size=1)

    def pos_encoding(self, t, channels):
        device = t.device
        inv_freq = 1.0 / (
            10000
            ** (torch.arange(0, channels, 2, device=device).float() / channels)
        )
        pos_enc_a = torch.sin(t.repeat(1, channels // 2) * inv_freq)
        pos_enc_b = torch.cos(t.repeat(1, channels // 2) * inv_freq)
        pos_enc = torch.cat([pos_enc_a, pos_enc_b], dim=-1)
        return pos_enc


    def forward(self, x, t, y):
        t = t.unsqueeze(-1).type(torch.float)
        t = self.pos_encoding(t, self.emb_dim) #b,256
        
        x1 = self.inc(x)
        if y is not None:
            x1 += self.mask_conv(y)

        x2 = self.down1(x1, t)
        x3 = self.down2(x2, t)
        x4 = self.down3(x3, t)

        x4 = self.bot1(x4)
        x4 = self.bot2(x4)
        self.features = self.bot3(x4)

        x = self.up1(self.features, x3, t)
        x = self.up2(x, x2, t)
        x = self.up3(x, x1, t)
        output = self.outc(x)
        return output

class SemsegUNetAttentionReduced(nn.Module):
    def __init__(self, c_in=3, c_out=3, hidden_dim=128, emb_dim=256, num_classes=19):
        super().__init__()
        hidden_dim = int(hidden_dim)
        print('Initializing SemsegUNetAttentionReduced with onehot encodings as conditioner')
        '''
        for now only 2 attention layers are utilized, the original design is left here, but commented out
        '''

        self.emb_dim = emb_dim
        self.num_classes = num_classes

        self.mask_conv = DoubleConv(num_classes, hidden_dim)
        
        self.inc = DoubleConv(c_in, hidden_dim)
        self.down1 = Down(hidden_dim, hidden_dim*2)
        #self.sa1 = SelfAttention(hidden_dim*2)
        self.down2 = Down(hidden_dim*2, hidden_dim*4)
        #self.sa2 = SelfAttention(hidden_dim*4)
        self.down3 = Down(hidden_dim*4, hidden_dim*4)
        self.sa3 = SelfAttention(hidden_dim*4)

        self.bot1 = DoubleConv(hidden_dim*4, hidden_dim*8)
        self.bot2 = DoubleConv(hidden_dim*8, hidden_dim*8)
        self.bot3 = DoubleConv(hidden_dim*8, hidden_dim*4)

        self.up1 = Up(hidden_dim*8, hidden_dim*2)
        self.sa4 = SelfAttention(hidden_dim*2)
        self.up2 = Up(hidden_dim*4, hidden_dim)
        #self.sa5 = SelfAttention(hidden_dim)
        self.up3 = Up(hidden_dim*2, hidden_dim)
        #self.sa6 = SelfAttention(hidden_dim)
        self.outc = nn.Conv2d(hidden_dim, c_out, kernel_size=1)

    def pos_encoding(self, t, channels):
        device = t.device
        inv_freq = 1.0 / (
            10000
            ** (torch.arange(0, channels, 2, device=device).float() / channels)
        )
        pos_enc_a = torch.sin(t.repeat(1, channels // 2) * inv_freq)
        pos_enc_b = torch.cos(t.repeat(1, channels // 2) * inv_freq)
        pos_enc = torch.cat([pos_enc_a, pos_enc_b], dim=-1)
        return pos_enc


    def forward(self, x, t, y):
        t = t.unsqueeze(-1).type(torch.float)
        t = self.pos_encoding(t, self.emb_dim) #b,256
        
        x1 = self.inc(x)
        if y is not None:
            x1 += self.mask_conv(y)

        x2 = self.down1(x1, t)
        #x2 = self.sa1(x2)
        x3 = self.down2(x2, t)
        #x3 = self.sa2(x3)
        x4 = self.down3(x3, t)
        x4 = self.sa3(x4)  

        x4 = self.bot1(x4)
        x4 = self.bot2(x4)
        self.features = self.bot3(x4)

        x = self.up1(self.features, x3, t)
        x = self.sa4(x)
        x = self.up2(x, x2, t)
        #x = self.sa5(x)
        x = self.up3(x, x1, t)
        #x = self.sa6(x)
        output = self.outc(x)
        return output

def count_params(model): return sum(p.numel() for p in model.parameters() if p.requires_grad) 