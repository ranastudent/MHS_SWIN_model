import timm
import torch
import torch.nn as nn
import torch.nn.functional as F


# ==== paste your modules ====
# DSPE
class  DSPE(nn.Module):
    def __init__(self):
        super().__init__()
        self.dw = nn.Conv2d(in_channels=3, out_channels=3, kernel_size=4, stride=4, groups=3)
        self.pw = nn.Conv2d(in_channels=3, out_channels=96, kernel_size=1)
        self.bn = nn.BatchNorm2d(96)

    def forward(self, x):
        x = self.dw(x)
        x = self.pw(x)
        x = self.bn(x)
        return x  # (B,96,56,56)
# CBAM
class  ChannelAttention(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction),
            nn.ReLU(),
            nn.Linear(in_channels // reduction, in_channels)
        )

    def forward(self, x):
        b, c, h, w = x.size()
        avg = F.adaptive_avg_pool2d(x, 1).view(b, c)
        max_ = F.adaptive_max_pool2d(x, 1).view(b, c)
        out = self.mlp(avg) + self.mlp(max_)
        scale = torch.sigmoid(out).view(b, c, 1, 1)
        return x * scale
class SpatialAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=7, padding=3)

    def forward(self, x):
        avg = torch.mean(x, dim=1, keepdim=True)
        max_, _ = torch.max(x, dim=1, keepdim=True)
        attn = torch.cat([avg, max_], dim=1)
        attn = torch.sigmoid(self.conv(attn))
        return x * attn  # channels preserved
class CBAM(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.ca = ChannelAttention(channels)
        self.sa = SpatialAttention()

    def forward(self, x):
        x = self.ca(x)
        x = self.sa(x)
        return x
# APFH
class  APFH(nn.Module):
    def __init__(self, num_classes=4):
        super().__init__()
        self.fc1 = nn.Linear(768 * 2, 512)
        self.act = nn.GELU()
        self.drop = nn.Dropout(0.3)
        self.fc2 = nn.Linear(512, num_classes)

    def forward(self, x):
        # x: (B,C,H,W)
        gap = torch.mean(x, dim=(2,3))
        gmp = torch.amax(x, dim=(2,3))
        x = torch.cat([gap, gmp], dim=1)
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        return x
# MHSSwin
class MHSSwin(nn.Module):
    def __init__(self):
        super().__init__()

        # Base Swin Tiny model (features only)
        base = timm.create_model(
            "swin_tiny_patch4_window7_224",
            pretrained=True,
            num_classes=0  # features only
        )

        # DSPE: converts input images to 96 channels
        self.dspe = DSPE()

        # Swin layers and norm
        self.layers = base.layers
        self.norm = base.norm  # final LayerNorm

        # CBAM modules for each stage
        self.cbam1 = CBAM(96)
        self.cbam2 = CBAM(192)
        self.cbam3 = CBAM(384)
        self.cbam4 = CBAM(768)

        # Classifier
        self.head = APFH(num_classes=4)

    def forward(self, x):
        # DSPE: (B,3,H,W) -> (B,96,56,56)
        x = self.dspe(x)

        # Permute to Swin format: (B,H,W,C)
        x = x.permute(0,2,3,1)

        # Stage 1
        x = self.layers[0](x)
        x = self.apply_cbam(x, self.cbam1)

        # Stage 2
        x = self.layers[1](x)
        x = self.apply_cbam(x, self.cbam2)

        # Stage 3
        x = self.layers[2](x)
        x = self.apply_cbam(x, self.cbam3)

        # Stage 4
        x = self.layers[3](x)
        x = self.apply_cbam(x, self.cbam4)

        # Final LayerNorm
        x = self.norm(x)  # Swin internally reshapes to (B,H*W,C)

        # Convert to (B,C,H,W) for classifier
        B,H,W,C = x.shape
        x = x.permute(0,3,1,2)

        # Classifier
        out = self.head(x)
        return out

    def apply_cbam(self, x, cbam):
        # x: (B,H,W,C)
        B,H,W,C = x.shape
        x_cbam = x.permute(0,3,1,2)  # (B,C,H,W)
        x_cbam = cbam(x_cbam)
        x = x_cbam.permute(0,2,3,1)  # (B,H,W,C)
        return x
# ============================

def get_model(weight_path="MHS_SWIN.pth"):
    model = MHSSwin()
    model.load_state_dict(torch.load(weight_path, map_location="cpu"))
    model.eval()
    return model