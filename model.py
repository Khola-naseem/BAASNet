
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
class BoundaryAwareSmoothAttention(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(BoundaryAwareSmoothAttention, self).__init__()
        self.query = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.key = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.value = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        
        self.edge_detect = nn.Sequential(
            nn.Conv2d(in_channels, out_channels//2, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels//2),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels//2, 1, kernel_size=1)
        )
        
        self.gamma = nn.Parameter(torch.zeros(1))
        self.beta = nn.Parameter(torch.zeros(1))  # Weight for boundary attention
        
    def forward(self, x):
        batch_size, C, H, W = x.size()
        
        #  attention
        proj_query = self.query(x).view(batch_size, -1, H * W).permute(0, 2, 1)
        proj_key = self.key(x).view(batch_size, -1, H * W)
        energy = torch.bmm(proj_query, proj_key)
        attention = F.softmax(energy, dim=-1)
        edge_map = torch.sigmoid(self.edge_detect(x))
        edge_attention = edge_map.view(batch_size, 1, H * W)
        final_attention = attention * (1 + self.beta * edge_attention)
        final_attention = F.normalize(final_attention, p=1, dim=2)  # Re-normalize
        proj_value = self.value(x).view(batch_size, -1, H * W)
        out = torch.bmm(proj_value, final_attention.permute(0, 2, 1))
        out = out.view(batch_size, C, H, W)
        
        out = self.gamma * out + x
        return out

class ASPP(nn.Module):
    """Atrous Spatial Pyramid Pooling for better capturing objects at multiple scales"""
    def __init__(self, in_channels, out_channels):
        super(ASPP, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.conv2 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=6, dilation=6)
        self.conv3 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=12, dilation=12)
        self.conv4 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=18, dilation=18)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.conv5 = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.final = nn.Conv2d(out_channels*5, out_channels, kernel_size=1)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, x):
        x1 = self.relu(self.conv1(x))
        x2 = self.relu(self.conv2(x))
        x3 = self.relu(self.conv3(x))
        x4 = self.relu(self.conv4(x))
        
        x5 = self.pool(x)
        x5 = self.conv5(x5)
        x5 = F.interpolate(x5, size=x.shape[2:], mode='bilinear', align_corners=True)
        
        out = torch.cat([x1, x2, x3, x4, x5], dim=1)
        out = self.final(out)
        out = self.bn(out)
        out = self.relu(out)
        return out
class BoundaryAwareAttentionUNet(nn.Module):
    def __init__(self, num_classes=1, pretrained=True):
        super(BoundaryAwareAttentionUNet, self).__init__()
        resnet = models.resnet50(pretrained=pretrained)
 
        self.enc1 = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu)  # 64 channels
        self.enc2 = nn.Sequential(resnet.maxpool, resnet.layer1)  # 256 channels
        self.enc3 = resnet.layer2  # 512 channels
        self.enc4 = resnet.layer3  # 1024 channels
        self.enc5 = resnet.layer4  # 2048 channels
        # Boundary-aware attention
        self.attention = BoundaryAwareSmoothAttention(2048, 2048)
        self.aspp = ASPP(2048, 256) 
        # ASPP for multi-scale feature extraction
        self.up4 = self.up_block(256 + 1024, 512)
        self.up3 = self.up_block(512 + 512, 256)
        self.up2 = self.up_block(256 + 256, 128)
        self.up1 = self.up_block(128 + 64, 64)
        # Final convolution
        self.final_conv = nn.Conv2d(64, num_classes, kernel_size=1)
        self.edge_branch = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, num_classes, kernel_size=1)
        )
 
    def up_block(self, in_ch, out_ch):
        return nn.Sequential(
            nn.ConvTranspose2d(in_ch, out_ch, kernel_size=2, stride=2),
            nn.BatchNorm2d(out_ch),                                                        
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )
 
    def forward(self, x):
        original_size = x.shape[2:]
        # Encoder
        x1 = self.enc1(x)  
        x2 = self.enc2(x1)  
        x3 = self.enc3(x2)  
        x4 = self.enc4(x3)  
        x5 = self.enc5(x4) 
        # Apply attention
        x5 = self.attention(x5)
        # ASPP for multi-scale feature extraction
        x5 = self.aspp(x5)
 
        # Decoder with skip connections
        d4 = self.up4(torch.cat([
            F.interpolate(x5, size=x4.shape[2:], mode='bilinear', align_corners=True), 
            x4
        ], dim=1))
        
        d3 = self.up3(torch.cat([
            F.interpolate(d4, size=x3.shape[2:], mode='bilinear', align_corners=True), 
            x3
        ], dim=1))
        
        d2 = self.up2(torch.cat([
            F.interpolate(d3, size=x2.shape[2:], mode='bilinear', align_corners=True), 
            x2
        ], dim=1))
        
        d1 = self.up1(torch.cat([
            F.interpolate(d2, size=x1.shape[2:], mode='bilinear', align_corners=True), 
            x1
        ], dim=1))
        
        # Main segmentation output
        out = self.final_conv(F.interpolate(d1, size=original_size, mode='bilinear', align_corners=True))
        
        edge_out = self.edge_branch(d1)
        edge_out = F.interpolate(edge_out, size=original_size, mode='bilinear', align_corners=True)
        
        if self.training:
            return out, edge_out
        else:
            return out
