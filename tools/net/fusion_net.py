import torch
import torch.nn as nn
import torch.nn.functional as F
import os
from baseblock import RadarBlock, ImageBlock
from fusionblock import Atten_Fusion_Conv
from ssd import build_ssd
import numpy as np

class Pointnet2MSG(nn.Module):
    def __init__(self, image_channel, radar_channel, phase='train'):
        """
        Args:
        phase: (string) Can be "test" or "train"
        size: input image size
        base: VGG16 layers for input, size of either 300 or 500
        extras: extra layers that feed to multibox loc and conf layers
        head: "multibox head" consists of loc and conf conv layers
        """
        super(Pointnet2MSG, self).__init__()

        self.radar_modules = nn.ModuleList()
        self.image_modules = nn.ModuleList()
        self.fusion_modules = nn.ModuleList()
        
        self.conv = nn.Conv2d(3*image_channel[1], image_channel[1], (3, 3), stride=(1, 1), padding=(1, 1))
        self.bn = nn.BatchNorm2d(image_channel[1])
        
        for i in range(4):
            self.radar_modules.append(RadarBlock(radar_channel))
            
            self.image_modules.append(ImageBlock(image_channel))
            self.fusion_modules.append(Atten_Fusion_Conv(image_channel, [radar_channel[0], 
                                                                         6*radar_channel[1], 
                                                                         radar_channel[2],
                                                                         radar_channel[3]]))
        self.fusion_modules.append(Atten_Fusion_Conv(image_channel, [radar_channel[0], 
                                                                         6*radar_channel[1], 
                                                                     radar_channel[2],
                                                                         radar_channel[3]]))
        self.conv_f1 = nn.Conv2d(radar_channel[1], radar_channel[1], (4, 3), stride=(4, 3), padding=(0, 0)) # (400, 300)
        self.conv_f2 = nn.Conv2d(radar_channel[1], radar_channel[1], (3, 3), stride=(4, 3), padding=(0, 0)) # (100, 100)
        self.bn_f1 = nn.BatchNorm2d(radar_channel[1])
        self.deconv_f1 = nn.ConvTranspose2d(radar_channel[1], radar_channel[1], (3, 3), stride=(3, 3), padding=(0, 0))
        self.bn_f2 = nn.BatchNorm2d(radar_channel[1])
        self.conv_f3 = nn.Conv2d(radar_channel[1], 3, (3, 3), stride=(1, 1), padding=(1, 1))
        self.bn_f3 = nn.BatchNorm2d(3)
        
        self.ssd = build_ssd(phase)
        print('Successful Initialzation of Network')   
        
    def forward(self, image_points, radar_points, radar_points_xy):
        image_features, radar_features = [image_points], [radar_points]

        for i in range(3):
            i_features = self.image_modules[i](image_features[-1])
            r_features = self.radar_modules[i](radar_features[-1])
            # print(i_features.size(), r_features.size())
            fusion_features = self.fusion_modules[i](i_features, r_features, radar_points_xy)
            
            image_features.append(i_features)
            radar_features.append(fusion_features)
        
        i_features = torch.cat([image_features[1], image_features[2], image_features[3]], dim=1) # B, 3*Ci, H, W
        i_features = F.relu(self.bn(self.conv(i_features)))
        
        r_features = self.radar_modules[-1](radar_features[-1])
        
        fusion_features = self.fusion_modules[-1](i_features, r_features, radar_points_xy)
        
        fusion_features = F.relu(self.bn_f1(self.conv_f2(self.conv_f1(fusion_features))))

        fusion_features = F.relu(self.bn_f2(self.deconv_f1(fusion_features)))

        fusion_features = F.relu(self.bn_f3(self.conv_f3(fusion_features)))
        
        fusion_features = self.ssd(fusion_features)
        
        return fusion_features
    
    def load_weights(self, base_file):
        other, ext = os.path.splitext(base_file)
        if ext == '.pkl' or '.pth':
            print('Loading weights into state dict...')
            self.load_state_dict(torch.load(base_file,
                                 map_location=lambda storage, loc: storage))
            print('Finished!')
        else:
            print('Sorry only .pth and .pkl files supported.')
        


if __name__ == '__main__':
    # H=1600, W=900
    lidar = torch.ones(1, 3, 1600, 900)
    image = torch.ones(1, 3, 1600, 900)
    x = Pointnet2MSG(list(image.size()), list(lidar.size()), phase='test')
    vgg_weights = torch.load('./vgg_weights/vgg16_reducedfc.pth')
    x.ssd.vgg.load_state_dict(vgg_weights)
    radar_points_xy = np.array([[1,1], [800, 450], [1599, 899]])
    output = x(image, lidar, radar_points_xy)
    # print(output.size())
    detections = output.data
    print(x)
