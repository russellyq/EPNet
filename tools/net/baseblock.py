import torch
import torch.nn as nn
import torch.nn.functional as F

class RadarBlock(nn.Module):
    def __init__(self, input_channel):
        super(RadarBlock, self).__init__()
        # point_cloud_channel: N, C, H, W
        N, C, H, W = input_channel
        
        self.conv1 = nn.Conv2d(C, C, 3, stride=2, padding=1)
        self.bn1 = nn.BatchNorm2d(C)

        self.conv2 = nn.Conv2d(C, C*2, 3, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(2*C)

        self.conv3 = nn.Conv2d(C*2, C*4, 3, stride=2, padding=1)
        self.bn3 = nn.BatchNorm2d(4*C)
        
        self.deconv1 = nn.ConvTranspose2d(C, C*2, (1, 1), stride=(1, 1), padding=(0, 0))
        self.deconv2 = nn.ConvTranspose2d(C*2, C*2, (2, 2), stride=(2, 2), padding=(0, 0))
        self.deconv3 = nn.ConvTranspose2d(C*4, C*2, (4, 4), stride=(4, 4), padding=(0, 1))
        
        self.deconv4 = nn.ConvTranspose2d(C*6, C*6, (2, 2), stride=(2, 2), padding=(0, 0))
        self.bn4 = nn.BatchNorm2d(C*6)
        # print('Successful Initialzation of Radar Block')

    def forward(self, lidar_points):
        lidar_features1 = F.relu(self.bn1(self.conv1(lidar_points)))
        lidar_features2 = F.relu(self.bn2(self.conv2(lidar_features1)))
        lidar_features3 = F.relu(self.bn3(self.conv3(lidar_features2)))

        lidar_features1 = self.deconv1(lidar_features1)
        lidar_features2 = self.deconv2(lidar_features2)
        lidar_features3 = self.deconv3(lidar_features3)
        # print(lidar_features1.size(), lidar_features2.size(), lidar_features3.size())
        lidar_features = torch.cat((lidar_features1, lidar_features2, lidar_features3), 1)
        
        lidar_features = F.relu(self.bn4(self.deconv4(lidar_features)))

        return lidar_features
        # output channel: (N, 6C, H, W)

class ImageBlock(nn.Module):
    def __init__(self, input_channel):
        super(ImageBlock, self).__init__()

        N, C, H, W = input_channel
        self.bn = nn.BatchNorm2d(C)
        self.conv1 = nn.Conv2d(C, C, (3, 3), stride=(1, 1), padding=(1, 1))
        self.conv2 = nn.Conv2d(C, C, (3, 3), stride=(2, 2), padding=(1, 1))
        self.deconv = nn.ConvTranspose2d(C, C, (2, 2), stride=(2, 2), padding=(0, 0))
        # print('Successful Initialzation of Image Block')
        
    def forward(self, image_points):
        image_features = F.relu(self.bn(self.conv1(image_points)))
        image_features = F.relu(self.bn(self.conv2(image_features)))
        image_features = F.relu(self.bn(self.deconv(image_features)))
        return image_features
        # output_channel: (N, C, H, W)

# class ImageDeBlock(nn.Module):
#     def __init__(self, input_channel):
#         super(ImageDeBlock, self).__init__()

#         N, C, H, W = input_channel
#         self.bn = nn.BatchNorm2d(C)
#         self.deconv = nn.ConvTranspose2d(C, C, (2, 2), stride=(2, 2), padding=(0, 0))


#     def forward(self, image_points):
#         image_features = F.relu(self.bn(self.deconv(image_points)))
#         return image_features
        # output_channel: (N, output_channel, 2*H, 2*C)

if __name__ == '__main__':
    lidar = torch.ones(3, 3, 1280, 720)
    x = RadarBlock(list(lidar.size()))
    print(x.forward(lidar).size())

    image = torch.ones(3, 3, 1280, 720)
    x = ImageBlock(list(image.size()))
    print(x.forward(image).size())