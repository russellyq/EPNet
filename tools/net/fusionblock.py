import torch
import torch.nn as nn
import torch.nn.functional as F
from anchor_generator import Anchor_generatorV2

def feature_gather(image_feas, radar_points_xy):
    # image_feas: B, C, H, W
    # radar_points_xy: (N, 2)
    # radar 投影到图像上的 x y 坐标    
    #
    if radar_points_xy is None:
        return image_feas
    else:
        img_feas_new = torch.zeros(image_feas.shape)
        
        output = Anchor_generatorV2(image_feas, radar_points_xy, sizes=[5.0], ratios=[3.0])
        #output (N, 4)
        for corners in output:
            xmin, xmax, ymin, ymax = corners
            true_where_x_on_img_l = (xmin < image_feas[:, 0]) & (image_feas[:, 0] < xmax) #x in img coords is cols of img
            true_where_y_on_img_l = (ymin < image_feas[:, 1]) & (image_feas[:, 1] < ymax)
            true_where_point_on_img_l = true_where_x_on_img_l & true_where_y_on_img_l

        img_feas_new = image_feas[true_where_point_on_img_l] # filter out points that don't project to image
    
        return img_feas_new

# lidar SA feature + Image conv features
class Atten_Fusion_Conv(nn.Module):
    def __init__(self, image_channel, radar_channel):
        # image_channel, radar_channel 的 H W 相同
        N, Ci, H, W = image_channel
        N, Cr, H, W = radar_channel
        out_channel = int(Cr/6)
        super(Atten_Fusion_Conv, self).__init__()
        self.radar_conv1 = nn.Conv2d(Cr, 1, (3, 3), stride=(1,1), padding=(1,1))
        self.radar_bn1 = nn.BatchNorm2d(1)
        
        self.image_conv1 = nn.Conv2d(Ci, 1, (3, 3), stride=(1,1), padding=(1,1))
        self.image_bn1 = nn.BatchNorm2d(1)
        
        self.image_conv2 = nn.Conv2d(Ci, Cr, (3, 3), stride=(1,1), padding=(1,1))
        self.image_bn2 = nn.BatchNorm2d(Cr)
        
        self.fusion_conv = nn.Conv2d(2*Cr, out_channel, (3, 3), stride=(1,1), padding=(1,1))
        self.fusion_bn = nn.BatchNorm2d(out_channel)
    
    def forward(self, image_feas, radar_feas, radar_points_xy):
        image_feas = feature_gather(image_feas, radar_points_xy) # N, Ci, H, W
        image_feas_f = F.relu(self.image_bn1(self.image_conv1(image_feas))) # N, 1, H, W
        radar_feas_f = F.relu(self.radar_bn1(self.radar_conv1(radar_feas))) # N, 1, H, W
        
        att = torch.sigmoid(torch.tanh(image_feas_f + radar_feas_f)) # N, 1, H, W
        
        image_feas_new = F.relu(self.image_bn2(self.image_conv2(image_feas))) # N, Cr, H, W
        att = image_feas_new * att # N, Cr, H, W
        
        fusion_features = torch.cat([att, radar_feas], dim=1) # N, 2*Cr, H, W
        fusion_features = F.relu(self.fusion_bn(self.fusion_conv(fusion_features)))
        return fusion_features # N, Cr/6, H, W


# if __name__ == '__main__':
#     lidar = torch.ones(3, 18, 1280, 720)
#     image = torch.ones(3, 3, 1280, 720)
#     x = Atten_Fusion_Conv(list(image.size()), list(lidar.size()))
#     y= x(image, lidar, None)
#     print(y.size())
    
    