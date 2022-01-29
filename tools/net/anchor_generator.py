import torch
import numpy as np

class AnchorGenerator(object):
    def __init__(self, anchor_range, anchor_generator_config):
        super().__init__()
        self.anchor_generator_cfg = anchor_generator_config
        self.anchor_range = anchor_range
        self.anchor_sizes = [config['anchor_sizes'] for config in anchor_generator_config]
        self.anchor_rotations = [config['anchor_rotations'] for config in anchor_generator_config]
        self.anchor_heights = [config['anchor_heights'] for config in anchor_generator_config]
        self.align_center = [config.get('align_center', False) for config in anchor_generator_config]

        assert len(self.anchor_sizes) == len(self.anchor_rotations) == len(self.anchor_heights)
        self.num_of_anchor_sets = len(self.anchor_sizes)

    def generate_anchors(self, grid_sizes):
        assert len(grid_sizes) == self.num_of_anchor_sets
        all_anchors = []
        num_anchors_per_location = []
        for grid_size, anchor_size, anchor_rotation, anchor_height, align_center in zip(
                grid_sizes, self.anchor_sizes, self.anchor_rotations, self.anchor_heights, self.align_center):

            num_anchors_per_location.append(len(anchor_rotation) * len(anchor_size) * len(anchor_height))
            if align_center:
                x_stride = (self.anchor_range[2] - self.anchor_range[0]) / grid_size[0]
                y_stride = (self.anchor_range[3] - self.anchor_range[1]) / grid_size[1]
                x_offset, y_offset = x_stride / 2, y_stride / 2
            else:
                x_stride = (self.anchor_range[2] - self.anchor_range[0]) / (grid_size[0] - 1)
                y_stride = (self.anchor_range[3] - self.anchor_range[1]) / (grid_size[1] - 1)
                x_offset, y_offset = 0, 0

            x_shifts = torch.arange(
                self.anchor_range[0] + x_offset, self.anchor_range[2] + 1e-5, step=x_stride, dtype=torch.float32,
            ).cuda()
            y_shifts = torch.arange(
                self.anchor_range[1] + y_offset, self.anchor_range[3] + 1e-5, step=y_stride, dtype=torch.float32,
            ).cuda()
            
            print(x_shifts)
            num_anchor_size, num_anchor_rotation = anchor_size.__len__(), anchor_rotation.__len__()
            anchor_rotation = x_shifts.new_tensor(anchor_rotation)
            anchor_size = x_shifts.new_tensor(anchor_size)
            print('original anchor size is :', anchor_size.shape)
            x_shifts, y_shifts = torch.meshgrid([
                x_shifts, y_shifts
            ])  # [x_grid, y_grid]
            anchors = torch.stack((x_shifts, y_shifts), dim=-1)  # [x, y, 2]
            # x , y , 1 ,2
            anchors = anchors[:, :,  None, :].repeat(1, 1, anchor_size.shape[0], 1)
            anchor_size = anchor_size.view(1, 1, -1, 2).repeat([*anchors.shape[0:2], 1, 1])
           
            anchors = torch.cat((anchors, anchor_size), dim=-1)
            anchors = anchors[:, :, :, None, :].repeat(1, 1, 1, num_anchor_rotation, 1)
            anchor_rotation = anchor_rotation.view(1, 1, 1, -1, 1).repeat([*anchors.shape[0:2], num_anchor_size, 1, 1])
            anchors = torch.cat((anchors, anchor_rotation), dim=-1)  # [x, y, num_size, num_rot, 7]

            anchors = anchors.permute(1, 0, 2, 3, 4).contiguous()
            #anchors = anchors.view(-1, anchors.shape[-1])
            anchors[..., 1] += anchors[..., 4] / 2  # shift to box centers
            all_anchors.append(anchors)
        return all_anchors, num_anchors_per_location


def Anchor_generatorV2(image_feas, radar_points_xy, sizes=[4,2,0.5],ratios=[3,1,0.5], cuda=True):
    in_height, in_width = image_feas.shape[-2:]
    for points in radar_points_xy:
        height , width = points
    num_sizes, num_ratios =  len(sizes), len(ratios)
    device = torch.device('cuda:0' if cuda else 'cpu')
    boxes_per_pixel = (num_sizes + num_ratios - 1)
    size_tensor = torch.tensor(sizes, device=device)
    ratio_tensor = torch.tensor(ratios, device=device)
    #offset centers by 0.5
    offset_h, offset_w = 0.5, 0.5
    steps_h = 1.0 / in_height  # Scaled steps in y axis
    steps_w = 1.0 / in_width  # Scaled steps in x axis

    # Generate all center points for the anchor boxes
    center_h = torch.arange(height, device=device) #+ offset_h) * steps_h
    center_w = torch.arange(width, device=device) #+ offset_w) * steps_w
    
    shift_y, shift_x = center_h, center_w
    shift_y, shift_x = shift_y.reshape(-1), shift_x.reshape(-1)

    # Generate `boxes_per_pixel` number of heights and widths that are later
    # used to create anchor box corner coordinates (xmin, xmax, ymin, ymax)
    
    w = torch.cat((size_tensor * ratio_tensor[0],
                sizes[0] * ratio_tensor[1:]))\
                * (in_height / in_width)  # Handle rectangular inputs
    print(w)
    h = torch.cat((size_tensor / ratio_tensor[0],
                sizes[0] / ratio_tensor[1:]))
    # Divide by 2 to get half height and half width
    anchor_manipulations = torch.stack((-w, -h, w, h)).T
    # print('anchor manipulations is:',anchor_manipulations)
    # Each center point will have `boxes_per_pixel` number of anchor boxes, so
    # generate a grid of all anchor box centers with `boxes_per_pixel` repeats
    out_grid = torch.stack([shift_x, shift_y, shift_x, shift_y],
                dim=1)
    print('out grid is:', out_grid)
    output = out_grid + anchor_manipulations
    return output.unsqueeze(0)

if __name__ == '__main__':
    from easydict import EasyDict
    config = [
        EasyDict({
            'anchor_sizes': [[4.3, 2.1], [0.9, 1.9], [1.8, 1.8]],
            'anchor_rotations': [0, 1.57],
            'anchor_heights': [0, 0.5]
        })
    ]

    A = AnchorGenerator(
        anchor_range=[-450, -800,  450, 800],
        anchor_generator_config=config
    )
    image_feas = torch.ones(1,3,1600,900)
    radar_xy = np.array([[0,0],[1,1],[1500,800],[1600,900]]).reshape(-1,2)
    Anchor_generatorV2(image_feas,radar_xy,sizes=[4,2,0.5],ratios=[3,1,0.5])
    
