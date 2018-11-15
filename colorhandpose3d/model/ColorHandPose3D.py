import torch

from .HandSegNet import HandSegNet
from .PoseNet import PoseNet
from .PosePrior import PosePrior
from .ViewPoint import ViewPoint
from ..utils.general import *
from ..utils.transforms import *


class ColorHandPose3D(torch.nn.Module):
    """ColorHandPose3D predicts the 3D joint location of a hand given the
    cropped color image of a hand."""

    def __init__(self, with_hand_segnet=True, crop_size=None, num_keypoints=None):
        super(ColorHandPose3D, self).__init__()
        if with_hand_segnet is True:
            self.handsegnet = HandSegNet()
        self.posenet = PoseNet()
        self.poseprior = PosePrior()
        self.viewpoint = ViewPoint()

        self.with_hand_segnet = with_hand_segnet

        if crop_size is None:
            self.crop_size = 256
        else:
            self.crop_size = crop_size

        if num_keypoints is None:
            self.num_keypoints = 21
        else:
            self.num_keypoints = num_keypoints

        # Load weights
        if self.with_hand_segnet is True:
            self.handsegnet.load_state_dict(
                torch.load('/home/ajdillhoff/dev/projects/colorhandpose3d-pytorch/saved/handsegnet.pth.tar'))
            # freeze hand segnet
            for param in self.handsegnet.parameters():
                param.requires_grad = False
        self.posenet.load_state_dict(torch.load('/home/ajdillhoff/dev/projects/colorhandpose3d-pytorch/saved/posenet.pth.tar'))
        self.poseprior.load_state_dict(torch.load('/home/ajdillhoff/dev/projects/colorhandpose3d-pytorch/saved/poseprior.pth.tar'))
        self.viewpoint.load_state_dict(torch.load('/home/ajdillhoff/dev/projects/colorhandpose3d-pytorch/saved/viewpoint.pth.tar'))


    def forward(self, x, hand_sides):
        """Forward pass through the network.

        Args:
            x - Tensor (B x C x H x W): Batch of images.
            hand_sides - Tensor (B x 2): One-hot vector indicating if the hand
                is left or right.

        Returns:
            coords_xyz_rel_normed (B x N_k x 3): Normalized 3D coordinates of
                the joints, where N_k is the number of keypoints.
        """

        if self.with_hand_segnet is True:
            # Segment the hand
            hand_scoremap = self.handsegnet.forward(x)

            # Calculate single highest scoring object
            hand_mask = single_obj_scoremap(hand_scoremap, self.num_keypoints)

            # crop and resize
            centers, _, crops = calc_center_bb(hand_mask)
            crops = crops.to(torch.float32)

            crops *= 1.25
            scale_crop = torch.min(
                torch.max(self.crop_size / crops,
                          torch.tensor(0.25, device=x.device)),
                torch.tensor(5.0, device=x.device))
            image_crop = crop_image_from_xy(x, centers, self.crop_size, scale_crop)
        else:
            image_crop = x

        # detect 2d keypoints
        keypoints_scoremap = self.posenet(image_crop)

        # estimate 3d pose
        coord_can = self.poseprior(keypoints_scoremap, hand_sides)

        rot_params = self.viewpoint(keypoints_scoremap, hand_sides)

        # get normalized 3d coordinates
        rot_matrix = get_rotation_matrix(rot_params)
        cond_right = torch.eq(torch.argmax(hand_sides, 1), 1)
        cond_right_all = torch.reshape(cond_right, [-1, 1, 1]).repeat(1, self.num_keypoints, 3)
        coords_xyz_can_flip = flip_right_hand(coord_can, cond_right_all)
        coords_xyz_rel_normed = coords_xyz_can_flip @ rot_matrix

        # flip left handed inputs wrt to the x-axis for Libhand compatibility.
        coords_xyz_rel_normed = flip_left_hand(coords_xyz_rel_normed, cond_right_all)

        # scale heatmaps
        keypoints_scoremap = F.interpolate(keypoints_scoremap,
                                           self.crop_size,
                                           mode='bilinear',
                                           align_corners=False)

        if self.with_hand_segnet is True:
            return coords_xyz_rel_normed, keypoints_scoremap, image_crop, centers, scale_crop
        else:
            return coords_xyz_rel_normed, keypoints_scoremap, coord_can
