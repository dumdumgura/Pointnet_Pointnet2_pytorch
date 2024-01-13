import torch.nn as nn
import torch.nn.functional as F
from pointnet2_utils import PointNetSetAbstraction
from pointnet_utils import PointNetEncoder, feature_transform_reguliarzer
import pyrender

class get_model(nn.Module):
    def __init__(self,num_class,normal_channel=True):
        super(get_model, self).__init__()
        if normal_channel:
            channel = 6
        else:
            channel = 3
        self.normal_channel = normal_channel
        self.feat = PointNetEncoder(global_feat=True, feature_transform=True, channel=channel)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, num_class)
        self.dropout = nn.Dropout(p=0.4)
        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(256)
        self.relu = nn.ReLU()
        #'''

    def forward(self, xyz):
        xyz = xyz.permute(0, 2, 1).contiguous() # batch, 6, num_pts
        #self.draw_point_cloud(xyz.permute(0, 2, 1).contiguous()[0].cpu().detach(),norm.permute(0, 2, 1).contiguous()[0].cpu().detach())

        x, trans, trans_feat = self.feat(xyz)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        #x = F.log_softmax(x, -1)

        return x, trans_feat

    def draw_point_cloud(self,points,colors):
        cloud = pyrender.Mesh.from_points(points,colors)
        scene = pyrender.Scene()
        scene.add(cloud)
        viewer = pyrender.Viewer(scene, use_raymond_lighting=True, point_size=5)


