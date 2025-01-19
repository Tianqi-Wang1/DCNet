from abc import *
import torch
import torch.nn as nn
import pickle as pkl
import torch.nn.functional as F

class SimclrLayer(nn.Module):
    """ contrastivie projection layers """
    def __init__(self, P, last_dim, feature_dim):
        super(SimclrLayer, self).__init__()

        self.fc1 = nn.Linear(last_dim, last_dim)
        self.relu = nn.ReLU()
        self.last = nn.ModuleList()

        for _ in range(P.n_tasks):
            self.last.append(nn.Linear(last_dim, feature_dim))

        self.ec = nn.ParameterList()
        for _ in range(P.n_tasks):
            self.ec.append(nn.Parameter(torch.randn(1, last_dim)))
        self.gate = torch.sigmoid

    def mask(self, t, s=1):
        gc1 = self.gate(s * self.ec[t])
        return gc1

    def mask_out(self, out, mask):
        out = out * mask.expand_as(out)
        return out

    def forward(self, t, features, s=1):
        gc1 = self.mask(t, s=s)

        out = self.fc1(features)
        out = self.relu(out)
        out = self.mask_out(out, gc1)
        out = self.last[t](out)

        return out, gc1

class BaseModel(nn.Module, metaclass=ABCMeta):
    def __init__(self, last_dim, P, num_classes=10, feature_dim=128):
        super(BaseModel, self).__init__()
        self.P = P
        d = pkl.load(open('./BV_pkl_CL/eq_100_4+10_256_0.1_9.pkl', 'rb')).data
        d = F.normalize(d, dim=1).cuda() #dim=1
        self.prototypes = d

        self.linear = nn.ModuleList()
        for _ in range(P.n_tasks):
            self.linear.append(nn.Sequential(nn.Linear(last_dim, last_dim), nn.Linear(last_dim, num_classes)))

        # Contrastive projection
        self.simclr_layer = SimclrLayer(P, last_dim, P.feat_dim)

        # Independent OOD classifier
        self.joint_distribution_layer = nn.ModuleList()
        for _ in range(P.n_tasks):
            self.joint_distribution_layer.append(nn.Linear(self.P.embedding_dim, 4 * num_classes+1, bias=True))

    @abstractmethod
    def penultimate(self, t, inputs, s, all_features=False):
        pass

    def forward(self, t, inputs, s=1, penultimate=False, feature=False, shift=False, joint=False):
        _aux = {}
        _return_aux = False

        penultimate_feature, masks = self.penultimate(t, inputs, s)

        output = self.linear[t](penultimate_feature)

        if penultimate:
            _return_aux = True
            _aux['penultimate'] = penultimate_feature

        if joint or feature:
            _return_aux = True
            feature, gc1 = self.simclr_layer(t, penultimate_feature, s)
            masks.append(gc1)
            _aux['feature'] = feature

        if joint:
            _return_aux = True
            _aux['joint'] = self.joint_distribution_layer[t](penultimate_feature)

        if _return_aux:
            return output, _aux, masks

        return output, masks
