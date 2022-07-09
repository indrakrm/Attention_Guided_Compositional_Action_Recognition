import torch
import torch.nn as nn
import torch.nn.functional as F
from model.resnet3d_xl import Net
from model.nonlocal_helper import Nonlocal
# from model.TRNmodule import RelationModuleMultiScale
from model.Attention_trans import Encoder

class VideoModelCoord(nn.Module):
    def __init__(self, opt):
        super(VideoModelCoord, self).__init__()
        self.nr_boxes = opt.num_boxes
        self.nr_actions = opt.num_classes
        self.nr_frames = opt.num_frames // 2
        self.coord_feature_dim = opt.coord_feature_dim

        self.coord_to_feature = nn.Sequential(
            nn.Linear(4, self.coord_feature_dim//2, bias=False),
            nn.BatchNorm1d(self.coord_feature_dim//2),
            nn.ReLU(inplace=True),
            nn.Linear(self.coord_feature_dim//2, self.coord_feature_dim, bias=False),
            nn.BatchNorm1d(self.coord_feature_dim),
            nn.ReLU()
        )

        self.spatial_node_fusion = nn.Sequential(
            nn.Linear(self.coord_feature_dim*2, self.coord_feature_dim, bias=False),
            nn.BatchNorm1d(self.coord_feature_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.coord_feature_dim, self.coord_feature_dim, bias=False),
            nn.BatchNorm1d(self.coord_feature_dim),
            nn.ReLU()
        )

        self.box_feature_fusion = nn.Sequential(
            nn.Linear(self.nr_frames*self.coord_feature_dim, self.coord_feature_dim, bias=False),
            nn.BatchNorm1d(self.coord_feature_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.coord_feature_dim, self.coord_feature_dim, bias=False),
            nn.BatchNorm1d(self.coord_feature_dim),
            nn.ReLU()
        )

        self.classifier = nn.Sequential(
            nn.Linear(self.coord_feature_dim, self.coord_feature_dim),
            # nn.BatchNorm1d(self.coord_feature_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.coord_feature_dim, 512), #self.coord_feature_dim),
            nn.ReLU(inplace=True),
            nn.Linear(512, self.nr_actions)
        )

        if opt.fine_tune:
            self.fine_tune(opt.fine_tune)

    def fine_tune(self, restore_path, parameters_to_train=['classifier']):
        weights = torch.load(restore_path)['state_dict']
        new_weights = {}
        #import pdb
        for k, v in weights.items():
            if not 'classifier.4' in k:
                new_weights[k.replace('module.', '')] = v
        #pdb.set_trace()
        self.load_state_dict(new_weights, strict=False)
        print('Num of weights in restore dict {}'.format(len(new_weights.keys())))

        frozen_weights = 0
        for name, param in self.named_parameters():
            if not 'classifier.4' in name:

                param.requires_grad = False
                frozen_weights += 1

            else:
                print('Training : {}'.format(name))
        print('Number of frozen weights {}'.format(frozen_weights))
        assert frozen_weights != 0, 'You are trying to fine tune, but no weights are frozen!!! ' \
                                    'Check the naming convention of the parameters'

    def forward(self, global_img_input, box_categories, box_input, video_label, is_inference=False):
        # local_img_tensor is (b, nr_frames, nr_boxes, 3, h, w)
        # global_img_tensor is (b, nr_frames, 3, h, w)
        # box_input is (b, nr_frames, nr_boxes, 4)
	
        b = box_input.size(0)
        #b, _, _, _h, _w = global_img_input.size()
        # global_imgs = global_img_input.view(b*self.nr_frames, 3, _h, _w)
        # local_imgs = local_img_input.view(b*self.nr_frames*self.nr_boxes, 3, _h, _w)

        box_input = box_input.transpose(2, 1).contiguous()
        box_input = box_input.view(b*self.nr_boxes*self.nr_frames, 4)

        bf = self.coord_to_feature(box_input)
        bf = bf.view(b, self.nr_boxes, self.nr_frames, self.coord_feature_dim)

        # spatial message passing (graph)
        spatial_message = bf.sum(dim=1, keepdim=True)  # (b, 1, self.nr_frames, coord_feature_dim)
        # message passed should substract itself, and normalize to it as a single feature
        spatial_message = (spatial_message - bf) / (self.nr_boxes - 1)  # message passed should substract itself
        bf_and_message = torch.cat([bf, spatial_message], dim=3)  # (b, nr_boxes, nr_frames, 2*coord_feature_dim)

        # (b*nr_boxes*nr_frames, coord_feature_dim)
        bf_spatial = self.spatial_node_fusion(bf_and_message.view(b*self.nr_boxes*self.nr_frames, -1))
        bf_spatial = bf_spatial.view(b, self.nr_boxes, self.nr_frames, self.coord_feature_dim)

        bf_temporal_input = bf_spatial.view(b, self.nr_boxes, self.nr_frames*self.coord_feature_dim)

        box_features = self.box_feature_fusion(bf_temporal_input.view(b*self.nr_boxes, -1))  # (b*nr_boxes, coord_feature_dim)
        box_features = torch.mean(box_features.view(b, self.nr_boxes, -1), dim=1)  # (b, coord_feature_dim)
        # video_features = torch.cat([global_features, local_features, box_features], dim=1)
        video_features = box_features

        cls_output = self.classifier(video_features)  # (b, num_classes)
        return cls_output

class VideoModelCoordLatent(nn.Module):
    def __init__(self, opt):
        super(VideoModelCoordLatent, self).__init__()
        self.nr_boxes = opt.num_boxes
        self.nr_actions = opt.num_classes
        self.nr_frames = opt.num_frames // 2
        self.img_feature_dim = opt.img_feature_dim
        self.coord_feature_dim = opt.coord_feature_dim

        self.category_embed_layer = nn.Embedding(3, opt.coord_feature_dim // 2, padding_idx=0, scale_grad_by_freq=True)

        self.coord_to_feature = nn.Sequential(
            nn.Linear(4, self.coord_feature_dim//2, bias=False),
            nn.BatchNorm1d(self.coord_feature_dim//2),
            nn.ReLU(inplace=True),
            nn.Linear(self.coord_feature_dim//2, self.coord_feature_dim, bias=False),
            nn.BatchNorm1d(self.coord_feature_dim),
            nn.ReLU()
        )

        self.coord_category_fusion = nn.Sequential(
            nn.Linear(self.coord_feature_dim+self.coord_feature_dim//2, self.coord_feature_dim, bias=False),
            nn.BatchNorm1d(self.coord_feature_dim),
            nn.ReLU(inplace=True),
        )
        """
        self.spatial_node_fusion = nn.Sequential(
            nn.Linear(self.coord_feature_dim*2, self.coord_feature_dim, bias=False),
            nn.BatchNorm1d(self.coord_feature_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.coord_feature_dim, self.coord_feature_dim, bias=False),
            nn.BatchNorm1d(self.coord_feature_dim),
            nn.ReLU()
        )

        self.box_feature_fusion = nn.Sequential(
            nn.Linear(self.nr_frames*self.coord_feature_dim, self.coord_feature_dim, bias=False),
            nn.BatchNorm1d(self.coord_feature_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.coord_feature_dim, self.coord_feature_dim, bias=False),
            nn.BatchNorm1d(self.coord_feature_dim),
            nn.ReLU()
        )
        """
        #changes

        self.spatial_trans_att=Encoder(self.coord_feature_dim,self.coord_feature_dim,self.coord_feature_dim,6,8,0)
        self.temporal_trans_att=Encoder(self.coord_feature_dim,self.coord_feature_dim,self.coord_feature_dim,6,8,0)

        self.spatial_ff= nn.Sequential(
             nn.Linear(self.nr_boxes*self.coord_feature_dim, self.nr_boxes*self.coord_feature_dim//2, bias=True),
             nn.ReLU(inplace=True),
             nn.Linear(self.nr_boxes*self.coord_feature_dim//2, self.coord_feature_dim, bias=True),
             #nn.BatchNorm1d(self.coord_feature_dim),
             #nn.ReLU()
             )
        self.temporal_ff= nn.Sequential(
             nn.Linear(self.nr_frames*self.coord_feature_dim, (self.nr_frames//2)*self.coord_feature_dim, bias=True),
             nn.ReLU(inplace=True),
             nn.Linear((self.nr_frames//2)*self.coord_feature_dim, self.coord_feature_dim, bias=True),
             nn.BatchNorm1d(self.coord_feature_dim),
             nn.ReLU()
             )

        self.classifier = nn.Sequential(
            nn.Linear(self.coord_feature_dim, self.coord_feature_dim),
            # nn.BatchNorm1d(self.coord_feature_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.coord_feature_dim, 512), #self.coord_feature_dim),
            nn.ReLU(inplace=True),
            nn.Linear(512, self.nr_actions)
        )

        if opt.fine_tune:
            self.fine_tune(opt.fine_tune)

    def fine_tune(self, restore_path, parameters_to_train=['classifier']):
        weights = torch.load(restore_path)['state_dict']
        new_weights = {}
        for k, v in weights.items():
            if not 'classifier.4' in k:
                new_weights[k.replace('module.', '')] = v
        self.load_state_dict(new_weights, strict=False)
        print('Num of weights in restore dict {}'.format(len(new_weights.keys())))

        frozen_weights = 0
        for name, param in self.named_parameters():
            if not 'classifier.4' in name:
                param.requires_grad = False
                frozen_weights += 1
            else:
                print('Training : {}'.format(name))
        print('Number of frozen weights {}'.format(frozen_weights))
        assert frozen_weights != 0, 'You are trying to fine tune, but no weights are frozen!!! ' \
                                    'Check the naming convention of the parameters'

    def forward(self, global_img_input, box_categories, box_input, video_label, is_inference=False):
        # local_img_tensor is (b, nr_frames, nr_boxes, 3, h, w)
        # global_img_tensor is (b, nr_frames, 3, h, w)
        # box_input is (b, nr_frames, nr_boxes, 4)

        b = box_input.size(0)
        #b, _, _, _h, _w = global_img_input.size()

        box_input = box_input.transpose(2, 1).contiguous()
        box_input = box_input.view(b*self.nr_boxes*self.nr_frames, 4)

        box_categories = box_categories.long()
        box_categories = box_categories.transpose(2, 1).contiguous()
        box_categories = box_categories.view(b*self.nr_boxes*self.nr_frames)
        box_category_embeddings = self.category_embed_layer(box_categories)  # (b*nr_b*nr_f, coord_feature_dim//2)

        bf = self.coord_to_feature(box_input)
        bf = torch.cat([bf, box_category_embeddings], dim=1)  # (b*nr_b*nr_f, coord_feature_dim + coord_feature_dim//2)
        bf = self.coord_category_fusion(bf)  # (b*nr_b*nr_f, coord_feature_dim)
        bf = bf.view(b, self.nr_frames, self.nr_boxes, self.coord_feature_dim)
        #bf=bf.transpose(1,2)

        """
        # spatial message passing (graph)
        spatial_message = bf.sum(dim=1, keepdim=True)  # (b, 1, self.nr_frames, coord_feature_dim)
        # message passed should substract itself, and normalize to it as a single feature
        spatial_message = (spatial_message - bf) / (self.nr_boxes - 1)  # message passed should substract itself
        bf_and_message = torch.cat([bf, spatial_message], dim=3)  # (b, nr_boxes, nr_frames, 2*coord_feature_dim)

        # (b*nr_boxes*nr_frames, coord_feature_dim)
        bf_spatial = self.spatial_node_fusion(bf_and_message.view(b*self.nr_boxes*self.nr_frames, -1))
        bf_spatial = bf_spatial.view(b, self.nr_boxes, self.nr_frames, self.coord_feature_dim)

        bf_temporal_input = bf_spatial.view(b, self.nr_boxes, self.nr_frames*self.coord_feature_dim)

        box_features = self.box_feature_fusion(bf_temporal_input.view(b*self.nr_boxes, -1))  # (b*nr_boxes, coord_feature_dim)

        box_features = torch.mean(box_features.view(b, self.nr_boxes, -1), dim=1)  # (b, coord_feature_dim)
        # video_features = torch.cat([global_features, local_features, box_features], dim=1)
        video_features = box_features
        """
        bf_spatial=self.spatial_trans_att(bf.view(b*self.nr_frames,self.nr_boxes, self.coord_feature_dim))
        bf_spatial=self.spatial_ff(bf_spatial.view(b,self.nr_frames,-1))
        bf_temporal=self.temporal_trans_att(bf_spatial)
        video_features=self.temporal_ff(bf_temporal.view(b,-1))
        cls_output = self.classifier(video_features)  # (b, num_classes)

        return cls_output


