import cue.img.constants
import cue.img.utils
from cue.img.heatmap import SVKeypointHeatmapUtility
from cue.nn import blocks
from enum import Enum
import torch.nn as nn
import torch


class CueModelConfig:
    MODEL_TYPE = Enum("TYPE", 'HG ')

    def __init__(self, config):
        self.config = config

    def get_model(self):
        model_type = self.MODEL_TYPE[self.config.model_architecture]
        return {
            self.MODEL_TYPE.HG: self.hourglass,
        }[model_type]()

    def hourglass(self):
        return MultiSVHG(self.config)

class MultiSVHG(nn.Module):
    # Stacked hourglass network for SV breakpoint prediction
    # Implementation based on Newell et al human pose estimation models (ECCV 2016, NeurIPS 2017)
    # PoseNet (https://github.com/princeton-vl/pose-ae-train)

    def __init__(self, config):
        super(MultiSVHG, self).__init__()
        self.config = config
        self.heatmap_generator = SVKeypointHeatmapUtility(config.image_dim, num_sv_labels=config.n_classes-1,
                                                          sigma=config.sigma, stride=config.stride,
                                                          peak_threshold=config.heatmap_peak_threshold)
        self.hg_in_dim = 256
        self.hg_out_dim = self.heatmap_generator.num_heatmap_channels
        self.hg_expansion = 128
        self.hg_depth = 4
        self.hg_stack_size = 4
        self.backbone = blocks.HourglassBackbone(self.config.n_channels, self.hg_in_dim)
        self.hg_stack = nn.ModuleList([nn.Sequential(
            blocks.Hourglass(self.hg_depth, self.hg_in_dim, self.hg_expansion)
        ) for _ in range(self.hg_stack_size)])
        self.features = nn.ModuleList([nn.Sequential(
            blocks.Residual(self.hg_in_dim, self.hg_in_dim),
            blocks.Conv(self.hg_in_dim, self.hg_in_dim, kernel_size=1, pool=False, bn=True, relu=True)
        ) for _ in range(self.hg_stack_size)])
        self.outs = nn.ModuleList([blocks.Conv(self.hg_in_dim, self.hg_out_dim, 1, pool=False, relu=False, bn=False)
                                   for _ in range(self.hg_stack_size)])
        self.merge_features = nn.ModuleList([
            blocks.Conv(self.hg_in_dim, self.hg_in_dim, kernel_size=1, pool=False, relu=False, bn=False)
            for _ in range(self.hg_stack_size)])
        self.merge_preds = nn.ModuleList([
            blocks.Conv(self.hg_out_dim, self.hg_in_dim, kernel_size=1, pool=False, relu=False, bn=False)
            for _ in range(self.hg_stack_size)])

    def forward(self, images, targets=None):
        images = cue.img.utils.batch_images(images)
        x = self.backbone(images)
        stage_outputs = []
        for i in range(self.hg_stack_size):
            hg = self.hg_stack[i](x)
            feature = self.features[i](hg)
            stack_output = self.outs[i](feature)
            stage_outputs.append(stack_output)
            if i < self.hg_stack_size - 1:
                x = x + self.merge_preds[i](stack_output) + self.merge_features[i](feature)

        outputs = [{cue.img.constants.TargetType.heatmaps: heatmaps} for heatmaps in stage_outputs[-1]]
        for output in outputs: self.heatmap_generator.heatmaps2predictions(output)
        if self.training:
            losses = {'loss_heatmaps': self.loss(stage_outputs, targets)}
            return losses, outputs
        return outputs

    def loss(self, stage_outputs, targets):
        for target in targets:
            self.heatmap_generator.keypoints2heatmaps(target)
        heatmaps_gt = torch.stack([t[cue.img.constants.TargetType.heatmaps].to(self.config.device) for t in targets], dim=0)
        stage_outputs = torch.stack(stage_outputs, dim=0)
        stage_weights = [1] * stage_outputs.shape[0]
        return self.focal_loss(stage_outputs, heatmaps_gt, stage_weights)

    def focal_loss(self, outputs, targets, stage_weights, gamma=1, alpha=0.1, beta=0.02, theta=0.01):
        # Focal L2 loss adapted from SimplePose (Li et al, AAAI 2020)
        dkt = torch.where(torch.ge(targets, theta), outputs - alpha, 1 - outputs - beta)
        flkt = (outputs - targets) ** 2 * (torch.abs(1. - dkt) ** gamma)
        fl = flkt.sum(dim=(1, 2, 3, 4))
        print('Focal L2 loss: ', fl.detach().cpu().numpy())
        return sum([fl[i] * stage_weights[i] for i in range(len(stage_weights))]) / sum(stage_weights)

