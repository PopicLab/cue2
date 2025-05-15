import cue.img.constants
from cue.img.constants import TargetType
import cue.img.utils
from collections import defaultdict
import numpy as np
from scipy.ndimage.filters import maximum_filter
import torch
import torchvision.transforms


class SVKeypointHeatmapUtility:
    def __init__(self, image_dim, num_sv_labels, sigma, stride, peak_threshold):
        self.heatmap_stride = stride
        self.sigma = sigma
        self.peak_threshold = peak_threshold
        self.heatmap_dim = int(image_dim / self.heatmap_stride)
        self.num_heatmap_channels = num_sv_labels

    def keypoints2heatmaps(self, target):
        # generate keypoint heatmaps: one heatmap channel per SV type and genotype
        heatmaps = np.zeros((self.heatmap_dim, self.heatmap_dim, self.num_heatmap_channels))
        label2keypoints = defaultdict(list)
        for label, keypoints in zip(target[TargetType.labels], target[TargetType.keypoints]):
            label2keypoints[label.item()].append(np.array(keypoints.cpu()))

        for sv_label in label2keypoints:
            heatmap_idx = sv_label - 1  # -1 to account for background label
            keypoints = [sv_kps[0] for sv_kps in label2keypoints[sv_label]]
            for point in keypoints:
                if point[2] == cue.img.constants.KP_VISIBLE:
                    heatmaps[:, :, heatmap_idx] = self.add_gaussian_at_point(point[:2], heatmaps[:, :, heatmap_idx])
        target[TargetType.heatmaps] = torchvision.transforms.ToTensor()(heatmaps).float()

    def heatmaps2predictions(self, target):
        heatmaps = target[TargetType.heatmaps].permute(1, 2, 0).detach().cpu().numpy()
        keypoints = []
        scores = []
        labels = []
        for heatmap_idx in range(self.num_heatmap_channels):
            for peak in self.find_peaks(heatmaps[:, :, heatmap_idx]):
                kp = cue.img.utils.upscale_keypoints(peak, self.heatmap_stride)
                keypoints.append([[kp[0], kp[1], 1]])
                labels.append(heatmap_idx + 1)
                scores.append(min(1, heatmaps[:, :, heatmap_idx][tuple(peak[::-1])]))
        target[TargetType.labels] = torch.as_tensor(labels, dtype=torch.int64)
        target[TargetType.keypoints] = torch.as_tensor(keypoints, dtype=torch.float32)
        target[TargetType.scores] = torch.as_tensor(scores, dtype=torch.float32)

    def add_gaussian_at_point(self, p, heatmap):
        x, y = np.meshgrid([i for i in range(int(self.heatmap_dim))], [i for i in range(int(self.heatmap_dim))])
        offset = self.heatmap_stride / 2.0 - 0.5
        exp = ((x * self.heatmap_stride + offset - p[0]) ** 2 +
               (y * self.heatmap_stride + offset - p[1]) ** 2) / 2.0 / self.sigma / self.sigma
        heatmap += np.multiply(exp <= 4.6052, np.exp(-exp))
        heatmap[heatmap > 1.0] = 1.0
        return heatmap

    def find_peaks(self, heatmap):
        return np.array(np.nonzero((maximum_filter(heatmap, size=8, mode='constant') == heatmap) *
                                   (heatmap > self.peak_threshold))[::-1]).T

