# Copyright (c) 2023-2025 Victoria Popic
#
#This program is free software: you can redistribute it and/or modify
#it under the terms of the GNU Affero General Public License as published by
#the Free Software Foundation, either version 3 of the License, or
#(at your option) any later version.

#This program is distributed in the hope that it will be useful,
#but WITHOUT ANY WARRANTY; without even the implied warranty of
#MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#GNU Affero General Public License for more details.

#You should have received a copy of the GNU Affero General Public License
#along with this program.  If not, see <http://www.gnu.org/licenses/>.


import cue.img
from cue.img.constants import *
from cue.img.plotting import *
import cue.img.utils as utils
from cue.seq.constants import *
from cue.seq.genome_scanner import TargetIntervalScanner
import cue.seq.io as io
from cue.seq.sv import SV, SVContainer

import cv2
import logging
import matplotlib
import math
import numpy as np
import os
import sys
import torch
import torchvision.transforms as transforms


class SVStreamingDataset(torch.utils.data.IterableDataset):
    def __init__(self, config, cue_index):
        super().__init__()
        self.config = config
        self.cue_index = cue_index
        self.svs = SVContainer(config.vcf, cue_index.chr_index) if config.vcf else None

    # ----- Training ------
    def get_target(self, interval_pair):
        svs = self.svs.overlap(interval_pair.intervalA)
        if not svs: return None, None, None
        labels = []
        boxes = []
        keypoints = []
        n_visible_keypoints = 0
        n_labeled = 0
        for sv in svs:
            x_min, x_max, y_min, y_max = self.get_img_breakpoints(sv, interval_pair)
            if self.must_add_to_target(sv):
                n_labeled += 1
                labels.append(self.get_label(sv))
                keypoints.append([[x_min, y_min, self.is_visible_keypoint(x_min, y_min)]])
                boxes.append(self.get_box(x_min, x_max, y_min, y_max))
            n_visible_keypoints += self.is_visible_keypoint(x_min, y_min)
            logging.debug("%s: [%d %d %d]" % (sv, x_min, y_min, self.is_visible_keypoint(x_min, y_min)))
        return {TargetType.labels: torch.as_tensor(labels, dtype=torch.int64),
                TargetType.keypoints: torch.as_tensor(keypoints, dtype=torch.float32),
                TargetType.boxes: torch.as_tensor(boxes, dtype=torch.float32),
                TargetType.gloc: torch.as_tensor(interval_pair.to_list())}, \
            n_visible_keypoints > 0, n_labeled > 0

    def get_img_breakpoints(self, sv, interval_pair):
        x_min = utils.bp_to_pixel(sv.start, interval_pair.intervalA, self.config.heatmap_dim)
        x_max = utils.bp_to_pixel(sv.end, interval_pair.intervalA, self.config.heatmap_dim)
        y_min = self.config.heatmap_dim - x_max
        y_max = self.config.heatmap_dim - x_min
        # adjust y to account for an interval shift in the pair (this works for intervals on the same chr)
        shit_pos = 2 * interval_pair.intervalB.start - interval_pair.intervalA.start
        delta_pixels = utils.bp_to_pixel(shit_pos, interval_pair.intervalB, self.config.heatmap_dim)
        y_min += delta_pixels
        y_max += delta_pixels
        return x_min, x_max, y_min, y_max

    def must_add_to_target(self, sv):
        # positive vs negative examples
        return sv.internal_type in self.config.classes and sv.len >= self.config.min_sv_len

    def get_label(self, sv):
        return SV.get_sv_type_labels(self.config.classes)[sv.internal_type]

    def is_visible_keypoint(self, x, y):
        return 0 < x < self.config.heatmap_dim and 0 < y < self.config.heatmap_dim

    def get_box(self, x_min, x_max, y_min, y_max):
        box_x_min = max(x_min, 0)
        box_x_max = min(x_max, self.config.heatmap_dim)
        box_y_min = min(max(y_min, 0), self.config.heatmap_dim)
        box_y_max = max(min(y_max, self.config.heatmap_dim), 0)
        return [box_x_min, box_y_min, box_x_max, box_y_max]

    def construct_channel(self, channel, interval_pair):
        if channel in [Channel.SRORD12_CDEL]:
            signals = CHANNEL_TO_SIGNAL_COMBO[channel]
            return self.cue_index.intersect(signals, interval_pair.intervalA, interval_pair.intervalB), 0
        elif channel != Channel.RD_DIV_RD:
            # channels with a direct mapping to index signals
            signal = CHANNEL_TO_SIGNAL[channel]
            if signal in SCALAR_SIGNALS:
                return self.cue_index.scalar_apply(CHANNEL_TO_SIGNAL[channel],
                                                   interval_pair.intervalA, interval_pair.intervalB), \
                       -self.config.signal_vmax[channel]
            else:
                return self.cue_index.intersect([signal], interval_pair.intervalA, interval_pair.intervalB), 0
        elif channel == Channel.RD_DIV_RD:
            div_counts, nz1 = self.cue_index.scalar_apply(SVSignal.RD_DIV, interval_pair.intervalA,
                                                          interval_pair.intervalB, op=max)
            rd_counts, nz2 = self.cue_index.scalar_apply(SVSignal.RD, interval_pair.intervalA,
                                                         interval_pair.intervalB, op=max)
            if nz1 and nz2:
                with np.errstate(invalid='ignore', divide='ignore'):
                    counts = div_counts / rd_counts
                    counts[np.isnan(counts)] = 0
            else:
                counts = rd_counts  # all zeros
            return (counts, (nz1 and nz2)), 0

    def is_valid_target(self, target, visible_keypoints):
        return self.config.allow_empty or target is not None and (visible_keypoints or not self.config.visible_only)

    def channel_transform(self, channel, channel_type, vmin):
        channel = channel.transpose()
        channel = np.flip(channel, 0)
        channel = cv2.resize(channel, dsize=(self.config.heatmap_dim, self.config.heatmap_dim),
                             interpolation=cv2.INTER_CUBIC)
        channel = np.clip(channel, vmin, self.config.signal_vmax[channel_type]).astype(float)
        if vmin < 0: channel = 0.5 + 0.5 * channel / self.config.signal_vmax[channel_type]
        else: channel = channel / self.config.signal_vmax[channel_type]
        return channel

    def make_image(self, interval_pair):
        image = np.zeros((self.config.heatmap_dim, self.config.heatmap_dim, self.config.n_channels_saved))
        non_zero = False
        for i, channel_type in enumerate(self.config.channels):
            channel_info, vmin = self.construct_channel(channel_type, interval_pair)
            channel, nz = channel_info
            if nz: non_zero = True
            image[:, :, i] = self.channel_transform(channel, channel_type, vmin)
        return image, non_zero

    def save_image(self, image, idx, interval_pair, garbage_collect=False):
        img_fname = "%d_%s.png" % (idx, str(interval_pair))
        for i in range(0, self.config.n_channels_saved, 3):
            save(image[:, :, i:i + 3], self.config.image_dir + ("split%d/" % (i // 3)) + img_fname,
                 garbage_collect=garbage_collect)

    def __iter__(self):
        for idx, interval_pair in enumerate(TargetIntervalScanner(self.config, self.cue_index)):
            # additional interval filters
            if interval_pair.intervalB.start - interval_pair.intervalA.end > self.config.max_sv_len: continue
            target = None
            # ------- training
            if self.svs:
                target, visible_keypoints, labeled_keypoints = self.get_target(interval_pair)
                if not self.is_valid_target(target, visible_keypoints): continue
                if not labeled_keypoints:
                    target = None
                image, any_signal = self.make_image(interval_pair)
                if not any_signal: continue  # skip images with no signal
                if self.config.store_image:
                    self.save_image(image, idx, interval_pair, garbage_collect=(idx % 50 == 0))
                    torch.save(target, self.config.annotation_dir + "%d_%s.target" % (idx, str(interval_pair)))
            else:
                # ----- inference
                image, any_signal = self.make_image(interval_pair)
                if not any_signal: continue
                if self.config.store_image:
                    self.save_image(image, idx, interval_pair, garbage_collect=(idx % 100 == 0))
                if self.config.view_mode:
                    save_channel_overlay(image.copy(), self.config.image_dir + "%s.png" % (str(interval_pair)), target,
                                         self.config.classes, channels=self.config.channels)

            image = image[:, :, :self.config.n_channels]
            image = transforms.functional.to_tensor(image).float()
            if target is None: target = {}
            image, target = utils.downscale_tensor(image, self.config.image_dim, target)
            target[TargetType.image_id] = torch.tensor([idx])
            target[TargetType.gloc] = torch.as_tensor(interval_pair.to_list())
            yield image, target
            if self.config.dry_run: break

class SVStaticDataset(torch.utils.data.Dataset):
    def __init__(self, config, dataset_id=0):
        super().__init__()
        self.config = config
        self.image_dir = self.config.image_dirs[dataset_id]
        self.max_len = self.config.dataset_lens[dataset_id]
        self.annotation_dir = self.config.annotation_dirs[dataset_id]
        self.target_image_dim = self.config.image_dim
        self.annotations = list(sorted(os.listdir(self.annotation_dir)))
        self.images = {}
        self.n_channels = self.config.n_channels
        channel_set_origin = ChannelSet[config.channel_set_origin]
        channel_set = ChannelSet[self.config.channel_set]
        self.channel_ids = CHANNEL_SET_TO_CHANNEL_IDX[channel_set_origin][channel_set]
        n_channels_origin = len(CHANNELS_BY_TYPE[channel_set_origin])
        n_channels_origin = ((n_channels_origin + 2) // 3) * 3
        for i in range(math.ceil(n_channels_origin / 3)):
            self.images[i] = list(sorted(os.listdir(self.image_dir + ("split%d" % i))))
            assert len(self.images[i]) == len(self.annotations), "Invalid dataset directory structure %d %d" % \
                                                                 (len(self.images[i]), len(self.annotations))
        self.transform = transforms.Compose([transforms.ToTensor()])
        self.dataset_id = dataset_id

    def __len__(self):
        return min(len(self.images[0]), self.max_len, 100 if self.config.dry_run else sys.maxsize)

    def __getitem__(self, index):
        image = []
        for i in range(len(self.images)):
            img_path = os.path.join(self.image_dir + ("split%d" % i), self.images[i][index])
            image.append(matplotlib.image.imread(img_path)[:, :, :3])
        image = np.concatenate(image, axis=2)
        image = np.take(image, self.channel_ids, axis=2)
        target_path = os.path.join(self.annotation_dir, self.annotations[index])
        #sys.modules['img'] = cue.img
        target = torch.load(target_path)
        if target is None:
            target = {TargetType.keypoints: torch.as_tensor([], dtype=torch.float32),
                      TargetType.labels: torch.as_tensor([], dtype=torch.int64)}
        target[TargetType.image_id] = torch.tensor([index])
        if self.transform is not None: image = self.transform(image)
        image, target = utils.downscale_tensor(image, self.target_image_dim, target)
        target[TargetType.dataset_id] = torch.tensor([self.dataset_id])
        return image, target


def collate_fn(batch):
    return zip(*batch)
