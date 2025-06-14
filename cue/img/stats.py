import cue.seq.intervals
from cue.img.constants import *
from cue.seq.constants import *
from cue.img.plotting import save_channel_overlay
from collections import defaultdict, Counter
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path


class DatasetStats:
    def __init__(self, dataset_name, config, dataset=None):
        self.dataset_name = dataset_name
        self.dataset = dataset
        self.config = config
        self.classes = config.classes
        self.channels = config.channels
        if dataset:
            self.aln_index = dataset.cue_index
            self.chr_index = dataset.cue_index.chr_index
            self.img_dir = {}
            for sv_type in self.classes:
                self.img_dir[sv_type] = "%s/%s/%s/" % (self.config.info_dir, dataset_name, sv_type.name)
                Path(self.img_dir[sv_type]).mkdir(parents=True, exist_ok=True)
        self.num_images = 0
        self.no_labels = 0
        self.visible_svs_per_image = []
        self.all_svs_per_image = []
        self.label2all = defaultdict(list)
        self.label2visible = defaultdict(list)
        self.label2profile = defaultdict(set)
        self.svs = defaultdict(int)
        self.sv2count = defaultdict(int)
        self.sv2len = defaultdict(list)
        self.sv2count_all = defaultdict(int)

    def merge(self, dataset_stats):
        self.num_images += dataset_stats.num_images
        self.no_labels += dataset_stats.no_labels
        self.svs.update(dataset_stats.svs)
        self.sv2count.update({k: self.sv2count[k] + dataset_stats.sv2count[k] for k in dataset_stats.sv2count})
        self.sv2count_all.update({k: self.sv2count_all[k] + dataset_stats.sv2count_all[k] for k in dataset_stats.sv2count_all})
        self.sv2len.update({k: self.sv2len[k] + dataset_stats.sv2len[k] for k in dataset_stats.sv2len})

    def update(self, target, image=None):
        if target is None: return
        if image is not None: image = image.permute(1, 2, 0).cpu().numpy()
        self.num_images += 1
        if TargetType.labels not in target:
            self.no_labels += 1
            return
        # count the number of SVs in the target
        label2visible = defaultdict(int)
        label2all = defaultdict(int)
        for i in range(len(target[TargetType.labels])):
            if any([p[2] for p in target[TargetType.keypoints][i]]):
                label2visible[target[TargetType.labels][i].item()] += 1
            label2all[target[TargetType.labels][i].item()] += 1
        for label, count in label2all.items(): self.label2all[label].append(count)
        for label, count in label2visible.items(): self.label2visible[label].append(count)
        self.visible_svs_per_image.append(sum([label2visible[label] for label in label2visible]))
        self.all_svs_per_image.append(len(target[TargetType.labels]))

        # compute the channel profile
        if image is not None and len(label2visible) == 1:
            profile = []
            for i in range(len(self.channels)):
                channel = self.channels[i]
                if channel in CHANNEL_TO_SIGNAL and CHANNEL_TO_SIGNAL[channel] in SCALAR_SIGNALS:
                    if np.min(image[:, :, i]) != 0.5 or np.max(image[:, :, i]) != 0.5:
                        profile.append(channel.name)
                else:
                    if np.sum(image[:, :, i]) > 0:
                        profile.append(channel.name)
            (label, _), = label2visible.items()
            self.label2profile[label].add(";".join(profile))

        # retrieve the SVs in ground truth
        if self.dataset and self.dataset.svs:
            interval_pair = cue.seq.intervals.GenomeIntervalPair.from_list(target[TargetType.gloc].tolist())
            svs = self.dataset.svs.contained(interval_pair.intervalA, interval_pair.intervalB)
            for sv in svs:
                sv_repr = (sv.chr_name, sv.internal_type, sv.start, sv.end)
                sv_len = sv.end - sv.start
                if sv_repr not in self.svs:
                    self.sv2count[sv.internal_type] += 1
                    self.sv2len[sv.internal_type].append(sv_len)
                self.svs[sv_repr] += 1
                self.sv2count_all[sv.internal_type] += 1
                if self.sv2count[sv.internal_type] < 20:
                    save_channel_overlay(image, self.img_dir[sv.internal_type] + str(self.sv2count[sv.internal_type]) + \
                            "_" + str(sv.start) + "_" + str(sv.end) + "_" + str(sv_len),
                                         None, self.config.classes, channels=self.channels)

    def batch_update(self, batch):
        for target in batch:
            self.update(target)

    def report(self):
        report_fname = "%s/%s_report.txt" % (self.config.info_dir, self.dataset_name)
        with open(report_fname, 'wt') as report:
            print("=====================", file=report)
            print("Number of images: %d" % self.num_images, file=report)
            print("Number of images without labels: %d" % self.no_labels, file=report)
            if self.all_svs_per_image:
                print("Number of SVs per image:", file=report)
                print(Counter(self.all_svs_per_image), file=report)
                for label, counts in self.label2visible.items():
                    print("Number of %s per image:" % self.classes[label], file=report)
                    print(Counter(counts), file=report)
                print("Number of visible SVs per image:", file=report)
                print(Counter(self.visible_svs_per_image), file=report)
            for label, counts in self.label2visible.items():
                print("Number of visible %s per image:" % self.classes[label], file=report)
                print(Counter(counts), file=report)
            for sv_type, count in self.sv2count.items():
                print("Number of unique %s: %d" % (sv_type, count), file=report)
            print("Number of unique SVs total: %d" % sum(self.sv2count.values()), file=report)
            print("Number of unique SVs (by repr): %d" % len(self.svs), file=report)    
            if self.dataset and self.dataset.svs:
                print("Number of SVs in genome: %d" % self.dataset.svs.size(), file=report)
                print("Number of missing SVs: %d" % (self.dataset.svs.size() - len(self.svs)), file=report)
                n_reported = 0
                for sv in self.dataset.svs:
                    if n_reported > 10:
                        break
                    sv_repr = (sv.chr_name, sv.internal_type.name, sv.start, sv.end)
                    if sv.chr_name == self.dataset_name and sv_repr not in self.svs:
                        print("Missing: %s, len %d" % (sv_repr, sv.end - sv.start), file=report)
                        n_reported += 1

            for sv_type, count in self.sv2count_all.items():
                print("Number of all %s: %d" % (sv_type, count), file=report)
            
            if self.sv2count_all:
                print("Number of all SVs total: %d" % sum(self.sv2count_all.values()), file=report)
            if self.svs:
                print("Max number of images with same SV: %d" % max(self.svs.values()), file=report)
            for sv_type in self.sv2len:
                print("Max len of %s: %d" % (sv_type, max(self.sv2len[sv_type])), file=report)
                print("Min len of %s: %d" % (sv_type, min(self.sv2len[sv_type])), file=report) 

            for label in self.label2profile:
                print(self.classes[label] + ": ", file=report)
                print("Profile = " + str(self.label2profile[label]), file=report)

        fig_name = "%s/%s_sv_lens.png" % (self.config.info_dir, self.dataset_name)
        fig = plt.figure(figsize=(10, 10))
        plt.subplots_adjust(hspace=.4)
        rows = min(int(np.sqrt(len(self.sv2len))), 4)
        cols = min(int(np.sqrt(len(self.sv2len))), 4)
        sv_types = list(self.sv2len.keys())
        for i in range(rows * cols):
            plt.subplot(rows, cols, i + 1)
            plt.hist(self.sv2len[sv_types[i]])  # density=False would make counts
            plt.title(sv_types[i])
        plt.savefig(fig_name)
        plt.close(fig)

