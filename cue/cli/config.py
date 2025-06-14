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

from cue.img.constants import *
from cue.seq.constants import *
import cue.seq.io
import cue.seq.sv
import cue.utils.types
import logging
import math
import os
from pathlib import Path
import sys
import torch
import yaml
logging.getLogger('matplotlib').setLevel(logging.ERROR)
logging.getLogger('tensorflow').setLevel(logging.WARNING)

SHARED_DEFAULTS = {
    'logging_level': "INFO",
    'class_set': "SINGLE5G",
    'channel_set': "LONG",
    'channel_set_origin': "LONG",
    'heatmap_dim': 1000,
    'image_dim': 256,
    'n_cpus': 1,
    'gpu_ids': [],
    'n_jobs_per_gpu': 1,
    'dry_run': False
}

DATA_DEFAULTS = {
    'chr_names': None,
    'vcf': None,
    'fixed_targets': None,
    'high_redundancy': False,
    'interval_size': 3000,
    'step_size': 1000,
    'bin_size': 15,
    'min_sv_len': 40,
    'max_sv_len': 1000000,
    'min_discord_len': 40,
    'max_discord_len': 1000000,
    'min_clipped_len': 40,
    'min_adj_support': 2,
    'min_mapq': 10,
    'high_mapq': 60,
    'min_frag_mapq': 10,
    'min_qual_score': 30,
    'view_mode': False,
    'store_image': False,
    'store_adj': True,
    'max_discordance': 0.05,
    'frag_len': 50,
    'signal_vmax': {channel: 200 for channel in Channel.__members__} | {"SM": 100, "RD_DIV_RD": 1, "LL_RR_LR": 1}
}
MODEL_DEFAULTS = {
    'batch_size': 16,
    'sigma': 10,
    'stride': 4,
    'heatmap_peak_threshold': 0.4,
    'report_interval': 1,
}
DISCOVER_DEFAULTS = {**SHARED_DEFAULTS, **DATA_DEFAULTS, **MODEL_DEFAULTS}
DISCOVER_DEFAULTS.update({
    'skip_inference': False,
    'refine_min_support': 2,
})
GENERATE_DEFAULTS = {**SHARED_DEFAULTS, **DATA_DEFAULTS}
GENERATE_DEFAULTS.update({
    'allow_empty': False,
    'visible_only': True,
    'store_image': True,
})
TRAIN_DEFAULTS = {**SHARED_DEFAULTS, **MODEL_DEFAULTS}
TRAIN_DEFAULTS.update({
    'num_epochs': 16,
    'report_interval': 50,
    'model_checkpoint_interval': 10000,
    'plot_confidence_maps': False,
    'validation_ratio': 0.1,
    'learning_rate': 0.0001,
    'learning_rate_decay_interval': 5,
    'learning_rate_decay_factor': 1,
})

class Config:
    def __init__(self, config_file):
        self.config_file = config_file
        self.experiment_dir = str(Path(config_file).parent.resolve())
        self.log_dir = self.experiment_dir + "/logs/"

        # -------- setup logging
        Path(self.log_dir).mkdir(parents=True, exist_ok=True)
        self.log_file = self.log_dir + 'main.log'
        # noinspection PyArgumentList
        logging.basicConfig(format='[%(levelname)s] %(message)s', level=logging.getLevelName(self.logging_level),
                            handlers=[logging.FileHandler(self.log_file, mode='w'), logging.StreamHandler(sys.stdout)])

        # -------- setup shared data/model configs
        self.classes = cue.seq.sv.TYPE_SET_TO_TYPES[cue.seq.sv.SVTypeSet[self.class_set]]
        self.channels = CHANNELS_BY_TYPE[ChannelSet[self.channel_set]]
        self.n_classes = len(self.classes)
        self.n_channels = len(self.channels)
        self.n_channels_saved = ((self.n_channels + 2) // 3) * 3

    def set_defaults(self, default_values_dict):
        for k, v, in default_values_dict.items():
            if not hasattr(self, k):
                self.__setattr__(k, v)

    def __str__(self):
        s = "==== Config ====\n\t"
        s += '\n\t'.join("{}: {}".format(k, v) for k, v in self.__dict__.items())
        return s


class DiscoverConfig(Config):
    def __init__(self, config_file, **entries):
        self.set_defaults(DISCOVER_DEFAULTS)
        self.__dict__.update(entries)
        super().__init__(config_file)

        # -------- set up computational resources
        self.devices = []
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, self.gpu_ids))
        if len(self.gpu_ids) > 0 and torch.cuda.is_available():
            for i in range(self.n_jobs_per_gpu * len(self.gpu_ids)):
                self.devices.append(torch.device("cuda:%d" % int(i / self.n_jobs_per_gpu)))
        else:
            for _ in range(self.n_cpus):
                self.devices.append(torch.device("cpu"))
        self.n_procs = len(self.devices)

        # -------- setup data inputs
        self.fai = "%s.fai" % self.fa
        if self.chr_names is None:
            chr_index = cue.seq.io.ChrFAIndex.load(self.fai)
            self.chr_names = list(chr_index.chr_names())

        # -------- setup the output directory structure
        self.report_dir = self.experiment_dir + "/results/"
        self.candidates_dir = self.report_dir + "/candidates/"
        self.index_dir = self.report_dir + "/index/"
        self.predictions_dir = self.report_dir + "/predictions/"
        self.image_dir = self.experiment_dir + "/images/"
        for i in range(math.ceil(self.n_channels / 3)):
            Path(self.image_dir + ("split%d" % i)).mkdir(parents=True, exist_ok=True)
        Path(self.report_dir).mkdir(parents=True, exist_ok=True)
        Path(self.candidates_dir).mkdir(parents=True, exist_ok=True)
        Path(self.index_dir).mkdir(parents=True, exist_ok=True)
        Path(self.predictions_dir).mkdir(parents=True, exist_ok=True)
        Path(self.image_dir).mkdir(parents=True, exist_ok=True)
        for chr_name in self.chr_names:
            predictions_dir = "%s/%s/" % (self.predictions_dir, chr_name)
            Path(predictions_dir).mkdir(parents=True, exist_ok=True)
        logging.info(self)


class TrainConfig(Config):
    def __init__(self, config_file, **entries):
        self.set_defaults(TRAIN_DEFAULTS)
        self.__dict__.update(entries)
        super().__init__(config_file)
        self.info_dir = self.experiment_dir + "/info/"
        self.report_dir = self.experiment_dir + "/results/"
        self.image_dirs = [dataset_dir + "/images/" for dataset_dir in self.dataset_dirs]
        self.annotation_dirs = [dataset_dir + "/annotations/" for dataset_dir in self.dataset_dirs]
        self.model_path = self.experiment_dir + "/model.pt"
        self.epoch_dirs = []
        # ------ setup the training directory structure
        Path(self.info_dir).mkdir(parents=True, exist_ok=True)
        for epoch in range(self.num_epochs):
            output_dir = "%s/epoch%d/" % (self.report_dir, epoch)
            Path(output_dir).mkdir(parents=True, exist_ok=True)
            self.epoch_dirs.append(output_dir)
        # -------- set up computational resources (only 1 GPU supported)
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, self.gpu_ids))
        if len(self.gpu_ids) > 0 and torch.cuda.is_available():
            self.device = torch.device("cuda")
        self.n_procs = 1
        logging.info(self)


class GenerateConfig(Config):
    def __init__(self, config_file, **entries):
        self.set_defaults(GENERATE_DEFAULTS)
        self.__dict__.update(entries)
        super().__init__(config_file)
        self.dataset_dir = str(Path(config_file).parent.resolve())
        self.index_dir = self.experiment_dir + "/index/"
        self.info_dir = self.experiment_dir + "/info/"
        self.image_dir = self.experiment_dir + "/images/"
        self.annotation_dir = self.experiment_dir + "/annotations/"
        self.annotated_images_dir = self.experiment_dir + "/annotated_images/"
        self.n_procs = self.n_cpus

        # ------ setup the imageset directory structure
        Path(self.index_dir).mkdir(parents=True, exist_ok=True)
        Path(self.info_dir).mkdir(parents=True, exist_ok=True)
        Path(self.image_dir).mkdir(parents=True, exist_ok=True)
        for i in range(math.ceil(self.n_channels / 3)):
            Path(self.image_dir + ("split%d" % i)).mkdir(parents=True, exist_ok=True)
        Path(self.annotation_dir).mkdir(parents=True, exist_ok=True)
        Path(self.annotated_images_dir).mkdir(parents=True, exist_ok=True)
        self.fai = "%s.fai" % self.fa
        if self.chr_names is None:
            chr_index = cue.seq.io.ChrFAIndex.load(self.fai)
            self.chr_names = list(chr_index.chr_names())
        logging.info(self)


ConfigType = cue.utils.types.make_enum("ConfigType", ["GENERATE", "TRAIN", "DISCOVER"], __name__)

def load_config(fname, config_type):
    with open(fname) as file: config = yaml.load(file, Loader=yaml.FullLoader)
    if config_type == ConfigType.TRAIN: return TrainConfig(fname, **config)
    elif config_type == ConfigType.DISCOVER: return DiscoverConfig(fname, **config)
    elif config_type == ConfigType.GENERATE: return GenerateConfig(fname, **config)
