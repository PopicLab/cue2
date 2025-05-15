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

import cue.cli.config
import cue.img.datasets
import cue.img.stats
import cue.seq.index
import cue.seq.utils
import joblib
import logging
import numpy
import pickle


def generate(config, chr_names): # generates images/annotations for the specified list of chromosomes
    logging.root.setLevel(logging.INFO)
    for chr_name in chr_names:
        dataset = cue.img.datasets.SVStreamingDataset(config, cue.seq.index.CueIndex.generate_or_load(chr_name, config))
        chr_stats = cue.img.stats.DatasetStats(chr_name, config, dataset)
        for image, target in dataset:
            chr_stats.update(target, image)
        chr_stats.report()
        with open("%s/%s_stats.pkl" % (config.info_dir, chr_name), 'wb') as fp:
            chr_stats.dataset = None
            pickle.dump(chr_stats, fp)
    return True


def main(config_yaml):
    config = cue.cli.config.load_config(config_yaml, config_type=cue.cli.config.ConfigType.GENERATE)
    chr_name_chunks = cue.seq.utils.partition_chrs(config.chr_names, config.n_procs)
    logging.info("Chromosomes/process partition: " + str([numpy.array2string(chk) for chk in chr_name_chunks]))
    joblib.Parallel(n_jobs=config.n_cpus)(joblib.delayed(generate)(config, chr_name_chunks[i])
                                          for i in range(config.n_cpus))
    # generate stats for the entire dataset
    logging.info("Generating dataset-level stats...")
    stats = cue.img.stats.DatasetStats("all", config)
    for chr_name in config.chr_names:
        with open("%s/%s_stats.pkl" % (config.info_dir, chr_name), 'rb') as fp:
            chr_stats = pickle.load(fp)
            stats.merge(chr_stats)
    stats.report()

