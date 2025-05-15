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
import cue.img.utils
import cue.nn.engine
import cue.nn.models
import cue.seq.filters
import cue.seq.index
import cue.seq.io as io
import cue.seq.refinery
import cue.seq.sv
import cue.seq.utils
import joblib
import logging
import numpy
import torch
import warnings
warnings.filterwarnings("ignore")


def call(config, device, chr_names):  # runs SV calling on the specified device for the specified list of chromosomes
    logging.root.setLevel(logging.getLevelName(config.logging_level))
    model = cue.nn.models.MultiSVHG(config)
    model.load_state_dict(torch.load(config.model_path, device))
    model.to(device)
    for chr_name in chr_names:
        data_loader = torch.utils.data.DataLoader(
            cue.img.datasets.SVStreamingDataset(config, cue.seq.index.CueIndex.generate(chr_name, config)),
            batch_size=config.batch_size, collate_fn=cue.img.datasets.collate_fn)
        chr_predictions = cue.nn.engine.evaluate(model, data_loader, config, device,
                                        output_dir="%s/%s/" % (config.predictions_dir, chr_name))
        torch.save(chr_predictions, "%s/%s/predictions.pkl" % (config.predictions_dir, chr_name))
    return True

def refine(config, chr_names):  # runs SV bp refinement and NMS filters on the specified list of chromosomes
    for chr_name in chr_names:
        chr_svs = [sv for sv in io.vcf_iter("%s/candidate_svs.%s.vcf" % (config.candidates_dir, chr_name))]
        chr_svs = cue.seq.refinery.refine_svs(chr_name, chr_svs, config, "%s/refine.%s.log" % (config.candidates_dir, chr_name))
        chr_svs = cue.seq.filters.nms1D(chr_svs, cue.seq.sv.SV.compare_by_support)
        io.write_vcf(chr_svs, "%s/refined_svs.%s.vcf" % (config.candidates_dir, chr_name), config.fai)
    return True

def main(config_yaml):
    config = cue.cli.config.load_config(config_yaml, config_type=cue.cli.config.ConfigType.DISCOVER)
    chr_name_chunks = cue.seq.utils.partition_chrs(config.chr_names, config.n_procs)
    logging.info("Chromosomes/process partition: " + str([numpy.array2string(chk) for chk in chr_name_chunks]))

    # ------ 1. Image-based discovery ------
    if not config.skip_inference:
        logging.info("Generating SV predictions...")
        joblib.Parallel(n_jobs=config.n_procs)(joblib.delayed(call)(config, config.devices[i], chr_name_chunks[i])
                                               for i in range(config.n_procs))
    logging.info("Loading image-based SV predictions...")
    image_predictions = {chr_name: torch.load("%s/%s/predictions.pkl" % (config.predictions_dir, chr_name))
                   for chr_name in config.chr_names}

    # ------ 2. Image-to-genome conversion ------
    sv_candidates = [] # genome-based SV candidate calls
    for chr_name in config.chr_names:
        chr_candidates = cue.seq.filters.nms1D(
            [sv for prediction in image_predictions[chr_name]
             for sv in cue.img.utils.img_to_svs(chr_name, prediction, config)], cue.seq.sv.SV.compare_by_score)
        io.write_vcf(chr_candidates, "%s/candidate_svs.%s.vcf" % (config.candidates_dir, chr_name), config.fai)
        sv_candidates.extend(chr_candidates)
    # output VCF with candidate SVs (pre-refinement)
    io.write_vcf(sv_candidates, "%s/cue_candidate_svs.vcf" % config.candidates_dir, config.fai)

    # ------ 3. Breakpoint refinement ------
    logging.info("Refining SV predictions...")
    joblib.Parallel(n_jobs=config.n_procs)(joblib.delayed(refine)(config, chr_name_chunks[i])
                                           for i in range(config.n_procs))

    # ------ 4. Output VCF with final SVs ------
    io.write_vcf([sv for chr_name in config.chr_names
                  for sv in io.vcf_iter("%s/refined_svs.%s.vcf" % (config.candidates_dir, chr_name))],
                 "%s/cue_svs.vcf" % config.report_dir, config.fai,
                 min_score=config.min_qual_score,
                 min_len=config.min_sv_len)
    logging.info("****************")
    logging.info("Final SV calls written to: %s" % "%s/cue_svs.vcf" % config.report_dir)
    logging.info("****************")

