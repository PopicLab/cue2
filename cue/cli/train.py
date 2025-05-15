# MIT License
#
# Copyright (c) 2022 Victoria Popic
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
# 
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import cue.cli.config
import cue.img.datasets
import cue.nn.engine
import cue.nn.models
import argparse
from enum import Enum
import logging
import torch
torch.manual_seed(0)

def main(cue_config):
    # ------ Initialization ------
    # load the model configs / setup the experiment
    config = cue.cli.config.load_config(cue_config, config_type=cue.cli.config.CONFIG_TYPE.TRAIN)

    # ---------Training dataset--------
    dataset = torch.utils.data.ConcatDataset(cue.img.datasets.SVStaticDataset(config, dataset_id)
                                             for dataset_id in range(len(config.dataset_dirs)))
    validation_size = int(config.validation_ratio * len(dataset))
    train_size = len(dataset) - validation_size
    train_data, validation_data = torch.utils.data.random_split(dataset, [train_size, validation_size])
    # images, targets = next(iter(torch.utils.data.DataLoader(dataset=dataset,
    #                                                         batch_size=min(len(dataset), 400), shuffle=True,
    #                                                         collate_fn=datasets.collate_fn)))

    # ---------Data loaders--------
    STAGE = Enum('STAGE', 'TRAIN VALIDATE')
    data_loaders = {STAGE.TRAIN: torch.utils.data.DataLoader(dataset=train_data, batch_size=config.batch_size,
                                                             shuffle=True, collate_fn=cue.img.datasets.collate_fn),
                    STAGE.VALIDATE: torch.utils.data.DataLoader(dataset=validation_data, batch_size=config.batch_size,
                                                                shuffle=False, collate_fn=cue.img.datasets.collate_fn)}
    logging.info("Size of train set: %d; validation set: %d" % (len(data_loaders[STAGE.TRAIN]),
                                                                len(data_loaders[STAGE.VALIDATE])))

    # ---------Model--------
    model = cue.nn.models.MultiSVHG(config)
    if config.pretrained_model is not None:
        model.load_state_dict(torch.load(config.pretrained_model, config.device))
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=config.learning_rate_decay_interval,
                                                   gamma=config.learning_rate_decay_factor)
    model.to(config.device)

    # ------ Training ------
    for epoch in range(config.num_epochs):
        cue.nn.engine.train(model, optimizer, data_loaders[STAGE.TRAIN], config, epoch, collect_data_metrics=(epoch == 0))
        torch.save(model.state_dict(), "%s.epoch%d" % (config.model_path, epoch))
        cue.nn.engine.evaluate(model, data_loaders[STAGE.VALIDATE], config, config.device, config.epoch_dirs[epoch],
                        collect_data_metrics=(epoch == 0), given_ground_truth=True, filters=False)
        lr_scheduler.step()
    torch.save(model.state_dict(), config.model_path)
