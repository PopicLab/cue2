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
import cue.nn.engine
import cue.nn.models
import cue.utils.types
import logging
import torch
torch.manual_seed(0)

def main(cue_config):
    # ------ Initialization ------
    # load the model configs / setup the experiment
    config = cue.cli.config.load_config(cue_config, config_type=cue.cli.config.ConfigType.TRAIN)

    # ---------Training dataset--------
    dataset = torch.utils.data.ConcatDataset(cue.img.datasets.SVStaticDataset(config, dataset_id)
                                             for dataset_id in range(len(config.dataset_dirs)))
    validation_size = int(config.validation_ratio * len(dataset))
    train_size = len(dataset) - validation_size
    train_data, validation_data = torch.utils.data.random_split(dataset, [train_size, validation_size])

    # ---------Data loaders--------
    Stage = cue.utils.types.make_enum('Stage', ["TRAIN", "VALIDATE"], __name__)
    data_loaders = {Stage.TRAIN: torch.utils.data.DataLoader(dataset=train_data, batch_size=config.batch_size,
                                                             shuffle=True, collate_fn=cue.img.datasets.collate_fn),
                    Stage.VALIDATE: torch.utils.data.DataLoader(dataset=validation_data, batch_size=config.batch_size,
                                                                shuffle=False, collate_fn=cue.img.datasets.collate_fn)}
    logging.info("Size of train set: %d; validation set: %d" % (len(data_loaders[Stage.TRAIN]),
                                                                len(data_loaders[Stage.VALIDATE])))

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
        cue.nn.engine.train(model, optimizer, data_loaders[Stage.TRAIN], config, epoch,
                            collect_data_metrics=(epoch == 0))
        torch.save(model.state_dict(), "%s/model.epoch%d.pt" % (config.epoch_dirs[epoch], epoch))
        cue.nn.engine.evaluate(model, data_loaders[Stage.VALIDATE], config, config.device, config.epoch_dirs[epoch],
                        collect_data_metrics=(epoch == 0), given_ground_truth=True, filters=False)
        lr_scheduler.step()
    torch.save(model.state_dict(), config.model_path)
