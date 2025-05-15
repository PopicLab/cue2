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


from cue.img import plotting
from cue.img.constants import TargetType
import cue.img.filters
import cue.img.stats
import cue.nn.coco_metrics
from collections import defaultdict
import torch
import logging

class MetricTracker:
    def __init__(self, report_interval, prefix=""):
        self.metrics = defaultdict(float)
        self.n_batch_updates = 0
        self.report_interval = report_interval
        self.prefix = prefix

    def batch_update(self, loss_dict):
        for loss_metric, value in loss_dict.items():
            self.metrics[loss_metric] += value.item()
        self.n_batch_updates += 1
        self.report()

    def report(self):
        if self.n_batch_updates % self.report_interval != 0: pass
        logging.info('%s [%d] %s ' % (self.prefix, self.n_batch_updates,
                                      " | ".join(['%s: %.3f' % (metric, self.metrics[metric] / self.n_batch_updates)
                                    for metric, value in self.metrics.items()])))

def train(model, optimizer, data_loader, config, epoch, collect_data_metrics=False):
    metrics_tracker = MetricTracker(config.report_interval, prefix="TRAIN EPOCH %d" % epoch)
    data_stats = cue.img.stats.DatasetStats("TRAIN", config)
    output_dir = config.epoch_dirs[epoch]
    model.train()
    torch.set_grad_enabled(True)
    for batch_id, (images, targets) in enumerate(data_loader):
        if collect_data_metrics:
            data_stats.batch_update(targets)
        images = list(image.to(config.device) for image in images)
        targets = [{k: v.to(config.device) for k, v in t.items()} for t in targets]
        losses, outputs = model(images, targets)
        total_loss = sum(loss for loss in losses.values())
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
        metrics_tracker.batch_update(losses)
        outputs = [{k: v.to(torch.device("cpu")) for k, v in t.items()} for t in outputs]
        for target, output, image in zip(targets, outputs, images):
            output[TargetType.image_id] = target[TargetType.image_id]
            if config.plot_confidence_maps and TargetType.heatmaps in output and \
                    len(target[TargetType.keypoints]) > 3:
                plotting.plot_heatmap_channels(image.cpu(), output[TargetType.heatmaps],
                                               fig_name="%s/heatmaps.train.%d.png" %
                                                        (output_dir, output["image_id"].item()))
                plotting.plot_heatmap_channels(image.cpu(), target[TargetType.heatmaps].cpu(),
                                               fig_name="%s/heatmaps.train.%d.gt.png" %
                                                        (output_dir, output["image_id"].item()))
                plotting.save(image.permute(1, 2, 0).cpu().numpy(), "%s/heatmaps.train.%d.orig.png" %
                              (output_dir, output["image_id"].item()))
        if batch_id and batch_id % config.report_interval == 0:
            plotting.plot_images(images, outputs, range(len(images)), config.classes, targets2=targets,
                                 fig_name="%s/train.batch%d.png" % (output_dir, batch_id))
        if batch_id and batch_id % config.model_checkpoint_interval == 0:
            torch.save(model.state_dict(), "%s.epoch%d.batch%d" % (config.model_path, epoch, batch_id))

    if collect_data_metrics:
        data_stats.report()

def evaluate(model, data_loader, config, device, output_dir, collect_data_metrics=False, given_ground_truth=False,
             filters=True, coco=True):
    coco_evaluator = cue.nn.coco_metrics.CocoKeypointEvaluator(config.classes, output_dir)
    data_stats = cue.img.stats.DatasetStats("EVAL", config)
    data_loader.shuffle = False
    model.eval()
    torch.set_grad_enabled(False)
    outputs = []
    for batch_id, (images, targets) in enumerate(data_loader):
        images = list(image.to(device) for image in images)
        predictions = model(images, targets)
        predictions = [{k: v.to(torch.device("cpu")) for k, v in t.items()} for t in predictions]

        for target, output in zip(targets, predictions):
            output[TargetType.image_id] = target[TargetType.image_id]
            if TargetType.gloc in target:
                output[TargetType.gloc] = target[TargetType.gloc]
        outputs.extend(predictions)
        if filters: # apply image-based filters
            cue.img.filters.filter_keypoints(predictions, config)
        if config.report_interval is not None and batch_id % config.report_interval == 0:
            plotting.plot_images(images, predictions, range(len(images)), config.classes,
                                 fig_name="%s/predictions.batch%d.png" % (output_dir, batch_id),
                                 targets2=targets)
        if given_ground_truth:
            if coco:
                coco_evaluator.batch_update(predictions, zip(images, targets))
            if collect_data_metrics:
                data_stats.batch_update(targets)

    if given_ground_truth:
        if coco:
            coco_evaluator.report()
        if collect_data_metrics:
            data_stats.report()
    return outputs
