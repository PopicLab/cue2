from cue.img.constants import TargetType, KP_FILTERED
import cue.seq.intervals
import cue.seq.sv
import logging
import torch.nn.functional

def upscale_keypoints(keypoints, ratio):
    return keypoints * ratio

def downscale_target(target, ratio):
    if TargetType.boxes in target:
        target[TargetType.boxes] = target[TargetType.boxes]//ratio
    if TargetType.keypoints in target:
        for i in range(len(target[TargetType.keypoints])):
            for j in range(len(target[TargetType.keypoints][i])):
                target[TargetType.keypoints][i][j][:2] = target[TargetType.keypoints][i][j][:2]//ratio
    return target

def downscale_tensor(image, to_image_dim, target=None):
    ratio = image.shape[1] / to_image_dim
    if target is not None:
        target = downscale_target(target, ratio)
    return torch.nn.functional.interpolate(image.unsqueeze(0), size=(to_image_dim, to_image_dim)).squeeze(0), target

def downscale_image(image, to_image_dim, target=None):
    image_dim_orig = image.shape[0]
    assert image_dim_orig >= to_image_dim, "Input image size cannot be smaller than the target image size"
    assert image_dim_orig % to_image_dim == 0, "Input image size must be a multiple of the target image size"
    ratio = int(image_dim_orig / to_image_dim)
    if target is not None: target = downscale_target(target, ratio)
    return image.reshape((to_image_dim, ratio, to_image_dim, ratio, 3)).max(3).max(1), target

def bp_to_pixel(genome_pos, genome_interval, pixels_in_interval):
    bp_per_pixel = len(genome_interval) / pixels_in_interval
    return int((genome_pos - genome_interval.start) / bp_per_pixel)

def pixel_to_bp(pixel_pos, genome_interval, pixels_in_interval):
    bp_per_pixel = len(genome_interval) / pixels_in_interval
    return int(round(pixel_pos * bp_per_pixel)) + genome_interval.start

def batch_images(images):
    batch_shape = [len(images)] + list(images[0].shape)
    batched_imgs = images[0].new_full(batch_shape, 0)
    for img, pad_img in zip(images, batched_imgs):
        pad_img[: img.shape[0], : img.shape[1], : img.shape[2]].copy_(img)
    return batched_imgs

def img_to_svs(chr_name, target, config, max_predictions=20, max_high_predictions=10):
    if len(target[TargetType.labels]) > max_predictions: return []
    svs = []
    genome_interval_pair = cue.seq.intervals.GenomeIntervalPair.from_list(target[TargetType.gloc].tolist())
    n_high_score_svs = 0
    for sv_idx in range(len(target[TargetType.labels])):
        x, y, v = target[TargetType.keypoints][sv_idx].tolist()[0]  # top left corner
        y = config.image_dim - y
        x_bp = pixel_to_bp(x, genome_interval_pair.intervalA, config.image_dim)
        y_bp = pixel_to_bp(y, genome_interval_pair.intervalB, config.image_dim)
        score = int(float(target[TargetType.scores][sv_idx].item()) * 100)
        if score == 1:
            n_high_score_svs += 1
            if n_high_score_svs > max_high_predictions: return [] # failed image
        sv_type, gt = cue.seq.sv.SV.parse_internal_type(config.classes[target[TargetType.labels][sv_idx]])
        logging.debug("img2sv: %s %s %d %d %d %d" % (genome_interval_pair, sv_type, x, y, x_bp, y_bp))
        sv = cue.seq.sv.SV(sv_type, chr_name, x_bp, y_bp, score, gt)
        # skip detections below the diagonal (y < x) or duplicates
        if sv.end > sv.start and v != KP_FILTERED: svs.append(sv)
    return svs
