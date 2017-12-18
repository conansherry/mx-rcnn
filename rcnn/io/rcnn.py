"""
Fast R-CNN:
data =
    {'data': [num_images, c, h, w],
    'rois': [num_rois, 5]}
label =
    {'label': [num_rois],
    'bbox_target': [num_rois, 4 * num_classes],
    'bbox_weight': [num_rois, 4 * num_classes]}
roidb extended format [image_index]
    ['image', 'height', 'width', 'flipped',
     'boxes', 'gt_classes', 'gt_overlaps', 'max_classes', 'max_overlaps', 'bbox_targets']
"""

import numpy as np
import numpy.random as npr
import cv2

from ..config import config
from ..io.image import get_image, tensor_vstack
from ..processing.bbox_transform import bbox_overlaps, bbox_transform, bbox_pred
from ..processing.bbox_regression import expand_bbox_regression_targets


def get_rcnn_testbatch(roidb):
    """
    return a dict of testbatch
    :param roidb: ['image', 'flipped'] + ['boxes']
    :return: data, label, im_info
    """
    assert len(roidb) == 1, 'Single batch only'
    imgs, roidb = get_image(roidb)
    im_array = imgs[0]
    im_info = np.array([roidb[0]['im_info']], dtype=np.float32)

    im_rois = roidb[0]['boxes']
    rois = im_rois
    batch_index = 0 * np.ones((rois.shape[0], 1))
    rois_array = np.hstack((batch_index, rois))[np.newaxis, :]

    data = {'data': im_array,
            'rois': rois_array}
    label = {}

    return data, label, im_info


def get_rcnn_batch(roidb):
    """
    return a dict of multiple images
    :param roidb: a list of dict, whose length controls batch size
    ['images', 'flipped'] + ['gt_boxes', 'boxes', 'gt_overlap'] => ['bbox_targets']
    :return: data, label
    """
    num_images = len(roidb)
    imgs, roidb = get_image(roidb)
    im_array = tensor_vstack(imgs)

    assert config.TRAIN.BATCH_ROIS % config.TRAIN.BATCH_IMAGES == 0, \
        'BATCHIMAGES {} must divide BATCH_ROIS {}'.format(config.TRAIN.BATCH_IMAGES, config.TRAIN.BATCH_ROIS)
    rois_per_image = config.TRAIN.BATCH_ROIS / config.TRAIN.BATCH_IMAGES
    fg_rois_per_image = np.round(config.TRAIN.FG_FRACTION * rois_per_image).astype(int)

    rois_array = list()
    labels_array = list()
    bbox_targets_array = list()
    bbox_weights_array = list()

    for im_i in range(num_images):
        roi_rec = roidb[im_i]

        # infer num_classes from gt_overlaps
        num_classes = roi_rec['gt_overlaps'].shape[1]

        # label = class RoI has max overlap with
        rois = roi_rec['boxes']
        labels = roi_rec['max_classes']
        overlaps = roi_rec['max_overlaps']
        bbox_targets = roi_rec['bbox_targets']

        im_rois, labels, bbox_targets, bbox_weights = \
            sample_rois(rois, fg_rois_per_image, rois_per_image, num_classes,
                        labels, overlaps, bbox_targets)

        # project im_rois
        # do not round roi
        rois = im_rois
        batch_index = im_i * np.ones((rois.shape[0], 1))
        rois_array_this_image = np.hstack((batch_index, rois))
        rois_array.append(rois_array_this_image)

        # add labels
        labels_array.append(labels)
        bbox_targets_array.append(bbox_targets)
        bbox_weights_array.append(bbox_weights)

    rois_array = np.array(rois_array)
    labels_array = np.array(labels_array)
    bbox_targets_array = np.array(bbox_targets_array)
    bbox_weights_array = np.array(bbox_weights_array)

    data = {'data': im_array,
            'rois': rois_array}
    label = {'label': labels_array,
             'bbox_target': bbox_targets_array,
             'bbox_weight': bbox_weights_array}

    return data, label


def sample_rois(rois, fg_rois_per_image, rois_per_image, num_classes,
                labels=None, overlaps=None, bbox_targets=None, gt_boxes=None):
    """
    generate random sample of ROIs comprising foreground and background examples
    :param rois: all_rois [n, 4]; e2e: [n, 5] with batch_index
    :param fg_rois_per_image: foreground roi number
    :param rois_per_image: total roi number
    :param num_classes: number of classes
    :param labels: maybe precomputed
    :param overlaps: maybe precomputed (max_overlaps)
    :param bbox_targets: maybe precomputed
    :param gt_boxes: optional for e2e [n, 5] (x1, y1, x2, y2, cls)
    :return: (labels, rois, bbox_targets, bbox_weights)
    """
    if labels is None:
        overlaps = bbox_overlaps(rois[:, 1:].astype(np.float), gt_boxes[:, :4].astype(np.float))
        gt_assignment = overlaps.argmax(axis=1)
        overlaps = overlaps.max(axis=1)
        labels = gt_boxes[gt_assignment, 4]

    # foreground RoI with FG_THRESH overlap
    fg_indexes = np.where(overlaps >= config.TRAIN.FG_THRESH)[0]
    # guard against the case when an image has fewer than fg_rois_per_image foreground RoIs
    fg_rois_per_this_image = np.minimum(fg_rois_per_image, fg_indexes.size)
    # Sample foreground regions without replacement
    if len(fg_indexes) > fg_rois_per_this_image:
        fg_indexes = npr.choice(fg_indexes, size=fg_rois_per_this_image, replace=False)

    # Select background RoIs as those within [BG_THRESH_LO, BG_THRESH_HI)
    bg_indexes = np.where((overlaps < config.TRAIN.BG_THRESH_HI) & (overlaps >= config.TRAIN.BG_THRESH_LO))[0]
    # Compute number of background RoIs to take from this image (guarding against there being fewer than desired)
    bg_rois_per_this_image = rois_per_image - fg_rois_per_this_image
    bg_rois_per_this_image = np.minimum(bg_rois_per_this_image, bg_indexes.size)
    # Sample foreground regions without replacement
    if len(bg_indexes) > bg_rois_per_this_image:
        bg_indexes = npr.choice(bg_indexes, size=bg_rois_per_this_image, replace=False)

    # indexes selected
    keep_indexes = np.append(fg_indexes, bg_indexes)
    neg_idx = np.where(overlaps < config.TRAIN.FG_THRESH)[0]
    neg_rois = rois[neg_idx]
    # pad more to ensure a fixed minibatch size
    while keep_indexes.shape[0] < rois_per_image:
        gap = np.minimum(len(neg_rois), rois_per_image - keep_indexes.shape[0])
        gap_indexes = npr.choice(range(len(neg_rois)), size=gap, replace=False)
        keep_indexes = np.append(keep_indexes, neg_idx[gap_indexes])

    # select labels
    labels = labels[keep_indexes]
    # set labels of bg_rois to be 0
    labels[fg_rois_per_this_image:] = 0
    rois = rois[keep_indexes]

    # load or compute bbox_target
    if bbox_targets is not None:
        bbox_target_data = bbox_targets[keep_indexes, :]
    else:
        targets = bbox_transform(rois[:, 1:], gt_boxes[gt_assignment[keep_indexes], :4])
        if config.TRAIN.BBOX_NORMALIZATION_PRECOMPUTED:
            targets = ((targets - np.array(config.TRAIN.BBOX_MEANS))
                       / np.array(config.TRAIN.BBOX_STDS))
        bbox_target_data = np.hstack((labels[:, np.newaxis], targets))

    bbox_targets, bbox_weights = \
        expand_bbox_regression_targets(bbox_target_data, num_classes)

    return rois, labels, bbox_targets, bbox_weights

def sample_rois_fpn(rois, fg_rois_per_image, rois_per_image, num_classes,
                    labels=None, overlaps=None, bbox_targets=None, gt_boxes=None):
    """
    generate random sample of ROIs comprising foreground and background examples
    :param rois: all_rois [n, 4]; e2e: [n, 5] with batch_index
    :param fg_rois_per_image: foreground roi number
    :param rois_per_image: total roi number
    :param num_classes: number of classes
    :param labels: maybe precomputed
    :param overlaps: maybe precomputed (max_overlaps)
    :param bbox_targets: maybe precomputed
    :param gt_boxes: optional for e2e [n, 5] (x1, y1, x2, y2, cls)
    :return: (rois, labels, bbox_targets, bbox_weights)
    """
    DEBUG = False
    if labels is None:
        if len(gt_boxes) == 0:
            gt_boxes = np.zeros((1, 5))
            gt_assignment = np.zeros((len(rois), ), dtype=np.int32)
            overlaps = np.zeros((len(rois), ))
            labels = np.zeros((len(rois), ))
        else:
            overlaps = bbox_overlaps(rois[:, 1:].astype(np.float), gt_boxes[:, :4].astype(np.float))
            gt_assignment = overlaps.argmax(axis=1)
            overlaps = overlaps.max(axis=1)
            labels = gt_boxes[gt_assignment, 4]

    num_rois = rois.shape[0]
    # foreground RoI with FG_THRESH overlap
    fg_indexes = np.where(overlaps >= config.TRAIN.FG_THRESH)[0]
    # guard against the case when an image has fewer than fg_rois_per_image foreground RoIs
    fg_rois_per_this_image = np.minimum(fg_rois_per_image, fg_indexes.size)

    if DEBUG:
        print 'fg total num:', len(fg_indexes)

    # Sample foreground regions without replacement
    if len(fg_indexes) > fg_rois_per_this_image:
        fg_indexes = npr.choice(fg_indexes, size=fg_rois_per_this_image, replace=False)

    # Select background RoIs as those within [BG_THRESH_LO, BG_THRESH_HI)
    bg_indexes = np.where((overlaps < config.TRAIN.BG_THRESH_HI) & (overlaps >= config.TRAIN.BG_THRESH_LO))[0]
    if DEBUG:
        print 'bg total num:', len(bg_indexes)
    # Compute number of background RoIs to take from this image (guarding against there being fewer than desired)
    bg_rois_per_this_image = rois_per_image - fg_rois_per_this_image
    bg_rois_per_this_image = np.minimum(bg_rois_per_this_image, bg_indexes.size)
    # Sample foreground regions without replacement
    if len(bg_indexes) > bg_rois_per_this_image:
        bg_indexes = npr.choice(bg_indexes, size=bg_rois_per_this_image, replace=False)
    if DEBUG:
        print 'fg num:', len(fg_indexes)
        print 'bg num:', len(bg_indexes)

    # bg rois statistics
    if DEBUG:
        bg_rois = rois[bg_indexes]
        bg_rois_area = np.sqrt((bg_rois[:, 3] - bg_rois[:, 1]) * (bg_rois[:, 2] - bg_rois[:, 0]))

        area_threshold = [[np.inf, 448],
                          [448,    224],
                          [224,    112],
                          [112,     0]]

        area_threshold = area_threshold[0:len(config.RCNN_FEAT_STRIDE)]
        area_threshold[-1][-1] = 0

        bg_rois_on_levels = dict()
        for i, s in enumerate(config.RCNN_FEAT_STRIDE):
            thd = area_threshold[i]
            index = np.logical_and(thd[1] <= bg_rois_area, bg_rois_area < thd[0])
            bg_rois_on_levels.update({'stride%s'%s:np.sum(index)})
        print bg_rois_on_levels

    # indexes selected
    keep_indexes = np.append(fg_indexes, bg_indexes)

    neg_idx = np.where(overlaps < config.TRAIN.FG_THRESH)[0]
    neg_rois = rois[neg_idx]

    # pad more to ensure a fixed minibatch size
    while keep_indexes.shape[0] < rois_per_image:
        gap = np.minimum(len(neg_rois), rois_per_image - keep_indexes.shape[0])
        gap_indexes = npr.choice(range(len(neg_rois)), size=gap, replace=False)
        keep_indexes = np.append(keep_indexes, neg_idx[gap_indexes])

    # select labels
    labels = labels[keep_indexes]
    # set labels of bg_rois to be 0
    labels[fg_rois_per_this_image:] = 0
    rois = rois[keep_indexes]

    # load or compute bbox_target
    if bbox_targets is not None:
        bbox_target_data = bbox_targets[keep_indexes, :]
    else:
        targets = bbox_transform(rois[:, 1:], gt_boxes[gt_assignment[keep_indexes], :4])
        if config.TRAIN.BBOX_NORMALIZATION_PRECOMPUTED:
            targets = ((targets - np.array(config.TRAIN.BBOX_MEANS))
                       / np.array(config.TRAIN.BBOX_STDS))
        bbox_target_data = np.hstack((labels[:, np.newaxis], targets))

    bbox_targets, bbox_weights = \
        expand_bbox_regression_targets(bbox_target_data, num_classes)

    # Assign to levels

    rois_area = np.sqrt((rois[:, 4] - rois[:, 2]) * (rois[:, 3] - rois[:, 1]))

    area_threshold = [[np.inf, 448],
                      [448,    224],
                      [224,    112],
                      [112,     0]]

    area_threshold = area_threshold[0:len(config.RCNN_FEAT_STRIDE)]
    area_threshold[-1][-1] = 0

    proposal_target = []
    for i, s in enumerate(config.RCNN_FEAT_STRIDE):
        thd = area_threshold[i]
        index         = np.logical_and(thd[1] <= rois_area, rois_area < thd[0])
        _rois         = rois[index]
        _labels       = labels[index]
        _bbox_targets = bbox_targets[index]
        _bbox_weights = bbox_weights[index]

        tmp_blob = config.TRAIN.IMAGE_BLOB.copy()
        pos_index = np.where(_labels != 0)[0]
        for tmp_index in pos_index:
            cv2.rectangle(tmp_blob, (_rois[tmp_index, 1], _rois[tmp_index, 2]), (_rois[tmp_index, 3], _rois[tmp_index, 4]), (0, 255, 0))
        neg_index = np.where(_labels == 0)[0]
        for tmp_index in neg_index:
            cv2.rectangle(tmp_blob, (_rois[tmp_index, 1], _rois[tmp_index, 2]), (_rois[tmp_index, 3], _rois[tmp_index, 4]), (0, 0, 255))
        for tmp_gt in gt_boxes:
            cv2.rectangle(tmp_blob, (tmp_gt[0], tmp_gt[1]), (tmp_gt[2], tmp_gt[3]), (255, 0, 0), 2)
        cv2.imshow('first%s' % s, tmp_blob)
        cv2.waitKey()

        if _rois.shape[0] != 0:
            _rois = np.pad(_rois, [(0, (rois_per_image - _rois.shape[0])), (0, 0)], mode='edge')
        else:
            _rois = np.array([[0, 0, 0, 1, 1]])
            _rois = np.pad(_rois, [(0, (rois_per_image - _rois.shape[0])), (0, 0)], mode='edge')
        _labels = np.pad(_labels, [(0, (rois_per_image - _labels.shape[0]))], mode='constant', constant_values=-1)
        _bbox_targets = np.pad(_bbox_targets, [(0, (rois_per_image - _bbox_targets.shape[0])), (0, 0)], mode='constant', constant_values=0)
        _bbox_weights = np.pad(_bbox_weights, [(0, (rois_per_image - _bbox_weights.shape[0])), (0, 0)], mode='constant', constant_values=0)

        proposal_target.append(_rois)
        proposal_target.append(_labels)
        proposal_target.append(_bbox_targets)
        proposal_target.append(_bbox_weights)

    return proposal_target
