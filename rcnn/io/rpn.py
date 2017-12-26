"""
RPN:
data =
    {'data': [num_images, c, h, w],
     'im_info': [num_images, 4] (optional)}
label =
    {'gt_boxes': [num_boxes, 5] (optional),
     'label': [batch_size, 1] <- [batch_size, num_anchors, feat_height, feat_width],
     'bbox_target': [batch_size, num_anchors, feat_height, feat_width],
     'bbox_weight': [batch_size, num_anchors, feat_height, feat_width]}
"""

import logging
import numpy as np
import numpy.random as npr

from ..logger import logger
from ..config import config
from .image import get_image, tensor_vstack
from ..processing.generate_anchor import generate_anchors, anchors_plane
from ..processing.bbox_transform import bbox_overlaps, bbox_transform

import cv2

def get_rpn_testbatch(roidb):
    """
    return a dict of testbatch
    :param roidb: ['image', 'flipped']
    :return: data, label, im_info
    """
    assert len(roidb) == 1, 'Single batch only'
    imgs, roidb = get_image(roidb)
    im_array = imgs[0]
    im_info = np.array([roidb[0]['im_info']], dtype=np.float32)

    data = {'data': im_array,
            'im_info': im_info}
    label = {}

    return data, label, im_info


def get_rpn_batch(roidb):
    """
    prototype for rpn batch: data, im_info, gt_boxes
    :param roidb: ['image', 'flipped'] + ['gt_boxes', 'boxes', 'gt_classes']
    :return: data, label
    """
    assert len(roidb) == 1, 'Single batch only'
    imgs, roidb = get_image(roidb)
    im_array = imgs[0]
    im_info = np.array([roidb[0]['im_info']], dtype=np.float32)

    # gt boxes: (x1, y1, x2, y2, cls)
    if roidb[0]['gt_classes'].size > 0:
        gt_inds = np.where(roidb[0]['gt_classes'] != 0)[0]
        gt_boxes = np.empty((roidb[0]['boxes'].shape[0], 5), dtype=np.float32)
        gt_boxes[:, 0:4] = roidb[0]['boxes'][gt_inds, :]
        gt_boxes[:, 4] = roidb[0]['gt_classes'][gt_inds]
        gt_keypoints = roidb[0]['keypoints']
    else:
        gt_boxes = np.empty((0, 5), dtype=np.float32)

    data = {'data': im_array,
            'im_info': im_info}
    label = {'gt_boxes': gt_boxes, 'gt_keypoints': gt_keypoints}

    return data, label


def assign_anchor(feat_shape, gt_boxes, im_info, feat_stride=16,
                  scales=(8, 16, 32), ratios=(0.5, 1, 2), allowed_border=0):
    """
    assign ground truth boxes to anchor positions
    :param feat_shape: infer output shape
    :param gt_boxes: assign ground truth
    :param im_info: filter out anchors overlapped with edges
    :param feat_stride: anchor position step
    :param scales: used to generate anchors, affects num_anchors (per location)
    :param ratios: aspect ratios of generated anchors
    :param allowed_border: filter out anchors with edge overlap > allowed_border
    :return: dict of label
    'label': of shape (batch_size, 1) <- (batch_size, num_anchors, feat_height, feat_width)
    'bbox_target': of shape (batch_size, num_anchors * 4, feat_height, feat_width)
    'bbox_inside_weight': *todo* mark the assigned anchors
    'bbox_outside_weight': used to normalize the bbox_loss, all weights sums to RPN_POSITIVE_WEIGHT
    """
    def _unmap(data, count, inds, fill=0):
        """" unmap a subset inds of data into original data of size count """
        if len(data.shape) == 1:
            ret = np.empty((count,), dtype=np.float32)
            ret.fill(fill)
            ret[inds] = data
        else:
            ret = np.empty((count,) + data.shape[1:], dtype=np.float32)
            ret.fill(fill)
            ret[inds, :] = data
        return ret

    im_info = im_info[0]
    scales = np.array(scales, dtype=np.float32)
    base_anchors = generate_anchors(base_size=feat_stride, ratios=list(ratios), scales=scales)
    num_anchors = base_anchors.shape[0]
    feat_height, feat_width = feat_shape[-2:]

    logger.debug('anchors: %s' % base_anchors)
    logger.debug('anchor shapes: %s' % np.hstack((base_anchors[:, 2::4] - base_anchors[:, 0::4],
                                                 base_anchors[:, 3::4] - base_anchors[:, 1::4])))
    logger.debug('im_info %s' % im_info)
    logger.debug('height %d width %d' % (feat_height, feat_width))
    logger.debug('gt_boxes shape %s' % np.array(gt_boxes.shape))
    logger.debug('gt_boxes %s' % gt_boxes)

    # 1. generate proposals from bbox deltas and shifted anchors
    shift_x = np.arange(0, feat_width) * feat_stride
    shift_y = np.arange(0, feat_height) * feat_stride
    shift_x, shift_y = np.meshgrid(shift_x, shift_y)
    shifts = np.vstack((shift_x.ravel(), shift_y.ravel(), shift_x.ravel(), shift_y.ravel())).transpose()
    # add A anchors (1, A, 4) to
    # cell K shifts (K, 1, 4) to get
    # shift anchors (K, A, 4)
    # reshape to (K*A, 4) shifted anchors
    A = num_anchors
    K = shifts.shape[0]
    all_anchors = base_anchors.reshape((1, A, 4)) + shifts.reshape((1, K, 4)).transpose((1, 0, 2))
    all_anchors = all_anchors.reshape((K * A, 4))
    total_anchors = int(K * A)

    # only keep anchors inside the image
    inds_inside = np.where((all_anchors[:, 0] >= -allowed_border) &
                           (all_anchors[:, 1] >= -allowed_border) &
                           (all_anchors[:, 2] < im_info[1] + allowed_border) &
                           (all_anchors[:, 3] < im_info[0] + allowed_border))[0]
    logger.debug('total_anchors %d' % total_anchors)
    logger.debug('inds_inside %d' % len(inds_inside))

    # keep only inside anchors
    anchors = all_anchors[inds_inside, :]
    logger.debug('anchors shape %s' % np.array(anchors.shape))

    # label: 1 is positive, 0 is negative, -1 is dont care
    labels = np.empty((len(inds_inside),), dtype=np.float32)
    labels.fill(-1)

    if gt_boxes.size > 0:
        # overlap between the anchors and the gt boxes
        # overlaps (ex, gt)
        overlaps = bbox_overlaps(anchors.astype(np.float), gt_boxes.astype(np.float))
        argmax_overlaps = overlaps.argmax(axis=1)
        max_overlaps = overlaps[np.arange(len(inds_inside)), argmax_overlaps]
        gt_argmax_overlaps = overlaps.argmax(axis=0)
        gt_max_overlaps = overlaps[gt_argmax_overlaps, np.arange(overlaps.shape[1])]
        gt_argmax_overlaps = np.where(overlaps == gt_max_overlaps)[0]

        if not config.TRAIN.RPN_CLOBBER_POSITIVES:
            # assign bg labels first so that positive labels can clobber them
            labels[max_overlaps < config.TRAIN.RPN_NEGATIVE_OVERLAP] = 0

        # fg label: for each gt, anchor with highest overlap
        labels[gt_argmax_overlaps] = 1

        # fg label: above threshold IoU
        labels[max_overlaps >= config.TRAIN.RPN_POSITIVE_OVERLAP] = 1

        if config.TRAIN.RPN_CLOBBER_POSITIVES:
            # assign bg labels last so that negative labels can clobber positives
            labels[max_overlaps < config.TRAIN.RPN_NEGATIVE_OVERLAP] = 0
    else:
        labels[:] = 0

    # subsample positive labels if we have too many
    num_fg = int(config.TRAIN.RPN_FG_FRACTION * config.TRAIN.RPN_BATCH_SIZE)
    fg_inds = np.where(labels == 1)[0]
    if len(fg_inds) > num_fg:
        disable_inds = npr.choice(fg_inds, size=(len(fg_inds) - num_fg), replace=False)
        if logger.level == logging.INFO:
            disable_inds = fg_inds[:(len(fg_inds) - num_fg)]
        labels[disable_inds] = -1

    # subsample negative labels if we have too many
    num_bg = config.TRAIN.RPN_BATCH_SIZE - np.sum(labels == 1)
    bg_inds = np.where(labels == 0)[0]
    if len(bg_inds) > num_bg:
        disable_inds = npr.choice(bg_inds, size=(len(bg_inds) - num_bg), replace=False)
        if logger.level == logging.INFO:
            disable_inds = bg_inds[:(len(bg_inds) - num_bg)]
        labels[disable_inds] = -1

    bbox_targets = np.zeros((len(inds_inside), 4), dtype=np.float32)
    if gt_boxes.size > 0:
        bbox_targets[:] = bbox_transform(anchors, gt_boxes[argmax_overlaps, :4])

    bbox_weights = np.zeros((len(inds_inside), 4), dtype=np.float32)
    bbox_weights[labels == 1, :] = np.array(config.TRAIN.RPN_BBOX_WEIGHTS)

    if logger.level == logging.DEBUG:
        _sums = bbox_targets[labels == 1, :].sum(axis=0)
        _squared_sums = (bbox_targets[labels == 1, :] ** 2).sum(axis=0)
        _counts = np.sum(labels == 1)
        means = _sums / (_counts + 1e-14)
        stds = np.sqrt(_squared_sums / _counts - means ** 2)
        logger.debug('means %s' % means)
        logger.debug('stdevs %s' % stds)

    # map up to original set of anchors
    labels = _unmap(labels, total_anchors, inds_inside, fill=-1)
    bbox_targets = _unmap(bbox_targets, total_anchors, inds_inside, fill=0)
    bbox_weights = _unmap(bbox_weights, total_anchors, inds_inside, fill=0)

    if logger.level == logging.DEBUG:
        if gt_boxes.size > 0:
            logger.debug('rpn: max max_overlaps %f' % np.max(max_overlaps))
        logger.debug('rpn: num_positives %f' % np.sum(labels == 1))
        logger.debug('rpn: num_negatives %f' % np.sum(labels == 0))
        _fg_sum = np.sum(labels == 1)
        _bg_sum = np.sum(labels == 0)
        _count = 1
        logger.debug('rpn: num_positive avg %f' % (_fg_sum / _count))
        logger.debug('rpn: num_negative avg %f' % (_bg_sum / _count))

    labels = labels.reshape((1, feat_height, feat_width, A)).transpose(0, 3, 1, 2)
    labels = labels.reshape((1, A * feat_height * feat_width))
    bbox_targets = bbox_targets.reshape((1, feat_height, feat_width, A * 4)).transpose(0, 3, 1, 2)
    bbox_weights = bbox_weights.reshape((1, feat_height, feat_width, A * 4)).transpose((0, 3, 1, 2))

    label = {'label': labels,
             'bbox_target': bbox_targets,
             'bbox_weight': bbox_weights}
    return label

def assign_anchor_fpn(feat_shape, gt_boxes, im_info, feat_strides=[64,32,16,8,4],
                            scales=(8, 16, 32), ratios=(0.5, 1, 2), allowed_border=0):
    """
    assign ground truth boxes to anchor positions
    :param feat_shape: infer output shape
    :param gt_boxes: assign ground truth
    :param im_info: filter out anchors overlapped with edges
    :param feat_stride: anchor position step
    :param scales: used to generate anchors, affects num_anchors (per location)
    :param ratios: aspect ratios of generated anchors
    :param allowed_border: filter out anchors with edge overlap > allowed_border
    :return: tuple
    labels: of shape (batch_size, 1) <- (batch_size, num_anchors, feat_height, feat_width)
    bbox_targets: of shape (batch_size, num_anchors * 4, feat_height, feat_width)
    bbox_weights: mark the assigned anchors
    """
    def _unmap(data, count, inds, fill=0):
        """" unmap a subset inds of data into original data of size count """
        if len(data.shape) == 1:
            ret = np.empty((count,), dtype=np.float32)
            ret.fill(fill)
            ret[inds] = data
        else:
            ret = np.empty((count,) + data.shape[1:], dtype=np.float32)
            ret.fill(fill)
            ret[inds, :] = data
        return ret

    DEBUG = False
    # clean up boxes
    nonneg = np.where(gt_boxes[:, 4] != -1)[0]
    gt_boxes = gt_boxes[nonneg]
    scales = np.array(scales, dtype=np.float32)

    anchors_list = []
    anchors_num_list = []
    inds_inside_list = []
    feat_infos = []
    A_list = []
    for i in range(len(feat_strides)):
        base_anchors = generate_anchors(base_size=feat_strides[i], ratios=list(ratios), scales=scales)
        num_anchors = base_anchors.shape[0]
        feat_height, feat_width = feat_shape[i][-2:]
        feat_stride = feat_strides[i]
        feat_infos.append([feat_height, feat_width])

        A = num_anchors
        A_list.append(A)
        K = feat_height * feat_width

        all_anchors = anchors_plane(feat_height, feat_width, feat_stride, base_anchors)
        all_anchors = all_anchors.reshape((K * A, 4))

        total_anchors = int(K * A)
        anchors_num_list.append(total_anchors)
        # only keep anchors inside the image
        inds_inside = np.where((all_anchors[:, 0] >= -allowed_border) &
                               (all_anchors[:, 1] >= -allowed_border) &
                               (all_anchors[:, 2] < im_info[1] + allowed_border) &
                               (all_anchors[:, 3] < im_info[0] + allowed_border))[0]
        if DEBUG:
            print 'total_anchors stride%s' % feat_stride, total_anchors
            print 'inds_inside stride%s' % feat_stride, len(inds_inside)

        # keep only inside anchors
        anchors = all_anchors[inds_inside, :]

        anchors_list.append(anchors)
        inds_inside_list.append(inds_inside)

    # Concat anchors from each level
    anchors = np.concatenate(anchors_list)
    for i in range(1, len(inds_inside_list)):
        inds_inside_list[i] = inds_inside_list[i] + sum(anchors_num_list[:i])
    inds_inside = np.concatenate(inds_inside_list)
    total_anchors = sum(anchors_num_list)

    # label: 1 is positive, 0 is negative, -1 is dont care
    labels = np.empty((len(inds_inside),), dtype=np.float32)
    labels.fill(-1)

    if gt_boxes.size > 0:
        # overlap between the anchors and the gt boxes
        # overlaps (ex, gt)
        overlaps = bbox_overlaps(anchors.astype(np.float), gt_boxes.astype(np.float))
        argmax_overlaps = overlaps.argmax(axis=1)
        max_overlaps = overlaps[np.arange(len(inds_inside)), argmax_overlaps]
        gt_argmax_overlaps = overlaps.argmax(axis=0)
        gt_max_overlaps = overlaps[gt_argmax_overlaps, np.arange(overlaps.shape[1])]
        gt_argmax_overlaps = np.where(overlaps == gt_max_overlaps)[0]

        if not config.TRAIN.RPN_CLOBBER_POSITIVES:
            # assign bg labels first so that positive labels can clobber them
            labels[max_overlaps < config.TRAIN.RPN_NEGATIVE_OVERLAP] = 0

        # fg label: for each gt, anchor with highest overlap
        labels[gt_argmax_overlaps] = 1

        # fg label: above threshold IoU
        labels[max_overlaps >= config.TRAIN.RPN_POSITIVE_OVERLAP] = 1

        if config.TRAIN.RPN_CLOBBER_POSITIVES:
            # assign bg labels last so that negative labels can clobber positives
            labels[max_overlaps < config.TRAIN.RPN_NEGATIVE_OVERLAP] = 0
    else:
        labels[:] = 0

    # subsample positive labels if we have too many
    num_fg = int(config.TRAIN.RPN_FG_FRACTION * config.TRAIN.RPN_BATCH_SIZE)
    fg_inds = np.where(labels == 1)[0]
    if len(fg_inds) > num_fg:
        disable_inds = npr.choice(fg_inds, size=(len(fg_inds) - num_fg), replace=False)
        labels[disable_inds] = -1

    # subsample negative labels if we have too many
    num_bg = config.TRAIN.RPN_BATCH_SIZE - np.sum(labels == 1)
    bg_inds = np.where(labels == 0)[0]
    if len(bg_inds) > num_bg:
        disable_inds = npr.choice(bg_inds, size=(len(bg_inds) - num_bg), replace=False)
        labels[disable_inds] = -1

    # tmp_blob = config.TRAIN.IMAGE_BLOB.copy()
    # pos_index = np.where(labels == 1)[0]
    # for tmp_index in pos_index:
    #     cv2.rectangle(tmp_blob, (anchors[tmp_index, 0], anchors[tmp_index, 1]), (anchors[tmp_index, 2], anchors[tmp_index, 3]),
    #                   (0, 255, 0))
    # neg_index = np.where(labels == 0)[0]
    # for tmp_index in neg_index:
    #     cv2.rectangle(tmp_blob, (anchors[tmp_index, 0], anchors[tmp_index, 1]), (anchors[tmp_index, 2], anchors[tmp_index, 3]),
    #                   (0, 0, 255))
    # for i in xrange(len(gt_boxes)):
    #     [x1, y1, x2, y2] = gt_boxes[i, :-1].astype(np.int32)
    #     cv2.rectangle(tmp_blob, (x1, y1), (x2, y2), (123, 13, 33), 2)
    # cv2.imshow('rpn', tmp_blob)
    # cv2.waitKey()

    bbox_targets = np.zeros((len(inds_inside), 4), dtype=np.float32)
    if gt_boxes.size > 0:
        bbox_targets[:] = bbox_transform(anchors, gt_boxes[argmax_overlaps, :4])

    bbox_weights = np.zeros((len(inds_inside), 4), dtype=np.float32)
    bbox_weights[labels == 1, :] = np.array(config.TRAIN.RPN_BBOX_WEIGHTS)

    if DEBUG:
        _sums = bbox_targets[labels == 1, :].sum(axis=0)
        _squared_sums = (bbox_targets[labels == 1, :] ** 2).sum(axis=0)
        _counts = np.sum(labels == 1)
        means = _sums / (_counts + 1e-14)
        stds = np.sqrt(_squared_sums / _counts - means ** 2)
        print 'means', means
        print 'stdevs', stds
        print '_sums', _sums
    # map up to original set of anchors
    labels = _unmap(labels, total_anchors, inds_inside, fill=-1)
    bbox_targets = _unmap(bbox_targets, total_anchors, inds_inside, fill=0)
    bbox_weights = _unmap(bbox_weights, total_anchors, inds_inside, fill=0)

    if DEBUG:
        if gt_boxes.size > 0:
            print 'rpn: max max_overlaps', np.max(max_overlaps)
        print 'rpn: num_positives', np.sum(labels == 1)
        print 'rpn: num_negatives', np.sum(labels == 0)
        _fg_sum = np.sum(labels == 1)
        _bg_sum = np.sum(labels == 0)
        _count = 1
        print 'rpn: num_positive avg', _fg_sum / _count
        print 'rpn: num_negative avg', _bg_sum / _count

    # resahpe
    label_list = list()
    bbox_target_list = list()
    bbox_weight_list = list()
    anchors_num_range = [0] + anchors_num_list
    for i in range(len(feat_strides)):
        feat_height, feat_width = feat_infos[i]
        A = A_list[i]
        label = labels[sum(anchors_num_range[:i+1]):sum(anchors_num_range[:i+1])+anchors_num_range[i+1]]
        bbox_target = bbox_targets[sum(anchors_num_range[:i+1]):sum(anchors_num_range[:i+1])+anchors_num_range[i+1]]
        bbox_weight = bbox_weights[sum(anchors_num_range[:i+1]):sum(anchors_num_range[:i+1])+anchors_num_range[i+1]]

        label = label.reshape((1, feat_height, feat_width, A)).transpose(0, 3, 1, 2)
        label = label.reshape((1, A * feat_height * feat_width))
        bbox_target = bbox_target.reshape((1, feat_height*feat_width, A * 4)).transpose(0, 2, 1)
        bbox_weight = bbox_weight.reshape((1, feat_height*feat_width, A * 4)).transpose((0, 2, 1))

        label_list.append(label)
        bbox_target_list.append(bbox_target)
        bbox_weight_list.append(bbox_weight)

    label_concat = np.concatenate(label_list, axis=1)
    bbox_target_concat = np.concatenate(bbox_target_list, axis=2)
    bbox_weight_concat = np.concatenate(bbox_weight_list, axis=2)

    label= {'label': label_concat,
            'bbox_target': bbox_target_concat,
            'bbox_weight': bbox_weight_concat}
    return label
