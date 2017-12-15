"""
Proposal Target Operator selects foreground and background roi and assigns label, bbox_transform to them.
"""

import logging
import mxnet as mx
import numpy as np
from distutils.util import strtobool

from rcnn.logger import logger
from rcnn.io.rcnn import sample_rois_fpn


class ProposalFpnTargetOperator(mx.operator.CustomOp):
    def __init__(self, num_classes, batch_images, batch_rois, fg_fraction):
        super(ProposalFpnTargetOperator, self).__init__()
        self._num_classes = num_classes
        self._batch_images = batch_images
        self._batch_rois = batch_rois
        self._fg_fraction = fg_fraction

        if logger.level == logging.DEBUG:
            self._count = 0
            self._fg_num = 0
            self._bg_num = 0

    def forward(self, is_train, req, in_data, out_data, aux):
        assert self._batch_rois % self._batch_images == 0, \
            'BATCHIMAGES {} must devide BATCH_ROIS {}'.format(self._batch_images, self._batch_rois)
        rois_per_image = self._batch_rois / self._batch_images
        fg_rois_per_image = np.round(self._fg_fraction * rois_per_image).astype(int)

        all_rois = in_data[0].asnumpy()
        gt_boxes = in_data[1].asnumpy()

        # Include ground-truth boxes in the set of candidate rois
        zeros = np.zeros((gt_boxes.shape[0], 1), dtype=gt_boxes.dtype)
        all_rois = np.vstack((all_rois, np.hstack((zeros, gt_boxes[:, :-1]))))
        # Sanity check: single batch only
        assert np.all(all_rois[:, 0] == 0), 'Only single item batches are supported'

        proposal_target = sample_rois_fpn(all_rois, fg_rois_per_image, rois_per_image, self._num_classes, gt_boxes=gt_boxes)

        for ind, val in enumerate(proposal_target):
            self.assign(out_data[ind], req[ind], val)

    def backward(self, req, out_grad, in_data, out_data, in_grad, aux):
        self.assign(in_grad[0], req[0], 0)
        self.assign(in_grad[1], req[1], 0)

@mx.operator.register('proposal_fpn_target')
class ProposalFpnTargetProp(mx.operator.CustomOpProp):
    def __init__(self, num_classes, batch_images, batch_rois, fg_fraction='0.25'):
        super(ProposalFpnTargetProp, self).__init__(need_top_grad=False)
        self._num_classes = int(num_classes)
        self._batch_images = int(batch_images)
        self._batch_rois = int(batch_rois)
        self._fg_fraction = float(fg_fraction)

    def list_arguments(self):
        return ['rois', 'gt_boxes']

    def list_outputs(self):
        RCNN_FEAT_STRIDE = [32, 16, 8, 4]
        output_list = []
        for stride in RCNN_FEAT_STRIDE:
            output_list.append('rois_stride%s_output' % stride)
            output_list.append('stride%s_label' % stride)
            output_list.append('bbox_target_stride%s' % stride)
            output_list.append('bbox_weight_stride%s' % stride)
        return output_list

    def infer_shape(self, in_shape):
        rpn_rois_shape = in_shape[0]
        gt_boxes_shape = in_shape[1]

        RCNN_FEAT_STRIDE = [32, 16, 8, 4]
        output_shapes = []
        for stride in RCNN_FEAT_STRIDE:
            output_rois_shape = (self._batch_rois, 5)
            label_shape = (self._batch_rois, )
            bbox_target_shape = (self._batch_rois, self._num_classes * 4)
            bbox_weight_shape = (self._batch_rois, self._num_classes * 4)
            output_shapes.append(output_rois_shape)
            output_shapes.append(label_shape)
            output_shapes.append(bbox_target_shape)
            output_shapes.append(bbox_weight_shape)

        return [rpn_rois_shape, gt_boxes_shape], output_shapes

    def create_operator(self, ctx, shapes, dtypes):
        return ProposalFpnTargetOperator(self._num_classes, self._batch_images, self._batch_rois, self._fg_fraction)

    def declare_backward_dependency(self, out_grad, in_data, out_data):
        return []
