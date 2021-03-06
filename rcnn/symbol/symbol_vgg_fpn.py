import mxnet as mx
import proposal_fpn
import proposal_fpn_target
import fpn_roi_pooling
from rcnn.config import config


def get_vgg_conv(data):
    """
    shared convolutional layers
    :param data: Symbol
    :return: Symbol
    """
    # group 1
    conv1_1 = mx.symbol.Convolution(
        data=data, kernel=(3, 3), pad=(1, 1), num_filter=64, workspace=2048, name="conv1_1")
    relu1_1 = mx.symbol.Activation(data=conv1_1, act_type="relu", name="relu1_1")
    conv1_2 = mx.symbol.Convolution(
        data=relu1_1, kernel=(3, 3), pad=(1, 1), num_filter=64, workspace=2048, name="conv1_2")
    relu1_2 = mx.symbol.Activation(data=conv1_2, act_type="relu", name="relu1_2")
    pool1 = mx.symbol.Pooling(
        data=relu1_2, pool_type="max", kernel=(2, 2), stride=(2, 2), name="pool1")
    # group 2
    conv2_1 = mx.symbol.Convolution(
        data=pool1, kernel=(3, 3), pad=(1, 1), num_filter=128, workspace=2048, name="conv2_1")
    relu2_1 = mx.symbol.Activation(data=conv2_1, act_type="relu", name="relu2_1")
    conv2_2 = mx.symbol.Convolution(
        data=relu2_1, kernel=(3, 3), pad=(1, 1), num_filter=128, workspace=2048, name="conv2_2")
    relu2_2 = mx.symbol.Activation(data=conv2_2, act_type="relu", name="relu2_2")
    pool2 = mx.symbol.Pooling(
        data=relu2_2, pool_type="max", kernel=(2, 2), stride=(2, 2), name="pool2")
    # group 3
    conv3_1 = mx.symbol.Convolution(
        data=pool2, kernel=(3, 3), pad=(1, 1), num_filter=256, workspace=2048, name="conv3_1")
    relu3_1 = mx.symbol.Activation(data=conv3_1, act_type="relu", name="relu3_1")
    conv3_2 = mx.symbol.Convolution(
        data=relu3_1, kernel=(3, 3), pad=(1, 1), num_filter=256, workspace=2048, name="conv3_2")
    relu3_2 = mx.symbol.Activation(data=conv3_2, act_type="relu", name="relu3_2")
    conv3_3 = mx.symbol.Convolution(
        data=relu3_2, kernel=(3, 3), pad=(1, 1), num_filter=256, workspace=2048, name="conv3_3")
    relu3_3 = mx.symbol.Activation(data=conv3_3, act_type="relu", name="relu3_3")
    pool3 = mx.symbol.Pooling(
        data=relu3_3, pool_type="max", kernel=(2, 2), stride=(2, 2), name="pool3")
    # group 4
    conv4_1 = mx.symbol.Convolution(
        data=pool3, kernel=(3, 3), pad=(1, 1), num_filter=512, workspace=2048, name="conv4_1")
    relu4_1 = mx.symbol.Activation(data=conv4_1, act_type="relu", name="relu4_1")
    conv4_2 = mx.symbol.Convolution(
        data=relu4_1, kernel=(3, 3), pad=(1, 1), num_filter=512, workspace=2048, name="conv4_2")
    relu4_2 = mx.symbol.Activation(data=conv4_2, act_type="relu", name="relu4_2")
    conv4_3 = mx.symbol.Convolution(
        data=relu4_2, kernel=(3, 3), pad=(1, 1), num_filter=512, workspace=2048, name="conv4_3")
    relu4_3 = mx.symbol.Activation(data=conv4_3, act_type="relu", name="relu4_3")
    pool4 = mx.symbol.Pooling(
        data=relu4_3, pool_type="max", kernel=(2, 2), stride=(2, 2), name="pool4")
    # group 5
    conv5_1 = mx.symbol.Convolution(
        data=pool4, kernel=(3, 3), pad=(1, 1), num_filter=512, workspace=2048, name="conv5_1")
    relu5_1 = mx.symbol.Activation(data=conv5_1, act_type="relu", name="relu5_1")
    conv5_2 = mx.symbol.Convolution(
        data=relu5_1, kernel=(3, 3), pad=(1, 1), num_filter=512, workspace=2048, name="conv5_2")
    relu5_2 = mx.symbol.Activation(data=conv5_2, act_type="relu", name="relu5_2")
    conv5_3 = mx.symbol.Convolution(
        data=relu5_2, kernel=(3, 3), pad=(1, 1), num_filter=512, workspace=2048, name="conv5_3")
    relu5_3 = mx.symbol.Activation(data=conv5_3, act_type="relu", name="relu5_3")
    pool5 = mx.symbol.Pooling(
        data=relu5_3, pool_type="max", kernel=(2, 2), stride=(2, 2), name="pool5")

    pool_feat = [pool5, pool4, pool3, pool2]
    return pool_feat

def get_vgg_conv_down(pool_feat, num_filter=256):
    # C5 to P5, 1x1 dimension reduction to num_filter
    P5 = mx.symbol.Convolution(data=pool_feat[0], kernel=(1, 1), num_filter=num_filter, name="P5_lateral")

    # P5 2x upsampling + C4 = P4
    # P5_up = mx.sym.Deconvolution(data=P5, kernel=(4, 4), stride=(2, 2), pad=(1, 1), num_filter=num_filter, adj=(1, 1),
    #                              name='P5_upsampling')
    P5_up = mx.symbol.UpSampling(P5, scale=2, sample_type='nearest', workspace=512, name='P5_upsampling', num_args=1)
    P4_la = mx.symbol.Convolution(data=pool_feat[1], kernel=(1, 1), num_filter=num_filter, name="P4_lateral")
    P5_clip = mx.symbol.Crop(*[P5_up, P4_la], name="P4_clip")
    P4 = mx.sym.ElementWiseSum(*[P5_clip, P4_la], name="P4_sum")
    P4 = mx.symbol.Convolution(data=P4, kernel=(3, 3), pad=(1, 1), num_filter=num_filter, name="P4_aggregate")

    # P4 2x upsampling + C3 = P3
    # P4_up = mx.sym.Deconvolution(data=P4, kernel=(4, 4), stride=(2, 2), pad=(1, 1), num_filter=num_filter, adj=(1, 1),
    #                              name='P4_upsampling')
    P4_up = mx.symbol.UpSampling(P4, scale=2, sample_type='nearest', workspace=512, name='P4_upsampling', num_args=1)
    P3_la = mx.symbol.Convolution(data=pool_feat[2], kernel=(1, 1), num_filter=num_filter, name="P3_lateral")
    P4_clip = mx.symbol.Crop(*[P4_up, P3_la], name="P3_clip")
    P3 = mx.sym.ElementWiseSum(*[P4_clip, P3_la], name="P3_sum")
    P3 = mx.symbol.Convolution(data=P3, kernel=(3, 3), pad=(1, 1), num_filter=num_filter, name="P3_aggregate")

    # P3 2x upsampling + C2 = P2
    # P3_up = mx.sym.Deconvolution(data=P3, kernel=(4, 4), stride=(2, 2), pad=(1, 1), num_filter=num_filter, adj=(1, 1),
    #                              name='P3_upsampling')
    P3_up = mx.symbol.UpSampling(P3, scale=2, sample_type='nearest', workspace=512, name='P3_upsampling', num_args=1)
    P2_la = mx.symbol.Convolution(data=pool_feat[3], kernel=(1, 1), num_filter=num_filter, name="P2_lateral")
    P3_clip = mx.symbol.Crop(*[P3_up, P2_la], name="P2_clip")
    P2 = mx.sym.ElementWiseSum(*[P3_clip, P2_la], name="P2_sum")
    P2 = mx.symbol.Convolution(data=P2, kernel=(3, 3), pad=(1, 1), num_filter=num_filter, name="P2_aggregate")

    conv_fpn_feat = dict()
    conv_fpn_feat.update({"stride32": P5, "stride16": P4, "stride8": P3, "stride4": P2})

    return conv_fpn_feat, [P5, P4, P3, P2]

def get_vgg_fpn_rpn(num_anchors=config.NUM_ANCHORS):
    data = mx.symbol.Variable(name="data")
    rpn_label = mx.symbol.Variable(name='label')
    rpn_bbox_target = mx.symbol.Variable(name='bbox_target')
    rpn_bbox_weight = mx.symbol.Variable(name='bbox_weight')

    # begin share weights
    rpn_conv_weight = mx.symbol.Variable('rpn_conv_weight')
    rpn_conv_bias = mx.symbol.Variable('rpn_conv_bias')
    rpn_conv_cls_weight = mx.symbol.Variable('rpn_conv_cls_weight')
    rpn_conv_cls_bias = mx.symbol.Variable('rpn_conv_cls_bias')
    rpn_conv_bbox_weight = mx.symbol.Variable('rpn_conv_bbox_weight')
    rpn_conv_bbox_bias = mx.symbol.Variable('rpn_conv_bbox_bias')
    # end share weights

    pool_feat = get_vgg_conv(data)
    conv_fpn_feat, _ = get_vgg_conv_down(pool_feat)

    rpn_cls_score_list = []
    rpn_bbox_pred_list = []
    for stride in config.RPN_FEAT_STRIDE:
        rpn_conv = mx.symbol.Convolution(data=conv_fpn_feat['stride%s'%stride],
                                         kernel=(3, 3), pad=(1, 1),
                                         num_filter=512,
                                         weight=rpn_conv_weight,
                                         bias=rpn_conv_bias,
                                         name='rpn_conv_3x3_stride%s' % stride)
        rpn_relu = mx.symbol.Activation(data=rpn_conv, act_type="relu", name="rpn_relu_stride%s" % stride)
        rpn_cls_score = mx.symbol.Convolution(data=rpn_relu,
                                              kernel=(1, 1), pad=(0, 0),
                                              num_filter=2 * num_anchors,
                                              weight=rpn_conv_cls_weight,
                                              bias=rpn_conv_cls_bias,
                                              name="rpn_cls_score_stride%s" % stride)
        rpn_bbox_pred = mx.symbol.Convolution(data=rpn_relu,
                                              kernel=(1, 1), pad=(0, 0),
                                              num_filter=4 * num_anchors,
                                              weight=rpn_conv_bbox_weight,
                                              bias=rpn_conv_bbox_bias,
                                              name='rpn_bbox_pred_stride%s' % stride)

        # prepare rpn data
        rpn_cls_score_reshape_for_loss = mx.symbol.Reshape(data=rpn_cls_score,
                                                  shape=(0, 2, -1),
                                                  name="rpn_cls_score_reshape_for_loss_stride%s" % stride)
        rpn_bbox_pred_reshape_for_loss = mx.symbol.Reshape(data=rpn_bbox_pred,
                                                  shape=(0, 0, -1),
                                                  name="rpn_bbox_pred_reshape_for_loss_stride%s" % stride)

        rpn_bbox_pred_list.append(rpn_bbox_pred_reshape_for_loss)
        rpn_cls_score_list.append(rpn_cls_score_reshape_for_loss)

    # concat output of each level
    rpn_bbox_pred_concat = mx.symbol.concat(*rpn_bbox_pred_list, dim=2, name='rpn_bbox_prex_concat')
    rpn_cls_score_concat = mx.symbol.concat(*rpn_cls_score_list, dim=2, name='rpn_cls_score_concat')

    # loss
    rpn_cls_prob = mx.symbol.SoftmaxOutput(data=rpn_cls_score_concat,
                                           label=rpn_label,
                                           multi_output=True,
                                           normalization='valid', use_ignore=True, ignore_label=-1,
                                           name='rpn_cls_prob')

    rpn_bbox_loss_ = rpn_bbox_weight * mx.symbol.smooth_l1(name='rpn_bbox_loss_', scalar=3.0,
                                                           data=(rpn_bbox_pred_concat - rpn_bbox_target))

    rpn_bbox_loss = mx.sym.MakeLoss(name='rpn_bbox_loss', data=rpn_bbox_loss_,
                                    grad_scale=1.0 / config.TRAIN.RPN_BATCH_SIZE)

    # group output
    group = mx.symbol.Group([rpn_cls_prob, rpn_bbox_loss])
    return group

def get_vgg_fpn_test(num_classes=config.NUM_CLASSES, num_anchors=config.NUM_ANCHORS):
    data = mx.symbol.Variable(name="data")
    im_info = mx.symbol.Variable(name="im_info")

    pool_feat = get_vgg_conv(data)
    conv_fpn_feat, _ = get_vgg_conv_down(pool_feat)

    # begin share weights
    rpn_conv_weight = mx.symbol.Variable('rpn_conv_weight')
    rpn_conv_bias = mx.symbol.Variable('rpn_conv_bias')
    rpn_conv_cls_weight = mx.symbol.Variable('rpn_conv_cls_weight')
    rpn_conv_cls_bias = mx.symbol.Variable('rpn_conv_cls_bias')
    rpn_conv_bbox_weight = mx.symbol.Variable('rpn_conv_bbox_weight')
    rpn_conv_bbox_bias = mx.symbol.Variable('rpn_conv_bbox_bias')

    rcnn_fc6_weight = mx.symbol.Variable('rcnn_fc6_weight')
    rcnn_fc6_bias = mx.symbol.Variable('rcnn_fc6_bias')
    rcnn_fc7_weight = mx.symbol.Variable('rcnn_fc7_weight')
    rcnn_fc7_bias = mx.symbol.Variable('rcnn_fc7_bias')
    rcnn_fc_cls_weight = mx.symbol.Variable('rcnn_fc_cls_weight')
    rcnn_fc_cls_bias = mx.symbol.Variable('rcnn_fc_cls_bias')
    rcnn_fc_bbox_weight = mx.symbol.Variable('bbox_pred_weight')
    rcnn_fc_bbox_bias = mx.symbol.Variable('bbox_pred_bias')
    # end share weights

    rpn_cls_prob_dict = dict()
    rpn_bbox_pred_dict = dict()
    for stride in config.RPN_FEAT_STRIDE:
        rpn_conv = mx.symbol.Convolution(data=conv_fpn_feat['stride%s' % stride],
                                         kernel=(3, 3), pad=(1, 1),
                                         num_filter=512,
                                         weight=rpn_conv_weight,
                                         bias=rpn_conv_bias,
                                         name='rpn_conv_3x3_stride%s' % stride)
        rpn_relu = mx.symbol.Activation(data=rpn_conv, act_type="relu", name="rpn_relu_stride%s" % stride)
        rpn_cls_score = mx.symbol.Convolution(data=rpn_relu,
                                              kernel=(1, 1), pad=(0, 0),
                                              num_filter=2 * num_anchors,
                                              weight=rpn_conv_cls_weight,
                                              bias=rpn_conv_cls_bias,
                                              name="rpn_cls_score_stride%s" % stride)
        rpn_bbox_pred = mx.symbol.Convolution(data=rpn_relu,
                                              kernel=(1, 1), pad=(0, 0),
                                              num_filter=4 * num_anchors,
                                              weight=rpn_conv_bbox_weight,
                                              bias=rpn_conv_bbox_bias,
                                              name='rpn_bbox_pred_stride%s' % stride)

        rpn_cls_score_reshape = mx.symbol.Reshape(data=rpn_cls_score,
                                                  shape=(0, 2, -1, 0),
                                                  name="rpn_cls_score_reshape_stride%s" % stride)
        rpn_cls_prob_softmax_activation = mx.symbol.SoftmaxActivation(data=rpn_cls_score_reshape,
                                                                      mode="channel",
                                                                      name="rpn_cls_prob_softmax_activation_stride%s" % stride)
        rpn_cls_prob_softmax_activation_reshape = mx.symbol.Reshape(data=rpn_cls_prob_softmax_activation,
                                                                    shape=(0, 2 * num_anchors, -1, 0),
                                                                    name='rpn_cls_prob_softmax_activation_reshape_stride%s' % stride)

        rpn_cls_prob_dict.update({'rpn_cls_prob_stride%s' % stride: rpn_cls_prob_softmax_activation_reshape})
        rpn_bbox_pred_dict.update({'rpn_bbox_pred_stride%s' % stride: rpn_bbox_pred})

    args_dict = dict(rpn_cls_prob_dict.items() + rpn_bbox_pred_dict.items())
    aux_dict = {'im_info': im_info, 'name': 'rois',
                'op_type': 'proposal_fpn', 'output_score': False,
                'feat_stride': config.RPN_FEAT_STRIDE, 'scales': tuple(config.ANCHOR_SCALES),
                'rpn_pre_nms_top_n': config.TEST.RPN_PRE_NMS_TOP_N,
                'rpn_post_nms_top_n': config.TEST.RPN_POST_NMS_TOP_N,
                'rpn_min_size': config.RPN_FEAT_STRIDE,
                'threshold': config.TEST.RPN_NMS_THRESH}

    # Proposal
    rois = mx.symbol.Custom(**dict(args_dict.items() + aux_dict.items()))

    # FPN roi pooling
    args_dict = {}
    for s in config.RCNN_FEAT_STRIDE:
        args_dict.update({'feat_stride%s' % s: conv_fpn_feat['stride%s' % s]})
    args_dict.update({'rois': rois, 'name': 'fpn_roi_pool',
                      'op_type': 'fpn_roi_pool',
                      'rcnn_strides': config.RCNN_FEAT_STRIDE,
                      'pool_h': 7, 'pool_w': 7})
    roi_pool_fpn = mx.symbol.Custom(**args_dict)

    # group 6
    flatten = mx.symbol.Flatten(data=roi_pool_fpn, name="flatten_stride%s" % stride)
    fc6 = mx.symbol.FullyConnected(data=flatten, num_hidden=4096, name="fc6_stride%s" % stride, weight=rcnn_fc6_weight,
                                   bias=rcnn_fc6_bias)
    relu6 = mx.symbol.Activation(data=fc6, act_type="relu", name="relu6_stride%s" % stride)
    drop6 = mx.symbol.Dropout(data=relu6, p=0.5, name="drop6_stride%s" % stride)
    # group 7
    fc7 = mx.symbol.FullyConnected(data=drop6, num_hidden=4096, name="fc7_stride%s" % stride, weight=rcnn_fc7_weight,
                                   bias=rcnn_fc7_bias)
    relu7 = mx.symbol.Activation(data=fc7, act_type="relu", name="relu7_stride%s" % stride)
    drop7 = mx.symbol.Dropout(data=relu7, p=0.5, name="drop7_stride%s" % stride)
    # classification
    cls_score = mx.symbol.FullyConnected(name='cls_score_stride%s' % stride, data=drop7, num_hidden=num_classes,
                                         weight=rcnn_fc_cls_weight, bias=rcnn_fc_cls_bias)
    cls_prob = mx.symbol.SoftmaxActivation(name='cls_prob', data=cls_score)
    # bounding box regression
    bbox_pred = mx.symbol.FullyConnected(name='bbox_pred_stride%s' % stride, data=drop7, num_hidden=num_classes * 4,
                                         weight=rcnn_fc_bbox_weight, bias=rcnn_fc_bbox_bias)

    # reshape output
    cls_prob = mx.symbol.Reshape(data=cls_prob, shape=(config.TEST.BATCH_IMAGES, -1, num_classes),
                                      name='cls_prob_reshape')
    bbox_pred = mx.symbol.Reshape(data=bbox_pred, shape=(config.TEST.BATCH_IMAGES, -1, 4 * num_classes),
                                       name='bbox_pred_reshape')

    # group output
    group = mx.symbol.Group([rois, cls_prob, bbox_pred])
    return group

def get_vgg_fpn_train(num_classes=config.NUM_CLASSES, num_anchors=config.NUM_ANCHORS):
    data = mx.symbol.Variable(name="data")
    im_info = mx.symbol.Variable(name="im_info")
    gt_boxes = mx.symbol.Variable(name="gt_boxes")
    rpn_label = mx.symbol.Variable(name='label')
    rpn_bbox_target = mx.symbol.Variable(name='bbox_target')
    rpn_bbox_weight = mx.symbol.Variable(name='bbox_weight')

    # begin share weights
    rpn_conv_weight = mx.symbol.Variable('rpn_conv_weight')
    rpn_conv_bias = mx.symbol.Variable('rpn_conv_bias')
    rpn_conv_cls_weight = mx.symbol.Variable('rpn_conv_cls_weight')
    rpn_conv_cls_bias = mx.symbol.Variable('rpn_conv_cls_bias')
    rpn_conv_bbox_weight = mx.symbol.Variable('rpn_conv_bbox_weight')
    rpn_conv_bbox_bias = mx.symbol.Variable('rpn_conv_bbox_bias')

    rcnn_fc6_weight = mx.symbol.Variable('rcnn_fc6_weight')
    rcnn_fc6_bias = mx.symbol.Variable('rcnn_fc6_bias')
    rcnn_fc7_weight = mx.symbol.Variable('rcnn_fc7_weight')
    rcnn_fc7_bias = mx.symbol.Variable('rcnn_fc7_bias')
    rcnn_fc_cls_weight = mx.symbol.Variable('rcnn_fc_cls_weight')
    rcnn_fc_cls_bias = mx.symbol.Variable('rcnn_fc_cls_bias')
    rcnn_fc_bbox_weight = mx.symbol.Variable('bbox_pred_weight')
    rcnn_fc_bbox_bias = mx.symbol.Variable('bbox_pred_bias')
    # end share weights

    pool_feat = get_vgg_conv(data)
    conv_fpn_feat, _ = get_vgg_conv_down(pool_feat)

    rpn_cls_score_list = []
    rpn_bbox_pred_list = []

    rpn_cls_prob_dict = dict()
    rpn_bbox_pred_dict = dict()
    for stride in config.RPN_FEAT_STRIDE:
        rpn_conv = mx.symbol.Convolution(data=conv_fpn_feat['stride%s'%stride],
                                         kernel=(3, 3), pad=(1, 1),
                                         num_filter=512,
                                         weight=rpn_conv_weight,
                                         bias=rpn_conv_bias,
                                         name='rpn_conv_3x3_stride%s' % stride)
        rpn_relu = mx.symbol.Activation(data=rpn_conv, act_type="relu", name="rpn_relu_stride%s" % stride)
        rpn_cls_score = mx.symbol.Convolution(data=rpn_relu,
                                              kernel=(1, 1), pad=(0, 0),
                                              num_filter=2 * num_anchors,
                                              weight=rpn_conv_cls_weight,
                                              bias=rpn_conv_cls_bias,
                                              name="rpn_cls_score_stride%s" % stride)
        rpn_bbox_pred = mx.symbol.Convolution(data=rpn_relu,
                                              kernel=(1, 1), pad=(0, 0),
                                              num_filter=4 * num_anchors,
                                              weight=rpn_conv_bbox_weight,
                                              bias=rpn_conv_bbox_bias,
                                              name='rpn_bbox_pred_stride%s' % stride)

        # fix overflow
        rpn_bbox_pred = mx.symbol.clip(data=rpn_bbox_pred, a_min=-1, a_max=1)

        # prepare rpn data
        rpn_cls_score_reshape_for_loss = mx.symbol.Reshape(data=rpn_cls_score,
                                                  shape=(0, 2, -1),
                                                  name="rpn_cls_score_reshape_for_loss_stride%s" % stride)
        rpn_bbox_pred_reshape_for_loss = mx.symbol.Reshape(data=rpn_bbox_pred,
                                                  shape=(0, 0, -1),
                                                  name="rpn_bbox_pred_reshape_for_loss_stride%s" % stride)

        rpn_bbox_pred_list.append(rpn_bbox_pred_reshape_for_loss)
        rpn_cls_score_list.append(rpn_cls_score_reshape_for_loss)

        rpn_cls_score_reshape = mx.symbol.Reshape(data=rpn_cls_score,
                                                           shape=(0, 2, -1, 0),
                                                           name="rpn_cls_score_reshape_stride%s" % stride)
        rpn_cls_prob_softmax_activation = mx.symbol.SoftmaxActivation(data=rpn_cls_score_reshape,
                                                   mode="channel",
                                                   name="rpn_cls_prob_softmax_activation_stride%s" % stride)
        rpn_cls_prob_softmax_activation_reshape = mx.symbol.Reshape(data=rpn_cls_prob_softmax_activation,
                                                 shape=(0, 2 * num_anchors, -1, 0),
                                                 name='rpn_cls_prob_softmax_activation_reshape_stride%s' % stride)

        rpn_cls_prob_dict.update({'rpn_cls_prob_stride%s' % stride: rpn_cls_prob_softmax_activation_reshape})
        rpn_bbox_pred_dict.update({'rpn_bbox_pred_stride%s' % stride: rpn_bbox_pred})

    # concat output of each level
    rpn_bbox_pred_concat = mx.symbol.concat(*rpn_bbox_pred_list, dim=2, name='rpn_bbox_prex_concat')
    rpn_cls_score_concat = mx.symbol.concat(*rpn_cls_score_list, dim=2, name='rpn_cls_score_concat')

    # loss
    rpn_cls_prob = mx.symbol.SoftmaxOutput(data=rpn_cls_score_concat,
                                           label=rpn_label,
                                           multi_output=True,
                                           normalization='valid', use_ignore=True, ignore_label=-1,
                                           name='rpn_cls_prob')

    rpn_bbox_loss_ = rpn_bbox_weight * mx.symbol.smooth_l1(name='rpn_bbox_loss_', scalar=3.0,
                                                           data=(rpn_bbox_pred_concat - rpn_bbox_target))

    rpn_bbox_loss = mx.sym.MakeLoss(name='rpn_bbox_loss', data=rpn_bbox_loss_,
                                    grad_scale=1.0 / config.TRAIN.RPN_BATCH_SIZE)

    # ROI proposal
    args_dict = dict(rpn_cls_prob_dict.items() + rpn_bbox_pred_dict.items())
    aux_dict = {'im_info': im_info, 'name': 'rois',
                'op_type': 'proposal_fpn', 'output_score': False,
                'feat_stride': config.RPN_FEAT_STRIDE, 'scales': tuple(config.ANCHOR_SCALES),
                'rpn_pre_nms_top_n': config.TRAIN.RPN_PRE_NMS_TOP_N,
                'rpn_post_nms_top_n': config.TRAIN.RPN_POST_NMS_TOP_N,
                'rpn_min_size': config.RPN_FEAT_STRIDE,
                'threshold': config.TRAIN.RPN_NMS_THRESH}
    # Proposal
    rois = mx.symbol.Custom(**dict(args_dict.items() + aux_dict.items()))

    # ROI proposal target
    gt_boxes_reshape = mx.symbol.Reshape(data=gt_boxes, shape=(-1, 5), name='gt_boxes_reshape')
    proposal_target = mx.symbol.Custom(rois=rois, gt_boxes=gt_boxes_reshape, op_type='proposal_fpn_target',
                             num_classes=num_classes, batch_images=config.TRAIN.BATCH_IMAGES,
                             batch_rois=config.TRAIN.BATCH_ROIS, fg_fraction=config.TRAIN.FG_FRACTION)

    rois_dict = dict()
    label_dict = dict()
    bbox_target_dict = dict()
    bbox_weight_dict = dict()
    index = 0
    for s in config.RCNN_FEAT_STRIDE:
        rois_dict['rois_stride%s' % s] = proposal_target[index * 4]
        label_dict['label_stride%s' % s] = proposal_target[index * 4 + 1]
        bbox_target_dict['bbox_target_stride%s' % s] = proposal_target[index * 4 + 2]
        bbox_weight_dict['bbox_weight_stride%s' % s] = proposal_target[index * 4 + 3]
        index += 1

    label_list = []
    bbox_target_list = []
    bbox_weight_list = []
    for s in config.RCNN_FEAT_STRIDE:
        label_list.append(label_dict['label_stride%s' % s])
        bbox_target_list.append(bbox_target_dict['bbox_target_stride%s' % s])
        bbox_weight_list.append(bbox_weight_dict['bbox_weight_stride%s' % s])

    label = mx.symbol.concat(*label_list, dim=0, name='rcnn_label_concat')
    bbox_target = mx.symbol.concat(*bbox_target_list, dim=0, name='bbox_target_concat')
    bbox_weight = mx.symbol.concat(*bbox_weight_list, dim=0, name='bbox_weight_concat')

    # Fast R-CNN
    rcnn_cls_score_list = []
    rcnn_bbox_pred_list = []
    for stride in config.RCNN_FEAT_STRIDE:
        roi_pool = mx.symbol.ROIPooling(
            name='roi_pool_stride%s' % stride, data=conv_fpn_feat['stride%s' % stride], rois=rois_dict['rois_stride%s' % stride],
            pooled_size=(7, 7),
            spatial_scale=1.0 / stride)

        # group 6
        flatten = mx.symbol.Flatten(data=roi_pool, name="flatten_stride%s" % stride)
        fc6 = mx.symbol.FullyConnected(data=flatten, num_hidden=4096, name="fc6_stride%s" % stride, weight=rcnn_fc6_weight, bias=rcnn_fc6_bias)
        relu6 = mx.symbol.Activation(data=fc6, act_type="relu", name="relu6_stride%s" % stride)
        drop6 = mx.symbol.Dropout(data=relu6, p=0.5, name="drop6_stride%s" % stride)
        # group 7
        fc7 = mx.symbol.FullyConnected(data=drop6, num_hidden=4096, name="fc7_stride%s" % stride, weight=rcnn_fc7_weight, bias=rcnn_fc7_bias)
        relu7 = mx.symbol.Activation(data=fc7, act_type="relu", name="relu7_stride%s" % stride)
        drop7 = mx.symbol.Dropout(data=relu7, p=0.5, name="drop7_stride%s" % stride)
        # classification
        cls_score = mx.symbol.FullyConnected(name='cls_score_stride%s' % stride, data=drop7, num_hidden=num_classes, weight=rcnn_fc_cls_weight, bias=rcnn_fc_cls_bias)
        # bounding box regression
        bbox_pred = mx.symbol.FullyConnected(name='bbox_pred_stride%s' % stride, data=drop7, num_hidden=num_classes * 4, weight=rcnn_fc_bbox_weight, bias=rcnn_fc_bbox_bias)

        # fix overflow
        bbox_pred = mx.symbol.clip(data=bbox_pred, a_min=-5, a_max=5)

        rcnn_cls_score_list.append(cls_score)
        rcnn_bbox_pred_list.append(bbox_pred)

    # concat output of each level
    cls_score_concat = mx.symbol.concat(*rcnn_cls_score_list, dim=0, name='cls_score_concat')  # [num_rois_4level, num_class]
    bbox_pred_concat = mx.symbol.concat(*rcnn_bbox_pred_list, dim=0, name='bbox_pred_concat')  # [num_rois_4level, num_class*4]

    # loss
    cls_prob = mx.symbol.SoftmaxOutput(data=cls_score_concat,
                                       label=label,
                                       multi_output=True,
                                       normalization='valid', use_ignore=True, ignore_label=-1,
                                       name='rcnn_cls_prob')
    bbox_loss_ = bbox_weight * mx.symbol.smooth_l1(name='rcnn_bbox_loss_', scalar=1.0,
                                                   data=(bbox_pred_concat - bbox_target))

    bbox_loss = mx.sym.MakeLoss(name='bbox_loss', data=bbox_loss_, grad_scale=1.0 / config.TRAIN.BATCH_ROIS)

    # reshape output
    label = mx.symbol.Reshape(data=label, shape=(config.TRAIN.BATCH_IMAGES, -1), name='label_reshape')
    cls_prob = mx.symbol.Reshape(data=cls_prob, shape=(config.TRAIN.BATCH_IMAGES, -1, num_classes),
                                 name='cls_prob_reshape')
    bbox_loss = mx.symbol.Reshape(data=bbox_loss, shape=(config.TRAIN.BATCH_IMAGES, -1, 4 * num_classes),
                                  name='bbox_loss_reshape')

    loss_group = [rpn_cls_prob, rpn_bbox_loss, cls_prob, bbox_loss, mx.symbol.BlockGrad(label)]
    group = mx.symbol.Group(loss_group)
    return group

if __name__ == "__main__":
    network = get_vgg_fpn_train(2)
    tmp = mx.viz.plot_network(network)

    tmp.view()
