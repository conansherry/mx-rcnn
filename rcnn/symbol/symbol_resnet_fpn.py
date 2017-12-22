import mxnet as mx

from rcnn.config import config
from rcnn.symbol import fpn_roi_pooling, proposal_fpn

eps = 2e-5
use_global_stats = True
workspace = 512
res_deps = {'18': (2, 2, 2, 2), '34': (3, 4, 6, 3), '50': (3, 4, 6, 3), '101': (3, 4, 23, 3), '152': (3, 8, 36, 3), '200': (3, 24, 36, 3)}
res_type = '34'
units = res_deps[res_type]
if res_type != '34' and res_type != '18':
    filter_list = [256, 512, 1024, 2048]
else:
    filter_list = [64, 128, 256, 512]

def residual_unit(data, num_filter, stride, dim_match, name):
    if res_type == '34' or res_type == '18':
        bn1 = mx.sym.BatchNorm(data=data, fix_gamma=False, eps=eps, use_global_stats=use_global_stats,
                               name=name + '_bn1')
        act1 = mx.sym.Activation(data=bn1, act_type='relu', name=name + '_relu1')
        conv1 = mx.sym.Convolution(data=act1, num_filter=int(num_filter), kernel=(3, 3), stride=stride, pad=(1, 1),
                                   no_bias=True, workspace=workspace, name=name + '_conv1')
        bn2 = mx.sym.BatchNorm(data=conv1, fix_gamma=False, eps=eps, use_global_stats=use_global_stats,
                               name=name + '_bn2')
        act2 = mx.sym.Activation(data=bn2, act_type='relu', name=name + '_relu2')
        conv2 = mx.sym.Convolution(data=act2, num_filter=int(num_filter), kernel=(3, 3), stride=(1, 1), pad=(1, 1),
                                   no_bias=True, workspace=workspace, name=name + '_conv2')
        if dim_match:
            shortcut = data
        else:
            shortcut = mx.sym.Convolution(data=act1, num_filter=num_filter, kernel=(1, 1), stride=stride, no_bias=True,
                                          workspace=workspace, name=name + '_sc')
        sum = mx.sym.ElementWiseSum(*[conv2, shortcut], name=name + '_plus')
    else:
        bn1 = mx.sym.BatchNorm(data=data, fix_gamma=False, eps=eps, use_global_stats=use_global_stats,
                               name=name + '_bn1')
        act1 = mx.sym.Activation(data=bn1, act_type='relu', name=name + '_relu1')
        conv1 = mx.sym.Convolution(data=act1, num_filter=int(num_filter * 0.25), kernel=(1, 1), stride=(1, 1),
                                   pad=(0, 0),
                                   no_bias=True, workspace=workspace, name=name + '_conv1')
        bn2 = mx.sym.BatchNorm(data=conv1, fix_gamma=False, eps=eps, use_global_stats=use_global_stats,
                               name=name + '_bn2')
        act2 = mx.sym.Activation(data=bn2, act_type='relu', name=name + '_relu2')
        conv2 = mx.sym.Convolution(data=act2, num_filter=int(num_filter * 0.25), kernel=(3, 3), stride=stride,
                                   pad=(1, 1),
                                   no_bias=True, workspace=workspace, name=name + '_conv2')
        bn3 = mx.sym.BatchNorm(data=conv2, fix_gamma=False, eps=eps, use_global_stats=use_global_stats,
                               name=name + '_bn3')
        act3 = mx.sym.Activation(data=bn3, act_type='relu', name=name + '_relu3')
        conv3 = mx.sym.Convolution(data=act3, num_filter=num_filter, kernel=(1, 1), stride=(1, 1), pad=(0, 0),
                                   no_bias=True,
                                   workspace=workspace, name=name + '_conv3')
        if dim_match:
            shortcut = data
        else:
            shortcut = mx.sym.Convolution(data=act1, num_filter=num_filter, kernel=(1, 1), stride=stride, no_bias=True,
                                          workspace=workspace, name=name + '_sc')
        sum = mx.sym.ElementWiseSum(*[conv3, shortcut], name=name + '_plus')
    return sum

def get_resnet_conv(data):
    # res1
    data_bn = mx.sym.BatchNorm(data=data, fix_gamma=True, eps=eps, use_global_stats=use_global_stats, name='bn_data')
    conv0   = mx.sym.Convolution(data=data_bn, num_filter=64, kernel=(7, 7), stride=(2, 2), pad=(3, 3),
                               no_bias=True, name="conv0", workspace=workspace)
    bn0   = mx.sym.BatchNorm(data=conv0, fix_gamma=False, eps=eps, use_global_stats=use_global_stats, name='bn0')
    relu0 = mx.sym.Activation(data=bn0, act_type='relu', name='relu0')
    pool0 = mx.symbol.Pooling(data=relu0, kernel=(3, 3), stride=(2, 2), pad=(1, 1), pool_type='max', name='pool0')

    # res2
    unit = residual_unit(data=pool0, num_filter=filter_list[0], stride=(1, 1), dim_match=False, name='stage1_unit1')
    for i in range(2, units[0] + 1):
        unit = residual_unit(data=unit, num_filter=filter_list[0], stride=(1, 1), dim_match=True,
                             name='stage1_unit%s' % i)
    conv_C2 = unit

    # res3
    unit = residual_unit(data=unit, num_filter=filter_list[1], stride=(2, 2), dim_match=False, name='stage2_unit1')
    for i in range(2, units[1] + 1):
        unit = residual_unit(data=unit, num_filter=filter_list[1], stride=(1, 1), dim_match=True,
                             name='stage2_unit%s' % i)
    conv_C3 = unit

    # res4
    unit = residual_unit(data=unit, num_filter=filter_list[2], stride=(2, 2), dim_match=False, name='stage3_unit1')
    for i in range(2, units[2] + 1):
        unit = residual_unit(data=unit, num_filter=filter_list[2], stride=(1, 1), dim_match=True,
                             name='stage3_unit%s' % i)
    conv_C4 = unit

    # res5
    unit = residual_unit(data=unit, num_filter=filter_list[3], stride=(2, 2), dim_match=False, name='stage4_unit1')
    for i in range(2, units[3] + 1):
        unit = residual_unit(data=unit, num_filter=filter_list[3], stride=(1, 1), dim_match=True,
                             name='stage4_unit%s' % i)
    conv_C5 = unit

    conv_feat = [conv_C5, conv_C4, conv_C3, conv_C2]
    return conv_feat

def get_resnet_conv_down(conv_feat, num_filter=256):
    # C5 to P5, 1x1 dimension reduction to num_filter
    P5 = mx.symbol.Convolution(data=conv_feat[0], kernel=(1, 1), num_filter=num_filter, name="P5_lateral")

    # P5 2x upsampling + C4 = P4
    # P5_up = mx.sym.Deconvolution(data=P5, kernel=(4, 4), stride=(2, 2), pad=(1, 1), num_filter=num_filter, adj=(1, 1),
    #                              name='P5_upsampling')
    P5_up = mx.symbol.UpSampling(P5, scale=2, sample_type='nearest', workspace=512, name='P5_upsampling', num_args=1)
    P4_la = mx.symbol.Convolution(data=conv_feat[1], kernel=(1, 1), num_filter=num_filter, name="P4_lateral")
    P5_clip = mx.symbol.Crop(*[P5_up, P4_la], name="P4_clip")
    P4 = mx.sym.ElementWiseSum(*[P5_clip, P4_la], name="P4_sum")
    P4 = mx.symbol.Convolution(data=P4, kernel=(3, 3), pad=(1, 1), num_filter=num_filter, name="P4_aggregate")

    # P4 2x upsampling + C3 = P3
    # P4_up = mx.sym.Deconvolution(data=P4, kernel=(4, 4), stride=(2, 2), pad=(1, 1), num_filter=num_filter, adj=(1, 1),
    #                              name='P4_upsampling')
    P4_up = mx.symbol.UpSampling(P4, scale=2, sample_type='nearest', workspace=512, name='P4_upsampling', num_args=1)
    P3_la = mx.symbol.Convolution(data=conv_feat[2], kernel=(1, 1), num_filter=num_filter, name="P3_lateral")
    P4_clip = mx.symbol.Crop(*[P4_up, P3_la], name="P3_clip")
    P3 = mx.sym.ElementWiseSum(*[P4_clip, P3_la], name="P3_sum")
    P3 = mx.symbol.Convolution(data=P3, kernel=(3, 3), pad=(1, 1), num_filter=num_filter, name="P3_aggregate")

    # P3 2x upsampling + C2 = P2
    # P3_up = mx.sym.Deconvolution(data=P3, kernel=(4, 4), stride=(2, 2), pad=(1, 1), num_filter=num_filter, adj=(1, 1),
    #                              name='P3_upsampling')
    P3_up = mx.symbol.UpSampling(P3, scale=2, sample_type='nearest', workspace=512, name='P3_upsampling', num_args=1)
    P2_la = mx.symbol.Convolution(data=conv_feat[3], kernel=(1, 1), num_filter=num_filter, name="P2_lateral")
    P3_clip = mx.symbol.Crop(*[P3_up, P2_la], name="P2_clip")
    P2 = mx.sym.ElementWiseSum(*[P3_clip, P2_la], name="P2_sum")
    P2 = mx.symbol.Convolution(data=P2, kernel=(3, 3), pad=(1, 1), num_filter=num_filter, name="P2_aggregate")

    conv_fpn_feat = dict()
    conv_fpn_feat.update({"stride32": P5, "stride16": P4, "stride8": P3, "stride4": P2})

    return conv_fpn_feat, [P5, P4, P3, P2]

def get_resnet_fpn_rpn(num_anchors=config.NUM_ANCHORS):
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

    # shared convolutional layers, bottom up
    conv_feat = get_resnet_conv(data)
    # shared convolutional layers, top down
    conv_fpn_feat, _ = get_resnet_conv_down(conv_feat)

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

        rpn_bbox_pred = mx.symbol.clip(data=rpn_bbox_pred, a_min=-10, a_max=10)

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

    loss_group = [rpn_cls_prob, rpn_bbox_loss]
    group = mx.symbol.Group(loss_group)
    return group

def get_resnet_fpn_test(num_classes=config.NUM_CLASSES, num_anchors=config.NUM_ANCHORS):
    data = mx.symbol.Variable(name="data")
    im_info = mx.symbol.Variable(name="im_info")

    # shared convolutional layers
    conv_feat = get_resnet_conv(data)
    conv_fpn_feat, _ = get_resnet_conv_down(conv_feat)

    # # shared parameters for predictions
    rpn_conv_weight      = mx.symbol.Variable('rpn_conv_weight')
    rpn_conv_bias        = mx.symbol.Variable('rpn_conv_bias')
    rpn_conv_cls_weight  = mx.symbol.Variable('rpn_conv_cls_weight')
    rpn_conv_cls_bias    = mx.symbol.Variable('rpn_conv_cls_bias')
    rpn_conv_bbox_weight = mx.symbol.Variable('rpn_conv_bbox_weight')
    rpn_conv_bbox_bias   = mx.symbol.Variable('rpn_conv_bbox_bias')

    rcnn_fc6_weight = mx.symbol.Variable('rcnn_fc6_weight')
    rcnn_fc6_bias   = mx.symbol.Variable('rcnn_fc6_bias')
    rcnn_fc7_weight = mx.symbol.Variable('rcnn_fc7_weight')
    rcnn_fc7_bias   = mx.symbol.Variable('rcnn_fc7_bias')
    rcnn_fc_cls_weight  = mx.symbol.Variable('rcnn_fc_cls_weight')
    rcnn_fc_cls_bias    = mx.symbol.Variable('rcnn_fc_cls_bias')
    rcnn_fc_bbox_weight = mx.symbol.Variable('bbox_pred_weight')
    rcnn_fc_bbox_bias = mx.symbol.Variable('bbox_pred_bias')

    rpn_cls_prob_dict = {}
    rpn_bbox_pred_dict = {}
    for stride in config.RPN_FEAT_STRIDE:
        rpn_conv = mx.symbol.Convolution(data=conv_fpn_feat['stride%s'%stride],
                                         kernel=(3, 3), pad=(1, 1),
                                         num_filter=512,
                                         weight=rpn_conv_weight,
                                         bias=rpn_conv_bias)
        rpn_relu = mx.symbol.Activation(data=rpn_conv,
                                        act_type="relu",
                                        name="rpn_relu")
        rpn_cls_score = mx.symbol.Convolution(data=rpn_relu,
                                              kernel=(1, 1), pad=(0, 0),
                                              num_filter=2 * num_anchors,
                                              weight=rpn_conv_cls_weight,
                                              bias=rpn_conv_cls_bias,
                                              name="rpn_cls_score_stride%s"%stride)
        rpn_bbox_pred = mx.symbol.Convolution(data=rpn_relu,
                                              kernel=(1, 1), pad=(0, 0),
                                              num_filter=4 * num_anchors,
                                              weight=rpn_conv_bbox_weight,
                                              bias=rpn_conv_bbox_bias,
                                              name="rpn_bbox_pred_stride%s" % stride)

        # ROI Proposal
        rpn_cls_score_reshape = mx.symbol.Reshape(data=rpn_cls_score,
                                                  shape=(0, 2, -1, 0),
                                                  name="rpn_cls_score_reshape")
        rpn_cls_prob = mx.symbol.SoftmaxActivation(data=rpn_cls_score_reshape,
                                                   mode="channel",
                                                   name="rpn_cls_prob_stride%s" % stride)
        rpn_cls_prob_reshape = mx.symbol.Reshape(data=rpn_cls_prob,
                                                 shape=(0, 2 * num_anchors, -1, 0),
                                                 name='rpn_cls_prob_reshape')

        rpn_cls_prob_dict.update({'rpn_cls_prob_stride%s'%stride:rpn_cls_prob_reshape})
        rpn_bbox_pred_dict.update({'rpn_bbox_pred_stride%s'%stride:rpn_bbox_pred})

    args_dict = dict(rpn_cls_prob_dict.items()+rpn_bbox_pred_dict.items())
    aux_dict = {'im_info':im_info,'name':'rois',
                'op_type':'proposal_fpn','output_score':False,
                'feat_stride':config.RPN_FEAT_STRIDE,'scales':tuple(config.ANCHOR_SCALES),
                'rpn_pre_nms_top_n':config.TEST.RPN_PRE_NMS_TOP_N,
                'rpn_post_nms_top_n':config.TEST.RPN_POST_NMS_TOP_N,
                'rpn_min_size':config.RPN_FEAT_STRIDE,
                'threshold':config.TEST.RPN_NMS_THRESH}
    # Proposal
    rois = mx.symbol.Custom(**dict(args_dict.items()+aux_dict.items()))

    # FPN roi pooling
    args_dict={}
    for s in config.RCNN_FEAT_STRIDE:
        args_dict.update({'feat_stride%s'%s: conv_fpn_feat['stride%s'%s]})
    args_dict.update({'rois':rois, 'name':'fpn_roi_pool',
                      'op_type':'fpn_roi_pool',
                      'rcnn_strides':config.RCNN_FEAT_STRIDE,
                      'pool_h':14, 'pool_w':14})
    roi_pool_fpn = mx.symbol.Custom(**args_dict)

    # classification with fc layers
    flatten = mx.symbol.Flatten(data=roi_pool_fpn, name="flatten")
    fc6     = mx.symbol.FullyConnected(data=flatten, num_hidden=1024, weight=rcnn_fc6_weight, bias=rcnn_fc6_bias)
    relu6   = mx.symbol.Activation(data=fc6, act_type="relu", name="rcnn_relu6")
    drop6   = mx.symbol.Dropout(data=relu6, p=0.5, name="drop6")
    fc7     = mx.symbol.FullyConnected(data=drop6, num_hidden=1024, weight=rcnn_fc7_weight, bias=rcnn_fc7_bias)
    relu7   = mx.symbol.Activation(data=fc7, act_type="relu", name="rcnn_relu7")

    # classification
    rcnn_cls_score = mx.symbol.FullyConnected(data=relu7, weight=rcnn_fc_cls_weight,
                                              bias=rcnn_fc_cls_bias, num_hidden=num_classes)
    rcnn_cls_prob  = mx.symbol.SoftmaxActivation(name='rcnn_cls_prob', data=rcnn_cls_score)
    # bounding box regression
    rcnn_bbox_pred = mx.symbol.FullyConnected(data=relu7, weight=rcnn_fc_bbox_weight,
                                              bias=rcnn_fc_bbox_bias, num_hidden=num_classes * 4)

    # reshape output
    rcnn_cls_prob  = mx.symbol.Reshape(data=rcnn_cls_prob, shape=(config.TEST.BATCH_IMAGES, -1, num_classes),
                                 name='cls_prob_reshape')
    rcnn_bbox_pred = mx.symbol.Reshape(data=rcnn_bbox_pred, shape=(config.TEST.BATCH_IMAGES, -1, 4 * num_classes),
                                 name='bbox_pred_reshape')

    group = mx.symbol.Group([rois, rcnn_cls_prob, rcnn_bbox_pred])
    return group

def get_resnet_fpn_train(num_classes=config.NUM_CLASSES, num_anchors=config.NUM_ANCHORS):
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

    # shared convolutional layers, bottom up
    conv_feat = get_resnet_conv(data)
    # shared convolutional layers, top down
    conv_fpn_feat, _ = get_resnet_conv_down(conv_feat)

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
            pooled_size=(14, 14),
            spatial_scale=1.0 / stride)

        # group 6
        flatten = mx.symbol.Flatten(data=roi_pool, name="flatten_stride%s" % stride)
        fc6 = mx.symbol.FullyConnected(data=flatten, num_hidden=1024,
                                       weight=rcnn_fc6_weight,
                                       bias=rcnn_fc6_bias,
                                       name="rcnn_fc6_stride%s" % stride
                                       )
        relu6 = mx.symbol.Activation(data=fc6, act_type="relu", name="relu6_stride%s" % stride)
        drop6 = mx.symbol.Dropout(data=relu6, p=0.5, name="drop6_stride%s" % stride)
        # group 7
        fc7 = mx.symbol.FullyConnected(data=drop6, num_hidden=1024,
                                       weight=rcnn_fc7_weight,
                                       bias=rcnn_fc7_bias,
                                       name="rcnn_fc7_stride%s" % stride
                                       )
        relu7 = mx.symbol.Activation(data=fc7, act_type="relu", name="relu7_stride%s" % stride)
        # classification
        cls_score = mx.symbol.FullyConnected(data=relu7, num_hidden=num_classes,
                                             weight=rcnn_fc_cls_weight,
                                             bias=rcnn_fc_cls_bias,
                                             name='rcnn_cls_score_stride%s' % stride
                                             )
        # bounding box regression
        bbox_pred = mx.symbol.FullyConnected(data=relu7, num_hidden=num_classes * 4,
                                             weight=rcnn_fc_bbox_weight,
                                             bias=rcnn_fc_bbox_bias,
                                             name='rcnn_bbox_pred_stride%s' % stride
                                             )

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
    # network = get_resnet_fpn_rpn()
    # network = get_resnet_fpn_mask_test()
    network = get_resnet_fpn_train()

    tmp = mx.viz.plot_network(network, save_format='png')
    tmp.view()