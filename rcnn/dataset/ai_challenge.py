"""
Cityscape Database
"""

import cv2
import os
import numpy as np
import cPickle
import PIL.Image as Image
from imdb import IMDB
from ..processing.bbox_transform import bbox_overlaps
import json

class AiChallenge(IMDB):
    def __init__(self, image_set, root_path, dataset_path):
        """
        fill basic information to initialize imdb
        :param image_set: train or val or trainval or test
        :param root_path: 'cache' and 'rpn_data'
        :param dataset_path: data and results
        :return: imdb object
        """
        super(AiChallenge, self).__init__('aichallenge', image_set, root_path, dataset_path)
        self.image_set = image_set
        self.root_path = root_path
        self.data_path = dataset_path

        self.classes = ['__background__', 'person']
        self.num_classes = len(self.classes)
        self.image_set_index, self.annos = self.load_image_set_index()
        self.num_images = len(self.image_set_index)
        print 'num_images', self.num_images

    def load_image_set_index(self):
        """
        find out which indexes correspond to given image set (train or val)
        :return:
        """
        if self.image_set == 'train':
            self.image_data_path = os.path.join(self.data_path, 'ai_challenger_keypoint_train_20170909', 'keypoint_train_images_20170902')
            image_set_index_file = os.path.join(self.data_path, 'ai_challenger_keypoint_train_20170909', 'keypoint_train_annotations_20170909.json')
        else:
            self.image_data_path = os.path.join(self.data_path, 'ai_challenger_keypoint_validation_20170911', 'keypoint_validation_images_20170911')
            image_set_index_file = os.path.join(self.data_path, 'ai_challenger_keypoint_validation_20170911', 'keypoint_validation_annotations_20170911.json')
        assert os.path.exists(image_set_index_file), 'Path does not exist: {}'.format(image_set_index_file)
        annos = json.load(open(image_set_index_file, 'r'))
        deleteIdx = []
        for index in range(len(annos)):
            human_annotations = annos[index]['human_annotations']
            human_count = len(human_annotations.keys())
            for i in range(human_count):
                anno_key = human_annotations.keys()[i]
                rect = human_annotations[anno_key]
                if (rect[2] - rect[0]) * (rect[3] - rect[1]) <= 0 or rect[2] < rect[0] or rect[3] < rect[1]:
                    deleteIdx.append(index)
                    break
        annos = np.delete(annos, deleteIdx, axis=0)
        return range(len(annos)), annos

    def image_path_from_index(self, index):
        """
        given image index, find out full path
        :param index: index of a specific image
        :return: full path of this image
        """
        image_file = os.path.join(self.image_data_path, self.annos[index]['image_id'] + '.jpg')
        assert os.path.exists(image_file), 'Path does not exist: {}'.format(image_file)
        return image_file

    def gt_roidb(self):
        """
        return ground truth image regions database
        :return: imdb[image_index]['boxes', 'gt_classes', 'ins_id', 'ins_seg', 'gt_overlaps', 'flipped']
        """
        cache_file = os.path.join(self.cache_path, self.name + '_gt_roidb.pkl')
        if os.path.exists(cache_file):
            with open(cache_file, 'rb') as f:
                roidb = cPickle.load(f)
            print '{} gt roidb loaded from {}'.format(self.name, cache_file)
            return roidb
        gt_roidb = self.load_aichallenge_annotations()
        with open(cache_file, 'wb') as f:
            cPickle.dump(gt_roidb, f, cPickle.HIGHEST_PROTOCOL)

        return gt_roidb

    def load_aichallenge_annotations(self):
        """
        for a given index, load image and bounding boxes info from a single image list
        :return: list of record['boxes', 'gt_classes', 'ins_id', 'ins_seg', 'gt_overlaps', 'flipped']
        """
        roidb = []
        for im in range(self.num_images):
            roi_rec = dict()
            single_anno = self.annos[im]
            roi_rec['image'] = self.image_path_from_index(im)
            size = cv2.imread(roi_rec['image']).shape
            roi_rec['height'] = size[0]
            roi_rec['width'] = size[1]

            assert len(single_anno['human_annotations']) == len(single_anno['keypoint_annotations'])
            num_objs = len(single_anno['human_annotations'])

            boxes = np.zeros((num_objs, 4), dtype=np.int32)
            keypoints = np.zeros((num_objs, 14 * 3), dtype=np.float32)
            gt_classes = np.zeros((num_objs), dtype=np.int32)
            overlaps = np.zeros((num_objs, self.num_classes), dtype=np.float32)

            for ix in range(num_objs):
                box_key = 'human' + str(ix + 1)
                keypoint_key = 'human' + str(ix + 1)

                box = np.array(single_anno['human_annotations'][box_key])
                boxes[ix, :] = box

                keypoint = np.array(single_anno['keypoint_annotations'][keypoint_key])
                keypoints[ix, :] = keypoint

                gt_classes[ix] = 1
                overlaps[ix, 1] = 1.0

            roi_rec.update({'boxes': boxes,
                            'keypoints': keypoints,
                            'gt_classes': gt_classes,
                            'gt_overlaps': overlaps,
                            'max_classes': overlaps.argmax(axis=1),
                            'max_overlaps': overlaps.max(axis=1),
                            'flipped': False})

            roidb.append(roi_rec)

            if im % 5000 == 0:
                print 'Process', im

        return roidb

    def evaluate_detections(self, detections):
        raise NotImplementedError
