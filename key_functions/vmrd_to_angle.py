"""

    @brief      The key member functions for loading the VMRD dataset and transform the grasp annotation format

    @author:    Yiye Chen               yychen2019@gatech.edu
    @date       08/12/2021

    The goal is to transform the VMRD grasp annotation format (8D corner coordinates) from this file:
    https://github.com/ZhangHanbo/Visual-Manipulation-Relationship-Network-Pytorch/blob/pytorch1.0/datasets/vmrd.py

    to the Multigrasp annotation format (4D unoriented bounding box coordinates + orientation angle class) from this file:
    https://github.com/ivalab/grasp_multiObject_multiGrasp/blob/master/lib/datasets/graspRGB.py#L195-L253

    Please refer to the links above for further information. Some code is adopted from them
"""

import scipy
import numpy as np
import os
import xml.etree.ElementTree as ET
import pickle


def gt_roidb(self):
    """
    Overwrite the original VMRD database loader from the link below to include the grasp annotation transformation
    https://github.com/ZhangHanbo/Visual-Manipulation-Relationship-Network-Pytorch/blob/pytorch1.0/datasets/vmrd.py

    return:
        gt_roidb: a list of dict 
            {
                'boxes' (grasp) : UNORIENTED grasp boundingboxes, (num_objs, 4)
                'gt_classes' (grasp_orientation): grasp orientation category, (num_objs, )
                'gt_overlaps' : overlaps, (num_objs, num_classes) 1 if gt_class else 0
                'flipped' : False,
                'seg_areas' : seg_areas, (num_objs, ) compute gt bbox area
                'grasp_objs' : list of str, NO REPEAT
                'grasp_inds': list of int, point to grasp_objs
            }
    """
    cache_file = os.path.join(self.cache_path, self.name + '_gt_roidb.pkl')
    if os.path.exists(cache_file):
        with open(cache_file, 'rb') as fid:
            roidb = pickle.load(fid)
        print('{} gt roidb loaded from {}'.format(self.name, cache_file))

        if self._image_set == "trainval":
            widths, heights = self.widths, self.heights
            if self.augment_with_rotation:
                self._image_index = self._image_index * 4
                self._widths, self._heights = \
                    (widths + heights) * 2, (heights + widths) * 2

    #  original VMRD roidb
    gt_roidb = [{**self._load_vmrd_annotation(index), **self._load_grasp_annotation(index)}
                for index in self.image_index]

    #  Tranform the grasp annotation. Process the VMRD format to multi_grasp format
    gt_roidb = self._process_grasp_anno(gt_roidb)

    with open(cache_file, 'wb') as fid:
        pickle.dump(gt_roidb, fid, pickle.HIGHEST_PROTOCOL)
    print('wrote gt roidb to {}'.format(cache_file))

    return gt_roidb


def _process_grasp_anno(self, gt_roidb):
    """
    The function does two things:
        1. Turn grasp annotation from vmrd format to multigrasp format(with angle), which is from the link below
        https://github.com/ivalab/grasp_multiObject_multiGrasp/blob/master/lib/datasets/graspRGB.py#L195-L253
        2. Stop distinguishing between the same-category objects in an image

    @author:    Yiye Chen               yychen2019@gatech.edu

    Input:
        gt_roidb: a list of dict with keys: 
            ['boxes'(object detection), 'gt_classes'(object detection), 'gt_ishard', 'gt_overlaps', 'seg_areas', 'node_inds', 
            'parent_lists', 'child_lists', 'rotated', 
            'grasps', 2-d list, (N, 8)
            'grasp_inds' 1-d list
    Output:
        gt_roidb: a list of dict 
        {
            'boxes' (grasp) : boxes, (num_objs, 4)
            'gt_classes' (grasp_orientation): gt_classes, (num_objs, )
            'gt_overlaps' : overlaps, (num_objs, num_classes) 1 if gt_class else 0
            'flipped' : False,
            'seg_areas' : seg_areas, (num_objs, ) compute gt bbox area
            'grasp_objs' : list of str, NO REPEAT. (e.g. [])
            'grasp_inds': list of int, point to grasp_objs
        }
    """
    def _process_single_grasp(grasp):
        """
        Process a single grasp 8-d list
        """
        import math
        # get x_min, y_min, x_max, y_max
        x_cen = (grasp[0] + grasp[2] + grasp[4] + grasp[6]) / 4
        y_cen = (grasp[1] + grasp[3] + grasp[5] + grasp[7]) / 4
        edge1_len = ((grasp[2] - grasp[0])**2 + (grasp[3] - grasp[1])**2)**0.5
        edge2_len = ((grasp[4] - grasp[2])**2 + (grasp[5] - grasp[3])**2)**0.5
        w = ((grasp[2] - grasp[0])**2 + (grasp[3] - grasp[1])
             ** 2)**0.5  # w is the gripper plate width
        h = ((grasp[4] - grasp[2])**2 + (grasp[5] - grasp[3])**2)**0.5
        x_min = x_cen - w/2
        x_max = x_cen + w/2
        y_min = y_cen - h/2
        y_max = y_cen + h/2

        # get orientation class. Orientation of gripper plate
        x_1 = grasp[0]
        y_1 = grasp[1]
        x_2 = grasp[2]
        y_2 = grasp[3]

        if x_1 > x_2:
            theta = math.atan((y_2 - y_1)/(x_1 - x_2))
        elif x_1 < x_2:
            theta = math.atan((y_1 - y_2) / (x_2 - x_1))
        else:
            theta = math.pi/2  # vertical when x_1 = x_2
        ori_cls = round((theta/math.pi*180 + 90)/10) + 1

        return ori_cls, x_min, x_max, y_min, y_max

    def _test_process_single():
        """
        For testing the processing function
        """
        import scipy
        pi = scipy.pi
        dot = scipy.dot
        sin = scipy.sin
        cos = scipy.cos
        ar = scipy.array

        def Rotate2D(pts, cnt, ang=scipy.pi/4):
            '''pts = {} Rotates points(nx2) about center cnt(2) by angle ang(1) in radian'''
            return dot(pts-cnt, ar([[cos(ang), sin(ang)], [-sin(ang), cos(ang)]]))+cnt

        # define test case
        x_min = 20
        x_max = 100
        y_min = 20
        y_max = 50
        angle = 3
        pts = ar([[x_min, y_min], [x_max, y_min],
                 [x_max, y_max], [x_min, y_max]])
        cnt = ar([(x_min + x_max)/2, (y_min + y_max)/2])
        r_bbox = Rotate2D(pts, cnt, -pi/2-pi/20*(angle-1))
        grasp = [r_bbox[0, 0], r_bbox[0, 1], r_bbox[1, 0], r_bbox[1, 1],
                 r_bbox[2, 0], r_bbox[2, 1], r_bbox[3, 0], r_bbox[3, 1]]
        ori_cls, x_min_out, x_max_out, y_min_out, y_max_out = _process_single_grasp(
            grasp)

        # testify vertices coordinate
        if x_min_out == x_min and x_max_out == x_max and y_min_out == y_min and y_max_out == y_max:
            print('Vertices coordinates recover sucessful')
        else:
            sys.exit('Vertices coordinates recover test failed')

        # testify orientation class
        if ori_cls == angle:
            print('Orientation class recover successsful')
        else:
            sys.exit('Orientation class recover failed. GT={}, Output={}'.format(
                angle, ori_cls))

        import matplotlib.pyplot as plt
        plt.figure()
        plt.plot([r_bbox[0, 0], r_bbox[1, 0]], [r_bbox[0, 1], r_bbox[1, 1]],
                 color='k', alpha=0.7, linewidth=1, solid_capstyle='round', zorder=2)
        plt.plot([r_bbox[1, 0], r_bbox[2, 0]], [r_bbox[1, 1], r_bbox[2, 1]],
                 color='r', alpha=0.7, linewidth=3, solid_capstyle='round', zorder=2)
        plt.plot([r_bbox[2, 0], r_bbox[3, 0]], [r_bbox[2, 1], r_bbox[3, 1]],
                 color='k', alpha=0.7, linewidth=1, solid_capstyle='round', zorder=2)
        plt.plot([r_bbox[3, 0], r_bbox[0, 0]], [r_bbox[3, 1], r_bbox[0, 1]],
                 color='r', alpha=0.7, linewidth=3, solid_capstyle='round', zorder=2)

        plt.plot([x_min, x_max], [y_min, y_min], color='k', alpha=0.7,
                 linewidth=1, solid_capstyle='round', zorder=2)
        plt.plot([x_max, x_max], [y_min, y_max], color='r', alpha=0.7,
                 linewidth=3, solid_capstyle='round', zorder=2)
        plt.plot([x_max, x_min], [y_max, y_max], color='k', alpha=0.7,
                 linewidth=1, solid_capstyle='round', zorder=2)
        plt.plot([x_min, x_min], [y_max, y_min], color='r', alpha=0.7,
                 linewidth=3, solid_capstyle='round', zorder=2)
        plt.show()

    def _merge_same_category_instances(objs_inds, grasp_inds):
        ''' obj_inds points to obj_classses
            grasp_inds point to objs_inds

            eg. (duplicate case):
                obj_inds: [20, 16, 16]
                grasp_inds: [1,1,1, 2,2,2, 3,3,3]
            in this case, need to merge 2,3 as they all point to 16th object. 
            i.e. the output would be:
                objs_inds_new: [20, 16]
                grasp_inds: [1,1,1, 2,2,2, 2,2,2]
        '''
        objs_inds_new = []
        num_objs = 0
        for i in range(len(objs_inds)):
            if objs_inds[i] not in objs_inds_new:
                # if not added. Add, and link corr grasp to this new index
                objs_inds_new.append(objs_inds[i])
                num_objs += 1
                for j in range(len(grasp_inds)):
                    if grasp_inds[j] == i + 1:
                        # in this case, new index is the num_objs
                        grasp_inds[j] = num_objs
            else:
                # if has been added (means same-cat-instances), link to the previously added index
                ind = objs_inds_new.index(objs_inds[i])
                for j in range(len(grasp_inds)):
                    if grasp_inds[j] == i + 1:
                        # in this case, new index is the num_objs
                        grasp_inds[j] = ind + 1

        return objs_inds_new, grasp_inds

    # _test_process_single()

    gt_roidb_new = []
    for i in range(len(gt_roidb)):
        roidb_old = gt_roidb[i]
        num_objs = len(roidb_old['grasps'])

        boxes = np.zeros((num_objs, 4), dtype=np.uint16)
        gt_classes = np.zeros((num_objs), dtype=np.int32)
        overlaps = np.zeros((num_objs, self.num_classes), dtype=np.float32)
        seg_areas = np.zeros((num_objs), dtype=np.float32)

        # Processing the grasp angles
        for ix in range(num_objs):
            cls, x1, x2, y1, y2 = _process_single_grasp(
                roidb_old['grasps'][ix])

            # if not doing this, there is negative value when bbs around boundary of image, and when it got read back, it becomes 655xx
            if (x1 < 0 and x2 < 0) or (y1 < 0 and y2 < 0):
                print('yooooooooo')
                # TODO: might occur when adding rotation. Skip this grasp if that happens
                import sys
                sys.exit()
            if x1 < 0:
                x1 = 0
            if x2 < 0:
                x2 = 0
            if y1 < 0:
                y1 = 0
            if y2 < 0:
                y2 = 0
            gt_classes[ix] = cls
            boxes[ix, :] = [x1, y1, x2, y2]
            overlaps[ix, cls] = 1.0
            seg_areas[ix] = (x2 - x1 + 1) * (y2 - y1 + 1)

        overlaps = scipy.sparse.csr_matrix(overlaps)

        # Merge the index for the same-category object
        grasp_inds = roidb_old['grasp_inds'].astype(np.int16)
        objs_inds = roidb_old['gt_classes']
        objs_inds, grasp_inds = _merge_same_category_instances(
            objs_inds, grasp_inds)
        grasp_objs = [self.obj_classes[objs_inds[i]]
                      for i in range(len(objs_inds))]

        roidb_new = {'boxes': boxes,
                     'gt_classes': gt_classes,
                     'gt_overlaps': overlaps,
                     'flipped': False,
                     'seg_areas': seg_areas,
                     'grasp_objs': grasp_objs,
                     'grasp_inds': grasp_inds
                     }
        gt_roidb_new.append(roidb_new)

    return gt_roidb_new
