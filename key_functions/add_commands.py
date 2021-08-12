"""

    @ brief         Process the loaded VMRD database (with angle annotation, see the vmrd_to_angle.py file) 
                    to add the commands

    @ author        Yiye Chen               yychen3198@gatech.edu
    @ date          08/12/2021
"""

import numpy as np


def vmrd_prepare_commands(imdb, roidb):
    """
    Add the command to the database.

    Modified from the function:
    https://github.com/ivalab/grasp_multiObject_multiGrasp/blob/master/lib/roi_data_layer/roidb.py

    Input:
      roidb: a list of dict 
                {
                    'boxes' (grasp) : boxes, (num_objs, 4)
                    'gt_classes'- grasp_orientation: gt_classes, (num_objs, )
                    'gt_overlaps' : overlaps, (num_objs, num_classes) 1 if gt_class else 0. Sparse matrix
                    'flipped' : False,
                    'seg_areas' : seg_areas, (num_objs, ) compute gt bbox area
                    'grasp_objs' : list of str, obj_class, NO REPEAT
                    'grasp_inds': np.array (num_objs, )
                }
    """
    import copy
    import random
    import scipy
    roidb_new = []

    train_flag = (imdb._image_set == 'trainval')
    for i, roi in enumerate(roidb):
        # append image & width & height
        roi['image'] = imdb.image_path_at(i)  # append image
        if not (imdb.name.startswith('coco')):
            roi['width'] = imdb.widths[i]
            roi['height'] = imdb.heights[i]

        ######################
        # add a no grasp data. The command will indicate nothing from the image
        roi_new = copy.deepcopy(roi)
        obj_notcontain_list = [obj for obj in imdb._obj_classes[1:]
                               if obj not in roi['grasp_objs']]  # avoid __background__
        roi_new['command'] = gen_from_template(temp=imdb.command_temps, obj=random.choice(obj_notcontain_list), obj_only=True,
                                               is_training=train_flag)
        # need gt_overlaps as a dense array for argmax
        gt_overlaps = np.zeros(roi['gt_overlaps'].shape, dtype=np.float32)
        roi_new['gt_classes'][:] = 20  # to the __nograsp__ orientation class
        gt_overlaps[:, 20] = 1
        roi_new['gt_overlaps'] = scipy.sparse.csr_matrix(gt_overlaps)
        max_overlaps = gt_overlaps.max(axis=1)
        max_classes = gt_overlaps.argmax(axis=1)
        roi_new['max_classes'] = max_classes
        roi_new['max_overlaps'] = max_overlaps

        roidb_new.append(roi_new)

        ######################
        # for each obj in an image, generate a new data
        for i in range(len(roi['grasp_objs'])):
            roi_new = copy.deepcopy(roi)
            roi_new['command'] = gen_from_template(
                temp=imdb.command_temps, obj=roi['grasp_objs'][i], obj_only=True)
            # roi_new['command'] = roi['grasp_objs'][i]
            grasp_confirm_inds = np.where(roi['grasp_inds'] == (i+1))[0]
            nograsp_confirm_inds = np.where(roi['grasp_inds'] != (i+1))[0]
            gt_overlaps = np.zeros(roi['gt_overlaps'].shape, dtype=np.float32)
            roi_new['gt_classes'][nograsp_confirm_inds] = 20
            gt_overlaps[nograsp_confirm_inds, 20] = 1
            roi_new['gt_overlaps'] = scipy.sparse.csr_matrix(gt_overlaps)
            max_overlaps = gt_overlaps.max(axis=1)
            max_classes = gt_overlaps.argmax(axis=1)
            roi_new['max_classes'] = max_classes
            roi_new['max_overlaps'] = max_overlaps
            roidb_new.append(roi_new)

    return roidb_new


def gen_from_template(temp, obj=None, verb=None, obj_only=True, verb_only=False, is_training=False):
    """
    Generate a command form teh template
    """
    import random
    assert (obj is not None) or (
        verb is not None), 'An obj class or a verb is required for generating commands'
    if obj_only:
        template = random.choice(temp)
        if cfg.APPLY_MAPOUT and is_training:
            template = random_map_unk(template)
        sentence = template.replace("<obj>", obj)
    else:
        raise NotImplementedError

    return sentence


def random_map_unk(template):
    template_tokens = template.split(' ')
    template_tokens = [
        x for x in template_tokens if len(x) >= 1]  # skip '' token
    template_result = ""
    for token in template_tokens:
        if token == "<obj>" or np.random.uniform(0, 1) >= cfg.MAPOUT_RATE:
            template_result = template_result + ' ' + token
        else:
            template_result = template_result + ' <unk>'
    return template_result[1:]  # skip the first ' '
