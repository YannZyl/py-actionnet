# -*- coding: utf-8 -*-

__all__ = ['nms']

def compute_iou(bbox1, bbox2):
    overlap_xmin = max(bbox1[0], bbox2[0])
    overlap_ymin = max(bbox1[1], bbox2[1])
    overlap_xmax = min(bbox1[0]+bbox1[2], bbox2[0]+bbox2[2])
    overlap_ymax = min(bbox1[1]+bbox1[3], bbox2[1]+bbox2[3])
    
    overlap_area = max(0,overlap_xmax-overlap_xmin) * max(0,overlap_ymax-overlap_ymin)
    bbox1_area = bbox1[2]*bbox1[3]
    bbox2_area = bbox2[2]*bbox2[3]
    
    iou = overlap_area*1.0 / (bbox1_area + bbox2_area - overlap_area + 0.0)
    return iou
    
def nms(bboxes, thres=0.5):
    if len(bboxes) == 0:
        return bboxes
    bboxes = sorted(bboxes, key=lambda bbox:bbox[1]+bbox[3], reverse=True)
    new_bboxes = []
    new_bboxes.append(bboxes[0])
    for bbox in bboxes:
        # compute iou
        flag = True
        for bbox_new in new_bboxes:
            iou = compute_iou(bbox, bbox_new)
            if iou >= thres:
                flag = False
                break
        if flag:
            new_bboxes.append(bbox)
    
    return new_bboxes