import time
import numpy as np
import datetime
import pytz
import matplotlib.pyplot as plt
import seaborn as sns
import os
from terminaltables import AsciiTable
from mmcv_custom.logging_utils import get_root_logger, print_log


def calc_tp_fp(classes, coco_eval, areaRng='all', iouThr=0.75, maxDet=100, p=None, logger=None):
    '''
    Accumulate per image evaluation results and store the result in coco_eval.eval
    :param p: input params for evaluation
    :return: None
    '''
    print('Accumulating evaluation results...')
    tic = time.time()
    if not coco_eval.evalImgs:
        print('Please run evaluate() first')
    # allows input customized parameters
    if p is None:
        p = coco_eval.params
    p.catIds = p.catIds if p.useCats == 1 else [-1]
    R           = len(p.recThrs)
    K           = len(p.catIds) if p.useCats else 1
    assert K == len(classes)
    cls_to_id = {c.lower(): i for i, c in enumerate(classes)}
    precision   = -np.ones((K,R)) # -1 for the precision of absent categories
    recall      = -np.ones((K))
    scores      = -np.ones((K,R))

    targ_area_index = [i for i, aRng in enumerate(p.areaRngLbl) if aRng == areaRng][0]
    targ_iouThr_index = [i for i, thr in enumerate(p.iouThrs) if thr == iouThr][0]

    # create dictionary for future indexing
    _pe = coco_eval._paramsEval
    catIds = _pe.catIds if _pe.useCats else [-1]
    setK = set(catIds)
    setI = set(_pe.imgIds)
    # get inds to evaluate
    k_list = [n for n, k in enumerate(p.catIds)  if k in setK]
    i_list = [n for n, i in enumerate(p.imgIds)  if i in setI]
    I0 = len(_pe.imgIds)
    A0 = len(_pe.areaRng)

    det_table_data = np.array([["No.", "Data ID", "Predicted Class", "GT Class(Image)", "Confidence level", "Correct", "TP", "FP", "Precision", "Recall"]])

    for k, k0 in enumerate(k_list):
        Nk = k0*A0*I0 # k0 클래스에 대한 전체 이미지들의 추론 결과
        Na = targ_area_index*I0 # k0 클래스의 a0 사이즈에 대한 추론 결과

        E = [coco_eval.evalImgs[Nk + Na + i] for i in i_list] # k0 클래스의 a0 사이즈에 대한 전체 이미지들의 추론 결과 모음
        E = [e for e in E if not e is None]

        if len(E) == 0:
            continue
        dtScores = np.concatenate([e['dtScores'][0:maxDet] for e in E])
        imgNames = np.array([coco_eval.cocoDt.imgs[e['image_id']]['filename'] for e in E for _ in e['dtScores'][0:maxDet]])

        # different sorting method generates slightly different results.
        # mergesort is used to be consistent as Matlab implementation.
        inds = np.argsort(-dtScores, kind='mergesort')
        dtScoresSorted = dtScores[inds]
        imgNameSorted = imgNames[inds]
        n_samples = len(det_table_data) - 1
        nos = [n_samples+i for i in range(len(imgNameSorted))]

        dtm  = np.concatenate([e['dtMatches'][:,0:maxDet] for e in E], axis=1)[:,inds]
        dtIg = np.concatenate([e['dtIgnore'][:,0:maxDet]  for e in E], axis=1)[:,inds]
        gtIg = np.concatenate([e['gtIgnore'] for e in E])
        npig = np.count_nonzero(gtIg==0 )
        if npig == 0:
            continue
        tps = np.logical_and(               dtm,  np.logical_not(dtIg) )
        fps = np.logical_and(np.logical_not(dtm), np.logical_not(dtIg) )

        tp_sum = np.cumsum(tps, axis=1).astype(dtype=np.float) # (10(threshold 갯수), num_detected(class & area 기준))
        fp_sum = np.cumsum(fps, axis=1).astype(dtype=np.float) # (10, num_detected(class & area 기준))

        print('sorted imgnames: ', imgNameSorted.shape, imgNameSorted)
        tp_orig = tps[targ_iouThr_index]

        tp = np.array(tp_sum[targ_iouThr_index])
        fp = np.array(fp_sum[targ_iouThr_index])
        nd = len(tp)
        rc = tp / npig
        pr = tp / (fp+tp+np.spacing(1))
        q  = np.zeros((R,))
        ss = np.zeros((R,))

        if nd:
            recall[k] = rc[-1]
        else:
            recall[k] = 0

        # numpy is slow without cython optimization for accessing elements
        # use python array gets significant speed improvement
        pr = pr.tolist(); q = q.tolist()

        for i in range(nd-1, 0, -1):
            if pr[i] > pr[i-1]:
                pr[i-1] = pr[i]

        inds = np.searchsorted(rc, p.recThrs, side='left')
        try:
            for ri, pi in enumerate(inds):
                q[ri] = pr[pi]
                ss[ri] = dtScoresSorted[pi]
        except:
            pass
        precision[k, :] = np.array(q)
        scores[k, :] = np.array(ss)

        print('0', np.array(nos).shape)
        print('1', imgNameSorted.shape)
        results = np.concatenate(
            [
                np.array([nos]),
                np.array([imgNameSorted]),
                np.expand_dims(np.full(len(imgNameSorted), k), axis=0),
                np.array([[cls_to_id[i.split("/")[-1].split("_")[0].lower()] for i in imgNameSorted]]),
                np.around(np.array([dtScoresSorted]), decimals=2),
                np.array([tp_orig]),
                np.array([tp]),
                np.array([fp]),
                np.around(np.array([pr]), decimals=2),
                np.around(np.array([rc]), decimals=2),
            ],
            axis=0
        ).transpose()
        # print("Result shape!", results.shape)
        """
        # assert results.shape == (len(E), EVAL_COLS)
        # results[:, 0] = nos
        # results[:, 1] = np.array(imgNameSorted)
        # results[:, 2] = np.full((len(E), EVAL_COLS), k)
        # results[:, 3] = np.array([cls_to_id[i.split("/")[-1].split("_")[0]] for i in imgNameSorted])
        # results[:, 4] = np.array(dtScoresSorted)
        # results[:, 5] = np.array(tp_orig)
        # results[:, 6] = np.array(tp)
        # results[:, 7] = np.array(fp)
        # results[:, 8] = np.array(pr)
        # results[:, 9] = np.array(rc)
        """
        det_table_data = np.concatenate([det_table_data, results], axis=0)
        print("Data shape!", det_table_data.shape)
        # cls_data_table = AsciiTable(results.tolist())
        # print(cls_data_table.table)

    det_per_bbox_table = AsciiTable(det_table_data.tolist())
    ap_per_class = np.array([["Class", "AP"]])
    cls_ids = np.array([[c for c in classes]])
    ap_per_class_res = np.around(np.array([precision.mean(axis=1)]), decimals=2)
    ap_per_class = np.concatenate([ap_per_class, np.concatenate([cls_ids, ap_per_class_res], axis=0).transpose()], axis=0)
    ap_per_class_table = AsciiTable(ap_per_class.tolist())
    map_value = precision.mean()

    if not logger:
        logger = get_root_logger()
    print_log(f"\nDetection results per sample, with iou threshold {iouThr} within object {areaRng} area range, with maximum {maxDet} detection per image!", logger=logger)
    print_log('\n'+det_per_bbox_table.table, logger=logger)
    print_log(f"\nAP per class, with iou threshold {iouThr} within object {areaRng} area range, with maximum {maxDet} detection per image: ", logger=logger)
    print_log('\n'+ap_per_class_table.table, logger=logger)
    print_log(f"Calculated mAP, with iou threshold {iouThr}, max detections {maxDet} with object {areaRng} area range:  {round(map_value, 2)}", logger=logger)


    toc = time.time()
    print('DONE (t={:0.2f}s).'.format( toc-tic))
    return precision, p.recThrs


def draw_precision_recall_curve(classes, precisions, recall_thresholds, save_dir="./precision_recall_curves"):
    """
    - 여기서 precision들은 recall threshold 지점에 대응하는 precision 값들을 나타냄
    - recall_threshold는 0, 0.01, 0.02, ..., 1까지 가는 threshold
    """
    os.makedirs(save_dir, exist_ok=True)
    for c, cls in enumerate(classes):
        class_precision = precisions[c]
        plt.ylim([0.0, 1.0])
        plt.figure()
        plot = sns.lineplot(x=recall_thresholds, y=class_precision).set_title(cls.upper())
        fig = plot.get_figure()
        fig.savefig(os.path.join(save_dir, f"{cls}_precision_recall_curve.png"))
        plt.close()