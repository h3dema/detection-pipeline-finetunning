"""
The CocoEvaluator class is designed to handle the evaluation of object detection models using the COCO dataset.
It provides a convenient way to update the evaluator with new predictions, synchronize the evaluator across multiple processes, and summarize the evaluation results.

Important methods:

- prepare: Prepares the predictions for evaluation. It checks the IOU type and calls the corresponding method to prepare the predictions.
- prepare_for_coco_detection: It converts the bounding box coordinates to the COCO format, extracts the scores and labels, and creates a list of dictionaries containing the image ID, category ID, bounding box, and score.
- prepare_for_coco_segmentation: It converts the segmentation masks to the COCO format, extracts the scores and labels, and creates a list of dictionaries containing the image ID, category ID, segmentation mask, and score.
- prepare_for_coco_keypoint: It converts the keypoints to the COCO format, extracts the scores and labels, and creates a list of dictionaries containing the image ID, category ID, keypoints, and score.

"""

import copy
import io
from contextlib import redirect_stdout

import numpy as np
import pycocotools.mask as mask_util
import torch
from torch_utils import utils
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval


class CocoEvaluator:
    """
        The CocoEvaluator class is a utility class for evaluating the performance of object detection models using the COCO (Common Objects in Context) dataset.
        It provides methods to update the evaluator with new predictions, synchronize the evaluator across multiple processes, accumulate the evaluation results, and
        summarize the evaluation results.

    """

    def __init__(self, coco_gt, iou_types):
        """
        Initializes the CocoEvaluator with the ground truth COCO dataset and IOU types.

        Args:
            coco_gt (COCO): The ground truth COCO dataset.
            iou_types (list or tuple): A list or tuple of IOU types to evaluate, e.g., ['bbox', 'segm'].

        Attributes:
            coco_gt (COCO): A deep copy of the ground truth COCO dataset.
            iou_types (list or tuple): The IOU types to evaluate.
            coco_eval (dict): A dictionary mapping IOU types to COCOeval objects.
            img_ids (list): A list of image IDs that have been evaluated.
            eval_imgs (dict): A dictionary mapping IOU types to lists of evaluation images.
        """
        assert isinstance(iou_types, (list, tuple))
        coco_gt = copy.deepcopy(coco_gt)
        self.coco_gt = coco_gt

        self.iou_types = iou_types
        self.coco_eval = {}
        for iou_type in iou_types:
            self.coco_eval[iou_type] = COCOeval(coco_gt, iouType=iou_type)

        self.img_ids = []
        self.eval_imgs = {k: [] for k in iou_types}

    def update(self, predictions):
        """
        Updates the CocoEvaluator with new predictions.

        Args:
            predictions (dict): A dictionary mapping image IDs to lists of predictions, where each prediction is a dictionary containing
                                the keys 'bbox', 'scores', 'labels', and 'masks' (if panoptic segmentation is enabled).

        Note that the predictions are modified in-place to contain the rescaled bounding boxes and masks, and the predictions are
        then converted to COCO format. The evaluation results are accumulated in the eval_imgs attribute.
        """

        img_ids = list(np.unique(list(predictions.keys())))
        self.img_ids.extend(img_ids)

        for iou_type in self.iou_types:
            results = self.prepare(predictions, iou_type)
            with redirect_stdout(io.StringIO()):
                coco_dt = COCO.loadRes(self.coco_gt, results) if results else COCO()
            coco_eval = self.coco_eval[iou_type]

            coco_eval.cocoDt = coco_dt
            coco_eval.params.imgIds = list(img_ids)
            img_ids, eval_imgs = evaluate(coco_eval)

            self.eval_imgs[iou_type].append(eval_imgs)

    def synchronize_between_processes(self):
        """
        Synchronizes the CocoEvaluator between multiple processes.

        This method is a no-op if not using distributed evaluation. Otherwise, it is
        called once for each process and performs the following operations:

        1. Concatenates the evaluation results for each IOU type and stores them in
           the eval_imgs attribute.
        2. Calls create_common_coco_eval to create a COCOeval object that can be
           used to summarize the evaluation results across all processes.

        Attributes:
            eval_imgs (dict): A dictionary mapping IOU types to concatenated
                              evaluation results.
        """
        for iou_type in self.iou_types:
            self.eval_imgs[iou_type] = np.concatenate(self.eval_imgs[iou_type], 2)
            create_common_coco_eval(self.coco_eval[iou_type], self.img_ids, self.eval_imgs[iou_type])

    def accumulate(self):
        """
        Accumulates the evaluation results for each IOU type.

        This method is a no-op if not using distributed evaluation. Otherwise, it is
        called once for each process and accumulates the evaluation results for each
        IOU type in the coco_eval attribute.
        """
        for coco_eval in self.coco_eval.values():
            coco_eval.accumulate()

    def summarize(self):
        """
        Summarizes the evaluation results for each IoU type.

        This method iterates over the COCO evaluation objects stored in the coco_eval
        attribute. For each IoU type, it prints the IoU metric and calls the summarize
        method on the respective COCOeval object to display a summary of the evaluation
        results. It returns the statistics of the last evaluated IoU type.

        Returns:
            np.ndarray: The evaluation statistics of the COCOeval object for the last IoU type.
        """
        for iou_type, coco_eval in self.coco_eval.items():
            print(f"IoU metric: {iou_type}")
            coco_eval.summarize()
        return coco_eval.stats

    def prepare(self, predictions, iou_type):
        """
        Prepares the predictions for evaluation by the COCO evaluation object.

        Args:
            predictions (dict): A dictionary mapping image IDs to lists of predictions, where each prediction is a dictionary containing
                                the keys 'bbox', 'scores', 'labels', and 'masks' (if panoptic segmentation is enabled).
            iou_type (str): The IOU type to evaluate, e.g., 'bbox', 'segm', or 'keypoints'.

        Returns:
            list: A list of predictions in COCO format, where each prediction is a dictionary containing the keys 'image_id', 'category_id', 'bbox', 'score', and
                  possibly 'segmentation' or 'keypoints' if panoptic segmentation or keypoint detection is enabled.

        Raises:
            ValueError: If the IOU type is unknown.
        """
        if iou_type == "bbox":
            return self.prepare_for_coco_detection(predictions)
        if iou_type == "segm":
            return self.prepare_for_coco_segmentation(predictions)
        if iou_type == "keypoints":
            return self.prepare_for_coco_keypoint(predictions)
        raise ValueError(f"Unknown iou type {iou_type}")

    def prepare_for_coco_detection(self, predictions):
        """
        Prepares the predictions for COCO object detection evaluation.

        Args:
            predictions (dict): A dictionary mapping image IDs to lists of predictions, where each prediction is a dictionary containing
                                the keys 'boxes', 'scores', and 'labels'.

        Returns:
            list: A list of predictions in COCO format, where each prediction is a dictionary containing the keys 'image_id', 'category_id', 'bbox', and 'score'.
        """

        coco_results = []
        for original_id, prediction in predictions.items():
            if len(prediction) == 0:
                continue

            boxes = prediction["boxes"]
            boxes = convert_to_xywh(boxes).tolist()
            scores = prediction["scores"].tolist()
            labels = prediction["labels"].tolist()

            coco_results.extend(
                [
                    {
                        "image_id": original_id,
                        "category_id": labels[k],
                        "bbox": box,
                        "score": scores[k],
                    }
                    for k, box in enumerate(boxes)
                ]
            )
        return coco_results

    def prepare_for_coco_segmentation(self, predictions):
        """
        Prepare COCO-format results for segmentation task.

        Args:
            predictions (dict): A dictionary with the following key-value pairs:
                - "image_id" (int): The image id.
                - "scores" (torch.Tensor of shape (n, )): The scores of the predictions.
                - "labels" (torch.Tensor of shape (n, )): The labels of the predictions.
                - "masks" (torch.Tensor of shape (n, h, w)): The masks of the predictions.

        Returns:
            list: A list of predictions in COCO format, where each prediction is a dictionary containing the keys 'image_id', 'category_id', 'segmentation', and 'score'.
        """
        coco_results = []
        for original_id, prediction in predictions.items():
            if len(prediction) == 0:
                continue

            scores = prediction["scores"]
            labels = prediction["labels"]
            masks = prediction["masks"]

            masks = masks > 0.5

            scores = prediction["scores"].tolist()
            labels = prediction["labels"].tolist()

            rles = [
                mask_util.encode(np.array(mask[0, :, :, np.newaxis], dtype=np.uint8, order="F"))[0] for mask in masks
            ]
            for rle in rles:
                rle["counts"] = rle["counts"].decode("utf-8")

            coco_results.extend(
                [
                    {
                        "image_id": original_id,
                        "category_id": labels[k],
                        "segmentation": rle,
                        "score": scores[k],
                    }
                    for k, rle in enumerate(rles)
                ]
            )
        return coco_results

    def prepare_for_coco_keypoint(self, predictions):
        """
        Prepares predictions for COCO keypoint detection evaluation.

        Args:
            predictions (dict): A dictionary mapping image IDs to lists of predictions,
                                where each prediction is a dictionary containing
                                the keys 'boxes', 'scores', 'labels', and 'keypoints'.

        Returns:
            list: A list of predictions in COCO format, where each prediction is a
                dictionary containing the keys 'image_id', 'category_id', 'keypoints',
                and 'score'.
        """
        coco_results = []
        for original_id, prediction in predictions.items():
            if len(prediction) == 0:
                continue

            boxes = prediction["boxes"]
            boxes = convert_to_xywh(boxes).tolist()
            scores = prediction["scores"].tolist()
            labels = prediction["labels"].tolist()
            keypoints = prediction["keypoints"]
            keypoints = keypoints.flatten(start_dim=1).tolist()

            coco_results.extend(
                [
                    {
                        "image_id": original_id,
                        "category_id": labels[k],
                        "keypoints": keypoint,
                        "score": scores[k],
                    }
                    for k, keypoint in enumerate(keypoints)
                ]
            )
        return coco_results


def convert_to_xywh(boxes):
    """
    Converts bounding boxes from (x1, y1, x2, y2) format to (x, y, w, h) format.

    Args:
        boxes (Tensor[N, 4]): boxes in (x1, y1, x2, y2) format

    Returns:
        Tensor[N, 4]: boxes in (x, y, w, h) format
    """
    xmin, ymin, xmax, ymax = boxes.unbind(1)
    return torch.stack((xmin, ymin, xmax - xmin, ymax - ymin), dim=1)


def merge(img_ids, eval_imgs):
    """
    Merges image IDs and evaluation images across multiple processes.

    This function gathers image IDs and evaluation images from all processes,
    concatenates them, and retains only unique image IDs and their corresponding
    evaluation images in sorted order.

    Args:
        img_ids (list): A list of image IDs to be merged across processes.
        eval_imgs (list): A list of evaluation images corresponding to the image IDs.

    Returns:
        tuple: A tuple containing:
            - merged_img_ids (np.ndarray): An array of unique, sorted image IDs.
            - merged_eval_imgs (np.ndarray): An array of evaluation images corresponding
              to the unique image IDs.
    """
    all_img_ids = utils.all_gather(img_ids)
    all_eval_imgs = utils.all_gather(eval_imgs)

    merged_img_ids = []
    for p in all_img_ids:
        merged_img_ids.extend(p)

    merged_eval_imgs = []
    for p in all_eval_imgs:
        merged_eval_imgs.append(p)

    merged_img_ids = np.array(merged_img_ids)
    merged_eval_imgs = np.concatenate(merged_eval_imgs, 2)

    # keep only unique (and in sorted order) images
    merged_img_ids, idx = np.unique(merged_img_ids, return_index=True)
    merged_eval_imgs = merged_eval_imgs[..., idx]

    return merged_img_ids, merged_eval_imgs


def create_common_coco_eval(coco_eval, img_ids, eval_imgs):
    """
    Creates a common CocoEvaluator instance from the results of multiple processes.

    This function takes the CocoEvaluator instance and the results from multiple processes,
    merges them, and stores the merged results in the CocoEvaluator instance.

    Args:
        coco_eval (CocoEvaluator): The CocoEvaluator instance to store the merged results.
        img_ids (list): A list of image IDs from multiple processes.
        eval_imgs (list): A list of evaluation images from multiple processes.

    Returns:
        CocoEvaluator: The CocoEvaluator instance with the merged results.
    """
    img_ids, eval_imgs = merge(img_ids, eval_imgs)
    img_ids = list(img_ids)
    eval_imgs = list(eval_imgs.flatten())

    coco_eval.evalImgs = eval_imgs
    coco_eval.params.imgIds = img_ids
    coco_eval._paramsEval = copy.deepcopy(coco_eval.params)


def evaluate(imgs):
    """
    Evaluate the COCO evaluation results.

    Args:
        imgs (COCOeval): COCO evaluation object.

    Returns:
        tuple: A tuple containing:
            - img_ids (list): A list of unique image IDs.
            - eval_imgs (np.ndarray): A 3D array of evaluation images, where
              the first dimension is the image index, the second dimension is
              the area range index, and the third dimension is the image ID.
    """
    with redirect_stdout(io.StringIO()):
        imgs.evaluate()
    return imgs.params.imgIds, np.asarray(imgs.evalImgs).reshape(-1, len(imgs.params.areaRng), len(imgs.params.imgIds))
