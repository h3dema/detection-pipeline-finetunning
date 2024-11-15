"""
The file provides utility functions and classes for working with the COCO dataset,
including filtering and remapping categories, converting polygon annotations to masks,
and converting datasets to the COCO API format.

"""

import copy
import os

import torch
import torch.utils.data
import torchvision
# import transforms as T
from pycocotools import mask as coco_mask
from pycocotools.coco import COCO


class FilterAndRemapCocoCategories:

    def __init__(self, categories, remap=True):
        """
        Initializes the FilterAndRemapCocoCategories object.

        Args:
            categories (list): List of category IDs to filter and remap.
            remap (bool, optional): If True, remap categories to a contiguous
                range starting from 0. Defaults to True.
        """
        self.categories = categories
        self.remap = remap

    def __call__(self, image, target):
        """
        Applies the category filter and remapping to the given image and target.

        Args:
            image (PIL.Image): The input image.
            target (dict): The target dictionary containing annotations.

        Returns:
            tuple: A tuple containing the filtered and remapped image and target.
        """
        anno = target["annotations"]
        anno = [obj for obj in anno if obj["category_id"] in self.categories]
        if not self.remap:
            target["annotations"] = anno
            return image, target
        anno = copy.deepcopy(anno)
        for obj in anno:
            obj["category_id"] = self.categories.index(obj["category_id"])
        target["annotations"] = anno
        return image, target


def convert_coco_poly_to_mask(segmentations, height, width):
    """
    Converts COCO polygon annotations to binary masks.

    Args:
        segmentations (list): A list of polygon segmentations, where each segmentation
            is a list of polygons for a single object in the image.
        height (int): The height of the image.
        width (int): The width of the image.

    Returns:
        torch.Tensor: A tensor of shape (N, height, width) where N is the number of
        segmentations, containing the binary masks for each segmentation.
        If no segmentations are provided, returns an empty tensor with shape (0, height, width).
    """
    masks = []
    for polygons in segmentations:
        rles = coco_mask.frPyObjects(polygons, height, width)
        mask = coco_mask.decode(rles)
        if len(mask.shape) < 3:
            mask = mask[..., None]
        mask = torch.as_tensor(mask, dtype=torch.uint8)
        mask = mask.any(dim=2)
        masks.append(mask)
    if masks:
        masks = torch.stack(masks, dim=0)
    else:
        masks = torch.zeros((0, height, width), dtype=torch.uint8)
    return masks


class ConvertCocoPolysToMask:

    def __call__(self, image, target):
        """
        Args:
            image: a PIL Image of size (H, W)
            target: a dict containing the following keys:
                image_id: int
                annotations: list of dicts, each representing an object with the
                    following keys:
                    bbox: list of 4 numbers representing the bounding box
                    category_id: int
                    area: int
                    iscrowd: int
                    keypoints: list of 17 keypoints, each represented as
                        (x, y, v) where v is 0 or 1
                    segmentation: list of polygons, each represented as a list
                        of (x, y) points

        Returns:
            image: a PIL Image of size (H, W)
            target: a dict containing the following keys:
                boxes: tensor of shape (N, 4) representing the bounding boxes
                labels: tensor of shape (N) representing the object class labels
                masks: tensor of shape (N, H, W) representing the object masks
                image_id: int
                keypoints: tensor of shape (N, 17, 3) representing the keypoints
        """
        w, h = image.size

        image_id = target["image_id"]
        image_id = torch.tensor([image_id])

        anno = target["annotations"]

        anno = [obj for obj in anno if obj["iscrowd"] == 0]

        boxes = [obj["bbox"] for obj in anno]
        # guard against no boxes via resizing
        boxes = torch.as_tensor(boxes, dtype=torch.float32).reshape(-1, 4)
        boxes[:, 2:] += boxes[:, :2]
        boxes[:, 0::2].clamp_(min=0, max=w)
        boxes[:, 1::2].clamp_(min=0, max=h)

        classes = [obj["category_id"] for obj in anno]
        classes = torch.tensor(classes, dtype=torch.int64)

        segmentations = [obj["segmentation"] for obj in anno]
        masks = convert_coco_poly_to_mask(segmentations, h, w)

        keypoints = None
        if anno and "keypoints" in anno[0]:
            keypoints = [obj["keypoints"] for obj in anno]
            keypoints = torch.as_tensor(keypoints, dtype=torch.float32)
            num_keypoints = keypoints.shape[0]
            if num_keypoints:
                keypoints = keypoints.view(num_keypoints, -1, 3)

        keep = (boxes[:, 3] > boxes[:, 1]) & (boxes[:, 2] > boxes[:, 0])
        boxes = boxes[keep]
        classes = classes[keep]
        masks = masks[keep]
        if keypoints is not None:
            keypoints = keypoints[keep]

        target = {}
        target["boxes"] = boxes
        target["labels"] = classes
        target["masks"] = masks
        target["image_id"] = image_id
        if keypoints is not None:
            target["keypoints"] = keypoints

        # for conversion to coco api
        area = torch.tensor([obj["area"] for obj in anno])
        iscrowd = torch.tensor([obj["iscrowd"] for obj in anno])
        target["area"] = area
        target["iscrowd"] = iscrowd

        return image, target


def _coco_remove_images_without_annotations(dataset, cat_list=None):
    """
    Purpose: Filter out images without any annotations in the given categories.
    """

    def _has_only_empty_bbox(anno):
        """
        Checks if annotation has only empty bboxes

        Args:
            anno: annotation

        Returns:
            bool: True if all bboxes in annotation are empty
        """
        return all(any(o <= 1 for o in obj["bbox"][2:]) for obj in anno)

    def _count_visible_keypoints(anno):
        """Count the number of visible keypoints in the given annotation."""

        return sum(sum(1 for v in ann["keypoints"][2::3] if v > 0) for ann in anno)

    min_keypoints_per_image = 10

    def _has_valid_annotation(anno):
        # if it's empty, there is no annotation
        """
        Filter out images without any annotations in the given categories.

        Parameters:
        ----------
        anno : list[dict]
            annotation for an image

        Returns:
        -------
        bool
            whether the image has valid annotations
        """
        if len(anno) == 0:
            return False
        # if all boxes have close to zero area, there is no annotation
        if _has_only_empty_bbox(anno):
            return False
        # keypoints task have a slight different critera for considering
        # if an annotation is valid
        if "keypoints" not in anno[0]:
            return True
        # for keypoint detection tasks, only consider valid images those
        # containing at least min_keypoints_per_image
        if _count_visible_keypoints(anno) >= min_keypoints_per_image:
            return True
        return False

    assert isinstance(dataset, torchvision.datasets.CocoDetection)
    ids = []
    for ds_idx, img_id in enumerate(dataset.ids):
        ann_ids = dataset.coco.getAnnIds(imgIds=img_id, iscrowd=None)
        anno = dataset.coco.loadAnns(ann_ids)
        if cat_list:
            anno = [obj for obj in anno if obj["category_id"] in cat_list]
        if _has_valid_annotation(anno):
            ids.append(ds_idx)

    dataset = torch.utils.data.Subset(dataset, ids)
    return dataset


def convert_to_coco_api(ds):
    """
    Converts a given dataset to the COCO API format.

    Args:
        ds: The dataset to be converted, expected to yield a tuple of
            (image, targets) where targets is a dictionary containing
            the annotations.

    Returns:
        coco_ds: A COCO object with the dataset converted to COCO format,
            including images, categories, and annotations.

    This function iterates through each image in the dataset, extracting
    annotation details such as bounding boxes, labels, areas, and optionally
    masks and keypoints. It then constructs the necessary COCO data structures
    for images, categories, and annotations. The function also ensures that
    annotation IDs start from 1, complying with COCO format requirements.
    """
    coco_ds = COCO()
    # annotation IDs need to start at 1, not 0, see torchvision issue #1530
    ann_id = 1
    dataset = {"images": [], "categories": [], "annotations": []}
    categories = set()
    for img_idx in range(len(ds)):
        # find better way to get target
        # targets = ds.get_annotations(img_idx)
        img, targets = ds[img_idx]
        image_id = targets["image_id"].item()
        img_dict = {}
        img_dict["id"] = image_id
        img_dict["height"] = img.shape[-2]
        img_dict["width"] = img.shape[-1]
        dataset["images"].append(img_dict)
        bboxes = targets["boxes"]
        if len(bboxes) > 0:
            bboxes[:, 2:] -= bboxes[:, :2]
        bboxes = bboxes.tolist()
        labels = targets["labels"].tolist()
        areas = targets["area"].tolist()
        iscrowd = targets["iscrowd"].tolist()
        if "masks" in targets:
            masks = targets["masks"]
            # make masks Fortran contiguous for coco_mask
            masks = masks.permute(0, 2, 1).contiguous().permute(0, 2, 1)
        if "keypoints" in targets:
            keypoints = targets["keypoints"]
            keypoints = keypoints.reshape(keypoints.shape[0], -1).tolist()
        num_objs = len(bboxes)
        for i in range(num_objs):
            ann = {}
            ann["image_id"] = image_id
            ann["bbox"] = bboxes[i]
            ann["category_id"] = labels[i]
            categories.add(labels[i])
            ann["area"] = areas[i]
            ann["iscrowd"] = iscrowd[i]
            ann["id"] = ann_id
            if "masks" in targets:
                ann["segmentation"] = coco_mask.encode(masks[i].numpy())
            if "keypoints" in targets:
                ann["keypoints"] = keypoints[i]
                ann["num_keypoints"] = sum(k != 0 for k in keypoints[i][2::3])
            dataset["annotations"].append(ann)
            ann_id += 1
    dataset["categories"] = [{"id": i} for i in sorted(categories)]
    coco_ds.dataset = dataset
    coco_ds.createIndex()
    return coco_ds


def get_coco_api_from_dataset(dataset):
    """
    Gets a COCO API object from a dataset, whether it's a torchvision
    CocoDetection dataset or a custom dataset that uses the same format.

    Args:
        dataset: A torchvision CocoDetection dataset or a custom dataset that uses the same format.

    Returns:
        A COCO API object.

    Note that if the given dataset is a torchvision CocoDetection dataset,
    this function will return the COCO API object that is used internally by
    the dataset. If the given dataset is a custom dataset, this function will
    attempt to convert it to a COCO API object by calling the
    convert_to_coco_api function. The conversion is done by iterating through
    the dataset and collecting the necessary information such as images,
    categories, and annotations.
    """
    for _ in range(10):
        if isinstance(dataset, torchvision.datasets.CocoDetection):
            break
        if isinstance(dataset, torch.utils.data.Subset):
            dataset = dataset.dataset
    if isinstance(dataset, torchvision.datasets.CocoDetection):
        return dataset.coco
    return convert_to_coco_api(dataset)

class CocoDetection(torchvision.datasets.CocoDetection):
    """
    This is custom dataset class that inherits from PyTorch's CocoDetection class,
    allowing for additional transformations to be applied to the images and annotations.
    """

    def __init__(self, img_folder, ann_file, transforms):
        """
        Initializes the CocoDetection dataset with the provided image folder, annotation file, and transformations.

        Args:
            img_folder (str): The directory containing the images.
            ann_file (str): The path to the COCO annotations file.
            transforms (callable, optional): A function/transform to apply to the images and annotations.
        """
        super().__init__(img_folder, ann_file)
        self._transforms = transforms

    def __getitem__(self, idx):
        """
        Returns:
            tuple: A tuple containing the image and target. The image is a PIL Image of size (H, W),
            and the target is a dictionary containing the following keys:
                - image_id (int): The image ID.
                - annotations (list): A list of dictionaries, each representing an object with the
                    following keys:
                    - id (int): The annotation ID.
                    - image_id (int): The image ID.
                    - category_id (int): The category ID.
                    - segmentation (RLE or polygon): The object segmentation.
                    - area (float): The object area.
                    - bbox (list): The bounding box coordinates in (x, y, w, h) format.
                    - iscrowd (bool): Whether the object is crowded.
                    - keypoints (list): The keypoints.
        """
        img, target = super().__getitem__(idx)
        image_id = self.ids[idx]
        target = dict(image_id=image_id, annotations=target)
        if self._transforms is not None:
            img, target = self._transforms(img, target)
        return img, target


def get_coco(root, image_set, transforms, mode="instances"):
    """
    Load a COCO dataset from a given root directory.

    Args:
        root (str): The root directory containing the COCO dataset.
        image_set (str): The image set to load, can be "train" or "val".
        transforms (callable, optional): A function/transform to apply to the images and annotations.
        mode (str, optional): The COCO mode to load, can be "instances" or "person_keypoints". Defaults to "instances".

    Returns:
        dataset (torch.utils.data.Dataset): The loaded COCO dataset.

    """
    anno_file_template = "{}_{}2017.json"
    PATHS = {
        "train": ("train2017", os.path.join("annotations", anno_file_template.format(mode, "train"))),
        "val": ("val2017", os.path.join("annotations", anno_file_template.format(mode, "val"))),
        # "train": ("val2017", os.path.join("annotations", anno_file_template.format(mode, "val")))
    }

    t = [ConvertCocoPolysToMask()]

    if transforms is not None:
        t.append(transforms)
    transforms = T.Compose(t)

    img_folder, ann_file = PATHS[image_set]
    img_folder = os.path.join(root, img_folder)
    ann_file = os.path.join(root, ann_file)

    dataset = CocoDetection(img_folder, ann_file, transforms=transforms)

    if image_set == "train":
        dataset = _coco_remove_images_without_annotations(dataset)

    # dataset = torch.utils.data.Subset(dataset, [i for i in range(500)])

    return dataset


def get_coco_kp(root, image_set, transforms):
    """
    Load a COCO dataset with person keypoints from a given root directory.

    Args:
        root (str): The root directory containing the COCO dataset.
        image_set (str): The image set to load, can be "train" or "val".
        transforms (callable, optional): A function/transform to apply to the images and annotations.

    Returns:
        dataset (torch.utils.data.Dataset): The loaded COCO dataset with person keypoints.
    """
    return get_coco(root, image_set, transforms, mode="person_keypoints")