import os
import glob as glob
import random
from xml.etree import ElementTree as et
import cv2
import numpy as np
import yaml
from tqdm.auto import tqdm

import torch
from torch.utils.data import Dataset, DataLoader

from utils.transforms import (
    get_train_transform,
    get_valid_transform,
    get_train_aug,
    transform_mosaic
)


# the dataset class
class CustomDataset(Dataset):

    def __init__(
        self,
        images_path,
        labels_path,
        img_size,
        classes,
        transforms=None,
        use_train_aug=False,
        train=False,
        mosaic=1.0,
        square_training=False
    ):
        """
        Initializes the CustomDataset object.

        Args:
            images_path (str): The path to the directory containing images.
            labels_path (str): The path to the directory containing label files.
            img_size (int): The size to which images will be resized.
            classes (list): A list of class names.
            transforms (callable, optional): A function/transform to apply to the images.
            use_train_aug (bool, optional): Whether to use training augmentations. Default is False.
            train (bool, optional): If True, the dataset will use training data. Default is False.
            mosaic (float, optional): Mosaic augmentation probability. Default is 1.0.
            square_training (bool, optional): If True, images will be square during training. Default is False.

        Attributes:
            transforms (callable): The transform to be applied to the images.
            use_train_aug (bool): Whether to use training augmentations.
            images_path (str): Path to the images directory.
            labels_path (str): Path to the labels directory.
            img_size (int): Image size for resizing.
            classes (list): List of class names.
            train (bool): Indicates if training data is used.
            square_training (bool): Indicates if images are square during training.
            mosaic_border (list): Border size for mosaic augmentation.
            image_file_types (list): Supported image file types.
            all_image_paths (list): List of all image paths.
            log_annot_issue_x (bool): Log annotation issue on x-axis.
            mosaic (float): Mosaic augmentation probability.
            log_annot_issue_y (bool): Log annotation issue on y-axis.
        """
        self.transforms = transforms
        self.use_train_aug = use_train_aug
        self.images_path = images_path
        self.labels_path = labels_path
        self.img_size = img_size
        self.classes = classes
        self.train = train
        self.square_training = square_training
        self.mosaic_border = [-img_size // 2, -img_size // 2]
        self.image_file_types = ['*.jpg', '*.jpeg', '*.png', '*.ppm', '*.JPG']
        self.all_image_paths = []
        self.log_annot_issue_x = True
        self.mosaic = mosaic
        self.log_annot_issue_y = True

        self.load_dataset()

    def load_dataset(self) -> None:
        """
        Loads the dataset into the dataset object.

        The dataset is loaded as follows:

        1. Get all the image paths in sorted order.
        2. Get all the annotation paths.
        3. Get all the images and sort them.
        4. Remove all annotations and images when no object is present.
        """

        # get all the image paths in sorted order
        for file_type in self.image_file_types:
            self.all_image_paths.extend(glob.glob(os.path.join(self.images_path, file_type)))
        self.all_annot_paths = glob.glob(os.path.join(self.labels_path, '*.xml'))
        self.all_images = [image_path.split(os.path.sep)[-1] for image_path in self.all_image_paths]
        self.all_images = sorted(self.all_images)
        # Remove all annotations and images when no object is present.
        self.read_and_clean()

    def read_and_clean(self):
        """
        Cleans the dataset by checking for missing or problematic annotations.

        This method iterates through all images and their corresponding annotation
        files to verify their existence and validity. If an image lacks a corresponding
            print('Checking Labels and images...')
        Steps:
        - Verify if an annotation file exists for each image.
        - Check for invalid bounding boxes in the annotation files.
        - Remove images and annotations that are missing or contain invalid data.
        - Print warnings for images with invalid bounding boxes.

        Modifies:
        - self.all_images: Removes entries for images with missing or invalid annotations.
        - self.all_annot_paths: Removes paths for annotations associated with problematic images.

        Prints:
        - Warnings for each image with missing annotations.
        - A summary of removed problematic images and annotations.
        """
        images_to_remove = []
        problematic_images = []

        for image_name in tqdm(self.all_images, total=len(self.all_images)):
            possible_xml_name = os.path.join(self.labels_path, os.path.splitext(image_name)[0] + '.xml')
            if possible_xml_name not in self.all_annot_paths:
                print(f"⚠️ {possible_xml_name} not found... Removing {image_name}")
                images_to_remove.append(image_name)
                continue

            # Check for invalid bounding boxes
            tree = et.parse(possible_xml_name)
            root = tree.getroot()
            invalid_bbox = False

            for member in root.findall('object'):
                xmin = float(member.find('bndbox').find('xmin').text)
                xmax = float(member.find('bndbox').find('xmax').text)
                ymin = float(member.find('bndbox').find('ymin').text)
                ymax = float(member.find('bndbox').find('ymax').text)

                if xmin >= xmax or ymin >= ymax:
                    invalid_bbox = True
                    break

            if invalid_bbox:
                problematic_images.append(image_name)
                images_to_remove.append(image_name)

        # Remove problematic images and their annotations
        self.all_images = [img for img in self.all_images if img not in images_to_remove]
        self.all_annot_paths = [
            path for path in self.all_annot_paths
            if not any(os.path.splitext(os.path.basename(path))[0] + ext in images_to_remove
                       for ext in self.image_file_types)
        ]

        # Print warnings for problematic images
        if problematic_images:
            print("\n⚠️ The following images have invalid bounding boxes and will be removed:")
            for img in problematic_images:
                print(f"⚠️ {img}")

        print(f"Removed {len(images_to_remove)} problematic images and annotations.")

    def resize(self, im, square=False):
        """
        Resizes an image to the specified size, optionally forcing a square size.

        Args:
            im (numpy.ndarray): The image to resize.
            square (bool, optional): If True, forces a square size. Defaults to False.

        Returns:
            numpy.ndarray: The resized image.
        """
        if square:
            im = cv2.resize(im, (self.img_size, self.img_size))
        else:
            h0, w0 = im.shape[:2]  # orig hw
            r = self.img_size / max(h0, w0)  # ratio
            if r != 1:  # if sizes are not equal
                im = cv2.resize(im, (int(w0 * r), int(h0 * r)))
        return im

    def load_image_and_labels(self, index):
        """
        Loads an image and its corresponding labels from a given index.

        Args:
            index (int): The index of the image to load.

        Returns:
            image (numpy.ndarray): The loaded image as a numpy array.
            image_resized (numpy.ndarray): The resized image with the same RGB values as the original image.
            orig_boxes (numpy.ndarray): The original bounding box coordinates of the labels.
            boxes (numpy.ndarray): The resized bounding box coordinates of the labels.
            labels (numpy.ndarray): The labels corresponding to the bounding boxes.
            area (int): The area of the resized image.
            iscrowd (bool): A boolean indicating whether the image is crowded.
            dims (tuple): A tuple containing the height and width of the resized image.
        """
        image_name = self.all_images[index]
        image_path = os.path.join(self.images_path, image_name)

        # Read the image.
        image = cv2.imread(image_path)
        # Convert BGR to RGB color format.
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)
        image_resized = self.resize(image, square=self.square_training)
        image_resized /= 255.0

        # Capture the corresponding XML file for getting the annotations.
        annot_filename = os.path.splitext(image_name)[0] + '.xml'
        annot_file_path = os.path.join(self.labels_path, annot_filename)

        boxes = []
        orig_boxes = []
        labels = []

        # Get the height and width of the image.
        image_width = image.shape[1]
        image_height = image.shape[0]

        # Box coordinates for xml files are extracted and corrected for image size given.
        # try:
        tree = et.parse(annot_file_path)
        root = tree.getroot()
        for member in root.findall('object'):
            # Map the current object name to `classes` list to get
            # the label index and append to `labels` list.
            labels.append(self.classes.index(member.find('name').text))

            # xmin = left corner x-coordinates
            xmin = float(member.find('bndbox').find('xmin').text)
            # xmax = right corner x-coordinates
            xmax = float(member.find('bndbox').find('xmax').text)
            # ymin = left corner y-coordinates
            ymin = float(member.find('bndbox').find('ymin').text)
            # ymax = right corner y-coordinates
            ymax = float(member.find('bndbox').find('ymax').text)

            xmin, ymin, xmax, ymax = self.check_image_and_annotation(
                xmin,
                ymin,
                xmax,
                ymax,
                image_width,
                image_height,
                orig_data=True
            )

            orig_boxes.append([xmin, ymin, xmax, ymax])

            # Resize the bounding boxes according to the
            # desired `width`, `height`.
            xmin_final = (xmin / image_width) * image_resized.shape[1]
            xmax_final = (xmax / image_width) * image_resized.shape[1]
            ymin_final = (ymin / image_height) * image_resized.shape[0]
            ymax_final = (ymax / image_height) * image_resized.shape[0]

            xmin_final, ymin_final, xmax_final, ymax_final = self.check_image_and_annotation(
                xmin_final,
                ymin_final,
                xmax_final,
                ymax_final,
                image_resized.shape[1],
                image_resized.shape[0],
                orig_data=False
            )

            boxes.append([xmin_final, ymin_final, xmax_final, ymax_final])
        # except:
        #     pass
        # Bounding box to tensor.
        boxes_length = len(boxes)
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        # Area of the bounding boxes.
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0]) if boxes_length > 0 else torch.as_tensor(boxes, dtype=torch.float32)
        # No crowd instances.
        iscrowd = torch.zeros((boxes.shape[0],), dtype=torch.int64) if boxes_length > 0 else torch.as_tensor(boxes, dtype=torch.float32)
        # Labels to tensor.
        labels = torch.as_tensor(labels, dtype=torch.int64)
        return image, image_resized, orig_boxes, \
            boxes, labels, area, iscrowd, (image_width, image_height)

    def check_image_and_annotation(
        self,
        xmin: int,
        ymin: int,
        xmax: int,
        ymax: int,
        width: int,
        height: int,
        orig_data=False
    ):
        """
        Validates and adjusts the bounding box coordinates to ensure they fall within
        the specified image dimensions.

        Check that all x_max and y_max are not more than the image
        width or height. If the bounding box is invalid, correct it
        by setting x_max/xmin or y_max/ymin to the edge of the image.


        Args:
            xmin (int): The minimum x-coordinate of the bounding box.
            ymin (int): The minimum y-coordinate of the bounding box.
            xmax (int): The maximum x-coordinate of the bounding box.
            ymax (int): The maximum y-coordinate of the bounding box.
            width (int): The width of the image.
            height (int): The height of the image.
            orig_data (bool): A flag indicating if the original data annotations
                            are being used.

        Returns:
            tuple: A tuple containing the adjusted (xmin, ymin, xmax, ymax) coordinates.
        """
        if ymax > height:
            ymax = height
        if xmax > width:
            xmax = width
        if xmax - xmin <= 1.0:
            if orig_data:
                # print(
                    # '\n',
                    # '!!! xmax is equal to xmin in data annotations !!!'
                    # 'Please check data'
                # )
                # print(
                    # 'Increasing xmax by 1 pixel to continue training for now...',
                    # 'THIS WILL ONLY BE LOGGED ONCE',
                    # '\n'
                # )
                self.log_annot_issue_x = False
            xmin = xmin - 1
        if ymax - ymin <= 1.0:
            if orig_data:
                # print(
                #     '\n',
                #     '!!! ymax is equal to ymin in data annotations !!!',
                #     'Please check data'
                # )
                # print(
                #     'Increasing ymax by 1 pixel to continue training for now...',
                #     'THIS WILL ONLY BE LOGGED ONCE',
                #     '\n'
                # )
                self.log_annot_issue_y = False
            ymin = ymin - 1
        return xmin, ymin, xmax, ymax

    def load_cutmix_image_and_boxes(self, index, resize_factor=512):
        """
        Generates a CutMix augmented image and corresponding bounding boxes.

        This function creates a mosaic of images by combining four randomly selected images from the dataset,
        including the one at the specified index. The resulting image is a combination of these four images,
        and the function adjusts the bounding boxes to fit the new mosaic image.
        Adapted from: https://www.kaggle.com/shonenkov/oof-evaluation-mixup-efficientdet

        Args:
            index (int): The index of the primary image to be included in the CutMix operation.
            resize_factor (int, optional): The desired size to which the final mosaic image will be resized. Default is 512.

        Returns:
            tuple: A tuple containing:
                - result_image (numpy.ndarray): The CutMix augmented image.
                - result_boxes (torch.Tensor): The bounding boxes for the objects in the augmented image.
                - final_classes (torch.Tensor): The class labels for each bounding box.
                - area: The area of the bounding boxes (not transformed).
                - iscrowd: A boolean indicating if the image contains crowd instances.
                - dims: The dimensions of the original image.
        """

        s = self.img_size
        yc, xc = (int(random.uniform(-x, 2 * s + x)) for x in self.mosaic_border)  # mosaic center x, y
        indices = [index] + [random.randint(0, len(self.all_images) - 1) for _ in range(3)]

        # Create empty image with the above resized image.
        # result_image = np.full((h, w, 3), 1, dtype=np.float32)
        result_boxes = []
        result_classes = []

        for i, index in enumerate(indices):
            _, image_resized, orig_boxes, boxes, \
                labels, area, iscrowd, dims = self.load_image_and_labels(
                    index=index
                )

            h, w = image_resized.shape[:2]

            if i == 0:
                # Create empty image with the above resized image.
                result_image = np.full((s * 2, s * 2, image_resized.shape[2]), 114 / 255, dtype=np.float32)  # base image with 4 tiles
                x1a, y1a, x2a, y2a = max(xc - w, 0), max(yc - h, 0), xc, yc  # xmin, ymin, xmax, ymax (large image)
                x1b, y1b, x2b, y2b = w - (x2a - x1a), h - (y2a - y1a), w, h  # xmin, ymin, xmax, ymax (small image)
            elif i == 1:  # top right
                x1a, y1a, x2a, y2a = xc, max(yc - h, 0), min(xc + w, s * 2), yc
                x1b, y1b, x2b, y2b = 0, h - (y2a - y1a), min(w, x2a - x1a), h
            elif i == 2:  # bottom left
                x1a, y1a, x2a, y2a = max(xc - w, 0), yc, xc, min(s * 2, yc + h)
                x1b, y1b, x2b, y2b = w - (x2a - x1a), 0, max(xc, w), min(y2a - y1a, h)
            elif i == 3:  # bottom right
                x1a, y1a, x2a, y2a = xc, yc, min(xc + w, s * 2), min(s * 2, yc + h)
                x1b, y1b, x2b, y2b = 0, 0, min(w, x2a - x1a), min(y2a - y1a, h)
            result_image[y1a:y2a, x1a:x2a] = image_resized[y1b:y2b, x1b:x2b]
            padw = x1a - x1b
            padh = y1a - y1b

            if len(orig_boxes) > 0:
                boxes[:, 0] += padw
                boxes[:, 1] += padh
                boxes[:, 2] += padw
                boxes[:, 3] += padh

                result_boxes.append(boxes)
                result_classes += labels

        final_classes = []
        if len(result_boxes) > 0:
            result_boxes = np.concatenate(result_boxes, 0)
            np.clip(result_boxes[:, 0:], 0, 2 * s, out=result_boxes[:, 0:])
            result_boxes = result_boxes.astype(np.int32)
            for idx in range(len(result_boxes)):
                if ((result_boxes[idx, 2] - result_boxes[idx, 0]) * (result_boxes[idx, 3] - result_boxes[idx, 1])) > 0:
                    final_classes.append(result_classes[idx])
            result_boxes = result_boxes[
                np.where((result_boxes[:, 2] - result_boxes[:, 0]) * (result_boxes[:, 3] - result_boxes[:, 1]) > 0)
            ]
        # Resize the mosaic image to the desired shape and transform boxes.
        result_image, result_boxes = transform_mosaic(
            result_image, result_boxes, self.img_size
        )
        return (
            result_image,
            torch.tensor(result_boxes),
            torch.tensor(np.array(final_classes)),
            area,
            iscrowd,
            dims
        )

    def __getitem__(self, idx):
        """
        This is a special method in Python classes, `__getitem__`,
        which is used to retrieve an item from a dataset.

        The method takes an index `idx` as input and
        returns a resized image and a dictionary `target` containing bounding box coordinates, class labels, area, and iscrowd annotations for that image.
        The method applies different data loading and augmentation strategies
        depending on whether the dataset is in training mode (`self.train`) or not.

        Args:
            idx (_type_): _description_

        Returns:
            image_resized (np.ndarray): The resized image.
            target (dict): A dictionary containing the bounding box coordinates, class labels, area, and iscrowd annotations.
        """
        if not self.train:  # No mosaic during validation.
            image, image_resized, orig_boxes, boxes, \
                labels, area, iscrowd, dims = self.load_image_and_labels(
                    index=idx
                )

        if self.train:
            mosaic_prob = random.uniform(0.0, 1.0)
            if self.mosaic >= mosaic_prob:
                image_resized, boxes, labels, \
                    area, iscrowd, dims = self.load_cutmix_image_and_boxes(
                        idx, resize_factor=(self.img_size, self.img_size)
                    )
            else:
                image, image_resized, orig_boxes, boxes, \
                    labels, area, iscrowd, dims = self.load_image_and_labels(
                        index=idx
                    )

        # Prepare the final `target` dictionary.
        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["area"] = area
        target["iscrowd"] = iscrowd
        image_id = torch.tensor([idx])
        target["image_id"] = image_id

        if self.use_train_aug:  # Use train augmentation if argument is passed.
            train_aug = get_train_aug()
            sample = train_aug(image=image_resized,
                               bboxes=target['boxes'],
                               labels=labels)
            image_resized = sample['image']
            target['boxes'] = torch.Tensor(sample['bboxes']).to(torch.int64)
        else:
            sample = self.transforms(image=image_resized,
                                     bboxes=target['boxes'],
                                     labels=labels)
            image_resized = sample['image']
            target['boxes'] = torch.Tensor(sample['bboxes']).to(torch.int64)

        # Fix to enable training without target bounding boxes,
        # see https://discuss.pytorch.org/t/fasterrcnn-images-with-no-objects-present-cause-an-error/117974/4
        if np.isnan((target['boxes']).numpy()).any() or target['boxes'].shape == torch.Size([0]):
            target['boxes'] = torch.zeros((0, 4), dtype=torch.int64)
        return image_resized, target

    def __len__(self):
        """
        Number of images

        Returns:
            Returns the number of images in the dataset.

        """
        return len(self.all_images)


def collate_fn(batch):
    """
    To handle the data loading as different images may have different number
    of objects and to handle varying size tensors as well.
    """
    return tuple(zip(*batch))


class DatasetHandler():
    """
    The `DatasetHandler` class is responsible for loading and preparing datasets for training and validation.
    It takes in a configuration file and image size as input and provides methods to create datasets and data loaders.

    """

    def __init__(self, args):
        """
        Constructor for the dataset handler.

        Args:
            args (dict): A dictionary of arguments. Must contain the following keys:
                'config' (str): The path to the yaml configuration file containing the data configurations.
                'imgsz' (int): The size of the images to be resized to.

        Attributes:
            data_configs (dict): The data configurations loaded from the yaml file.
            train_dir_images (str): The path to the training images.
            train_dir_labels (str): The path to the training labels.
            valid_dir_images (str): The path to the validation images.
            valid_dir_labels (str): The path to the validation labels.
            img_size (int): The size of the images to be resized to.
            classes (list): The list of class names.
        """

        # Load the data configurations
        with open(args['config']) as file:
            self.data_configs = yaml.safe_load(file)

        self.train_dir_images = os.path.normpath(self.data_configs['TRAIN_DIR_IMAGES'])
        self.train_dir_labels = os.path.normpath(self.data_configs['TRAIN_DIR_LABELS'])
        self.valid_dir_images = os.path.normpath(self.data_configs['VALID_DIR_IMAGES'])
        self.valid_dir_labels = os.path.normpath(self.data_configs['VALID_DIR_LABELS'])

        self.img_size = args['imgsz']
        self.classes = self.data_configs['CLASSES']

    @property
    def save_valid_prediction_images(self):
        """
        Property that indicates whether to save the predictions of the validation set.

        This property retrieves the 'SAVE_VALID_PREDICTION_IMAGES' setting from the data configurations.
        If the setting is not specified, it defaults to False.

        Returns:
            bool: True if validation prediction images should be saved, False otherwise.
        """
        return self.data_configs.get('SAVE_VALID_PREDICTION_IMAGES', False)

    @property
    def num_classes(self):
        """
        Property that returns the number of classes in the dataset.

        This property retrieves the 'NC' setting from the data configurations.

        Returns:
            int: The number of classes in the dataset.
        """
        return self.data_configs['NC']

    # Prepare the final datasets and data loaders.
    def create_train_dataset(self,
        use_train_aug=False,
        mosaic=1.0,
        square_training=False
    ) -> Dataset:
        """
        Creates a training dataset with optional augmentations and configurations.

        Args:
            use_train_aug (bool, optional): If True, apply training augmentations. Default is False.
            mosaic (float, optional): Probability of applying mosaic augmentation. Default is 1.0.
            square_training (bool, optional): If True, ensure images are square during training. Default is False.

        Returns:
            Dataset: The configured training dataset.
        """

        train_dataset = CustomDataset(
            self.train_dir_images,
            self.train_dir_labels,
            self.img_size,
            self.classes,
            get_train_transform(),
            use_train_aug=use_train_aug,
            train=True,
            mosaic=mosaic,
            square_training=square_training
        )
        return train_dataset

    def create_valid_dataset(self, square_training=False) -> Dataset:
        """
        Creates a validation dataset with optional square training configuration.

        Args:
            square_training (bool, optional): If True, ensure images are square during validation. Default is False.

        Returns:
            Dataset: The configured validation dataset.
        """
        valid_dataset = CustomDataset(
            self.valid_dir_images,
            self.valid_dir_labels,
            self.img_size,
            self.classes,
            get_valid_transform(),
            train=False,
            square_training=square_training
        )
        return valid_dataset

    def create_train_loader(self,
        train_dataset: Dataset,
        batch_size: int,
        num_workers=0,
        batch_sampler=None
    ) -> DataLoader:
        """
        Creates a DataLoader for the training dataset.

        Args:
            train_dataset (Dataset): The dataset to load training data from.
            batch_size (int): Number of samples per batch to load.
            num_workers (int, optional): Number of subprocesses to use for data loading. Defaults to 0.
            batch_sampler (Sampler or Iterable, optional): Defines the strategy to draw samples from the dataset. Defaults to None.

        Returns:
            DataLoader: A DataLoader object for the training dataset.
        """
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            # shuffle=True,
            num_workers=num_workers,
            collate_fn=collate_fn,
            sampler=batch_sampler
        )
        return train_loader

    def create_valid_loader(self,
        valid_dataset: Dataset,
        batch_size:int,
        num_workers=0, batch_sampler=None
    ) -> DataLoader:
        """
        Creates a DataLoader for the validation dataset.

        Args:
            valid_dataset (Dataset): The dataset to load validation data from.
            batch_size (int): Number of samples per batch to load.
            num_workers (int, optional): Number of subprocesses to use for data loading. Defaults to 0.
            batch_sampler (Sampler or Iterable, optional): Defines the strategy to draw samples from the dataset. Defaults to None.

        Returns:
            DataLoader: A DataLoader object for the validation dataset.
        """
        valid_loader = DataLoader(
            valid_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            collate_fn=collate_fn,
            sampler=batch_sampler
        )
        return valid_loader
