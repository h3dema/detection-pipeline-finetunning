import albumentations as A
import numpy as np
import cv2

from albumentations.pytorch import ToTensorV2
from torchvision import transforms as transforms

def resize(im, img_size=640, square=False):
    """
    Resizes an image to a specified size, optionally forcing a square size.

    Args:
        im (numpy.ndarray): The image to resize.
        img_size (int, optional): The target size for the image's largest dimension. Defaults to 640.
        square (bool, optional): If True, resizes the image to a square with dimensions (img_size, img_size). Defaults to False, i.e., keeps aspect ratio.

    Returns:
        numpy.ndarray: The resized image.
    """

    if square:
        im = cv2.resize(im, (img_size, img_size))
    else:
        # Aspect ratio resize
        h0, w0 = im.shape[:2]  # orig hw
        r = img_size / max(h0, w0)  # ratio
        if r != 1:  # if sizes are not equal
            im = cv2.resize(im, (int(w0 * r), int(h0 * r)))
    return im

# Define the training tranforms
def get_train_aug():
    """Returns a list of train augmentations to be performed on the images and its bounding boxes.

    The augmentations are as follows:
    - Random blur of the image with probability 0.5.
    - Grayscaling of the image with probability 0.1.
    - Random brightness and contrast adjustment with probability 0.1.
    - Color jittering with probability 0.1.
    - Random gamma correction with probability 0.1.
    - Converting the image and its bounding boxes to a PyTorch tensor.

    The bounding boxes are expected to be in Pascal VOC format and the labels are expected to be in the 'labels' key of the bounding boxes dictionary.
    """
    return A.Compose([
        A.OneOf([
            A.Blur(blur_limit=3, p=0.5),
            A.MotionBlur(blur_limit=3, p=0.5),
            A.MedianBlur(blur_limit=3, p=0.5),
        ], p=0.5),
        A.ToGray(p=0.1),
        A.RandomBrightnessContrast(p=0.1),
        A.ColorJitter(p=0.1),
        A.RandomGamma(p=0.1),
        ToTensorV2(p=1.0),
    ], bbox_params=A.BboxParams(
        format='pascal_voc',
        label_fields=['labels'],
    ))

def get_train_transform():
    """
    Returns a list of train transforms to be performed on the images and its bounding boxes.

    The transforms are as follows:
    - Convert the image and its bounding boxes to a PyTorch tensor.

    The bounding boxes are expected to be in Pascal VOC format and the labels are expected to be in the 'labels' key of the bounding boxes dictionary.
    """
    return A.Compose([
        ToTensorV2(p=1.0),
    ], bbox_params=A.BboxParams(
        format='pascal_voc',
        label_fields=['labels'],
    ))

def transform_mosaic(mosaic, boxes, img_size=640):
    """
    Resizes the `mosaic` image to `img_size` which is the desired image size
    for the neural network input. Also transforms the `boxes` according to the
    `img_size`.

    :param mosaic: The mosaic image, Numpy array.
    :param boxes: Boxes Numpy.
    :param img_resize: Desired resize.
    """
    aug = A.Compose(
        [A.Resize(img_size, img_size, always_apply=True, p=1.0)
    ])
    sample = aug(image=mosaic)
    resized_mosaic = sample['image']
    transformed_boxes = (np.array(boxes) / mosaic.shape[0]) * resized_mosaic.shape[1]
    for box in transformed_boxes:
        # Bind all boxes to correct values. This should work correctly most of
        # of the time. There will be edge cases thought where this code will
        # mess things up. The best thing is to prepare the dataset as well as
        # as possible.
        if box[2] - box[0] <= 1.0:
            box[2] = box[2] + (1.0 - (box[2] - box[0]))
            if box[2] >= float(resized_mosaic.shape[1]):
                box[2] = float(resized_mosaic.shape[1])
        if box[3] - box[1] <= 1.0:
            box[3] = box[3] + (1.0 - (box[3] - box[1]))
            if box[3] >= float(resized_mosaic.shape[0]):
                box[3] = float(resized_mosaic.shape[0])
    return resized_mosaic, transformed_boxes

# Define the validation transforms
def get_valid_transform():
    """
    Returns a list of validation transforms to be performed on the images and its bounding boxes.

    The transforms are as follows:
    - Convert the image and its bounding boxes to a PyTorch tensor.

    The bounding boxes are expected to be in Pascal VOC format and the labels are expected to be in the 'labels' key of the bounding boxes dictionary.
    """
    return A.Compose([
        ToTensorV2(p=1.0),
    ], bbox_params=A.BboxParams(
        format='pascal_voc',
        label_fields=['labels'],
    ))

def infer_transforms(image):
    """
    Returns a torchvision transform to convert the image into a PyTorch tensor.

    The transforms are as follows:
    - Convert the image to a PIL image.
    - Convert the PIL image to a PyTorch tensor.

    The image is expected to be a numpy array of shape (height, width, channels).

    Returns:
        A transformed image as a PyTorch tensor.
    """
    # Define the torchvision image transforms.
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.ToTensor(),
    ])
    return transform(image)