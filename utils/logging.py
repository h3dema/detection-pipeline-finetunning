import logging
import os
import pandas as pd
import wandb
import cv2
import numpy as np
import json

from torch.utils.tensorboard.writer import SummaryWriter


logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


# Initialize Weights and Biases.
def wandb_init(name):
    """
    Initializes Weights and Biases for logging.

    Args:
        name (str): The name of the Weights and Biases run.

    Returns:
        None
    """

    wandb.init(name=name)

def set_log(log_dir):
    """
    Configures the logging settings for the application.

    This function sets up a basic configuration for logging, directing log
    messages to a file located at the specified log directory. It also adds
    a console handler to the logger, allowing log messages to be output to
    the console as well.

    :param log_dir: The directory path where the log file 'train.log'
                    will be created.
    """
    logging.basicConfig(
        # level=logging.DEBUG,
        format='%(message)s',
        # datefmt='%a, %d %b %Y %H:%M:%S',
        filename=f"{log_dir}/train.log",
        filemode='w'
    )
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    # add the handler to the root logger
    logging.getLogger().addHandler(console)


def log(content, *args):
    """
    Logs the given content, and any additional arguments provided, to the
    logger at the INFO level.

    The content and any additional arguments are concatenated together using
    the str() function, and the resulting string is passed to the logger's
    info() method.

    :param content: The main content to log.
    :type content: str
    :param args: Additional arguments to log.
    :type args: tuple
    """
    for arg in args:
        content += str(arg)
    logger.info(content)

def coco_log(log_dir, stats):
    """
    Logs the given COCO evaluation stats to the given log directory.

    The provided stats are expected to be an array of numbers, where each
    number corresponds to the value of the COCO evaluation metric at the
    same index.

    The following COCO evaluation metrics are supported:

        Average Precision (AP) @ IoU=0.50:0.95 | area=   all | maxDets=100
        Average Precision (AP) @ IoU=0.50      | area=   all | maxDets=100
        Average Precision (AP) @ IoU=0.75      | area=   all | maxDets=100
        Average Precision (AP) @ IoU=0.50:0.95 | area= small | maxDets=100
        Average Precision (AP) @ IoU=0.50:0.95 | area=medium | maxDets=100
        Average Precision (AP) @ IoU=0.50:0.95 | area= large | maxDets=100
        Average Recall (AR) @ IoU=0.50:0.95 | area=   all | maxDets=  1
        Average Recall (AR) @ IoU=0.50:0.95 | area=   all | maxDets= 10
        Average Recall (AR) @ IoU=0.50:0.95 | area=   all | maxDets=100
        Average Recall (AR) @ IoU=0.50:0.95 | area= small | maxDets=100
        Average Recall (AR) @ IoU=0.50:0.95 | area=medium | maxDets=100
        Average Recall (AR) @ IoU=0.50:0.95 | area= large | maxDets=100

    :param log_dir: The directory path where the log file 'train.log'
                    will be created.
    :type log_dir: str
    :param stats: The array of numbers representing the COCO evaluation
                  metrics to be logged.
    :type stats: list
    """
    log_dict_keys = [
        'Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ]',
        'Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ]',
        'Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ]',
        'Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ]',
        'Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ]',
        'Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ]',
        'Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ]',
        'Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ]',
        'Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ]',
        'Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ]',
        'Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ]',
        'Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ]',
    ]
    log_dict = {}
    # for i, key in enumerate(log_dict_keys):
    #     log_dict[key] = stats[i]

    with open(f"{log_dir}/train.log", 'a+') as f:
        f.writelines('\n')
        for i, key in enumerate(log_dict_keys):
            out_str = f"{key} = {stats[i]}"
            logger.debug(out_str) # DEBUG model so as not to print on console.
        logger.debug('\n'*2) # DEBUG model so as not to print on console.
    # f.close()


def set_summary_writer(log_dir):
    """
    Set up a tensorboard SummaryWriter instance with the given log_dir.

    :param log_dir: The directory path where the tensorboard log files
                    will be created.
    :type log_dir: str
    :return: The SummaryWriter instance.
    :rtype: tensorboardX.SummaryWriter
    """
    writer = SummaryWriter(log_dir=log_dir)
    return writer


def tensorboard_loss_log(name, loss_np_arr, writer, epoch):
    """
    To plot graphs for TensorBoard log. The save directory for this
    is the same as the training result save directory.
    """
    writer.add_scalar(name, loss_np_arr[-1], epoch)


def tensorboard_map_log(name, val_map_05, val_map, writer, epoch):
    """
    To plot graphs for TensorBoard log. The save directory for this
    is the same as the training result save directory.

    :param name: The name of the graph to be plotted.
    :type name: str
    :param val_map_05: The list of validation mAP@0.5 values.
    :type val_map_05: list
    :param val_map: The list of validation mAP@0.5:0.95 values.
    :type val_map: list
    :param writer: The SummaryWriter instance.
    :type writer: tensorboardX.SummaryWriter
    :param epoch: The current training epoch.
    :type epoch: int
    """
    writer.add_scalars(
        name,
        {
            'mAP@0.5': val_map_05[-1],
            'mAP@0.5_0.95': val_map[-1]
        },
        epoch
    )


def create_log_csv(log_dir):
    """
    Creates a CSV file in the specified log directory with the columns
    needed for saving the training and validation results.

    :param log_dir: The directory where the CSV file is to be saved.
    :type log_dir: str
    """
    cols = [
        'epoch',
        'map',
        'map_05',
        'train loss',
        'train cls loss',
        'train box reg loss',
        'train obj loss',
        'train rpn loss'
    ]
    results_csv = pd.DataFrame(columns=cols)
    results_csv.to_csv(os.path.join(log_dir, 'results.csv'), index=False)

def csv_log(
    log_dir,
    stats,
    epoch,
    train_loss_list,
    loss_cls_list,
    loss_box_reg_list,
    loss_objectness_list,
    loss_rpn_list
):
    """
    Logs the current epoch's stats to a CSV file.

    :param log_dir: The directory where the CSV file is to be saved.
    :type log_dir: str
    :param stats: List containing the validation mAP@0.5 and mAP@0.5:0.95.
    :type stats: List[float]
    :param epoch: The current epoch number.
    :type epoch: int
    :param train_loss_list: List containing the loss values for the current epoch's batches.
    :type train_loss_list: List[float]
    :param loss_cls_list: List containing the cls loss values for the current epoch's batches.
    :type loss_cls_list: List[float]
    :param loss_box_reg_list: List containing the box reg loss values for the current epoch's batches.
    :type loss_box_reg_list: List[float]
    :param loss_objectness_list: List containing the objectness loss values for the current epoch's batches.
    :type loss_objectness_list: List[float]
    :param loss_rpn_list: List containing the rpn loss values for the current epoch's batches.
    :type loss_rpn_list: List[float]
    """
    if epoch+1 == 1:
        create_log_csv(log_dir)

    df = pd.DataFrame(
        {
            'epoch': int(epoch+1),
            'map_05': [float(stats[0])],
            'map': [float(stats[1])],
            'train loss': train_loss_list[-1],
            'train cls loss': loss_cls_list[-1],
            'train box reg loss': loss_box_reg_list[-1],
            'train obj loss': loss_objectness_list[-1],
            'train rpn loss': loss_rpn_list[-1]
        }
    )
    df.to_csv(
        os.path.join(log_dir, 'results.csv'),
        mode='a',
        index=False,
        header=False
    )


def overlay_on_canvas(bg, image):
    """
    Overlay an image on a canvas (background) at the center of the canvas.

    Parameters
    ----------
    bg : numpy.ndarray
        The background (canvas) image.
    image : numpy.ndarray
        The image to overlay on the canvas.

    Returns
    -------
    numpy.ndarray
        The overlayed image.
    """
    bg_copy = bg.copy()
    h, w = bg.shape[:2]
    h1, w1 = image.shape[:2]
    # Center of canvas (background).
    cx, cy = (h - h1) // 2, (w - w1) // 2
    bg_copy[cy:cy + h1, cx:cx + w1] = image
    return bg_copy * 255.


def wandb_log(
    epoch_loss,
    loss_list_batch,
    loss_cls_list,
    loss_box_reg_list,
    loss_objectness_list,
    loss_rpn_list,
    loss_bbox_ctrness,
    val_map_05,
    val_map,
    val_pred_image,
    image_size
):
    """
    Logs the following to Weights and Biases:

    - Per-iteration training loss.
    - Per-epoch training loss.
    - Validation mAP_0.5:0.95.
    - Validation mAP_0.5.
    - Validation predictions.

    Parameters
    ----------
    epoch_loss : float
        Single loss value for the current epoch.
    loss_list_batch : list of float
        List containing loss values for the current epoch's loss value for each batch.
    loss_cls_list : list of float
        A list of classification losses for each iteration in the current epoch.
    loss_box_reg_list : list of float
        A list of bounding box regression losses for each iteration in the current epoch.
    loss_objectness_list : list of float
        A list of objectness losses for each iteration in the current epoch.
    loss_rpn_list : list of float
        A list of RPN losses for each iteration in the current epoch.
    loss_bbox_ctrness : list of float
        A list of bbox center-ness losses for each iteration in the current epoch.
    val_map_05 : float
        The validation mAP_0.5.
    val_map : float
        The validation mAP_0.5:0.95.
    val_pred_image : list of numpy.ndarray
        A list of prediction images for validation.
    image_size : int
        The size of the images.
    """

    # WandB logging.
    for i in range(len(loss_list_batch)):
        wandb.log(
            {'train_loss_iter': loss_list_batch[i],},
        )
    # for i in range(len(loss_cls_list)):
    wandb.log(
        {
            'train_loss_cls': loss_cls_list[-1],
            'train_loss_box_reg': loss_box_reg_list[-1],
            'train_loss_obj': loss_objectness_list[-1],
            'train_loss_rpn': loss_rpn_list[-1],
            'train_loss_bbox_ctrness': loss_bbox_ctrness[-1],
        }
    )
    wandb.log(
        {
            'train_loss_epoch': epoch_loss
        },
    )
    wandb.log(
        {'val_map_05_95': val_map}
    )
    wandb.log(
        {'val_map_05': val_map_05}
    )

    bg = np.full((image_size * 2, image_size * 2, 3), 114, dtype=np.float32)

    if len(val_pred_image) == 1:
        log_image = overlay_on_canvas(bg, val_pred_image[0])
        wandb.log({'predictions': [wandb.Image(log_image)]})

    if len(val_pred_image) == 2:
        log_image = cv2.hconcat(
            [
                overlay_on_canvas(bg, val_pred_image[0]),
                overlay_on_canvas(bg, val_pred_image[1])
            ]
        )
        wandb.log({'predictions': [wandb.Image(log_image)]})

    if len(val_pred_image) > 2 and len(val_pred_image) <= 8:
        log_image = overlay_on_canvas(bg, val_pred_image[0])
        for i in range(len(val_pred_image)-1):
            log_image = cv2.hconcat([
                log_image,
                overlay_on_canvas(bg, val_pred_image[i+1])
            ])
        wandb.log({'predictions': [wandb.Image(log_image)]})

    if len(val_pred_image) > 8:
        log_image = overlay_on_canvas(bg, val_pred_image[0])
        for i in range(len(val_pred_image)-1):
            if i == 7:
                break
            log_image = cv2.hconcat([
                log_image,
                overlay_on_canvas(bg, val_pred_image[i-1])
            ])
        wandb.log({'predictions': [wandb.Image(log_image)]})


def wandb_save_model(model_dir):
    """
    Uploads the models to Weights&Biases.

    :param model_dir: Local disk path where models are saved.
    """
    wandb.save(os.path.join(model_dir, 'best_model.pth'))


class LogJSON():
    """
    The LogJSON class is designed to manage a JSON file that stores image and annotation data in the COCO format.

    Note that the update method assumes that the input data is in a specific format,
    with boxes and labels being arrays of bounding box coordinates and class labels, respectively.
    The classes parameter is a dictionary mapping class IDs to class names.
    """
    def __init__(self, output_filename):
        """
        :param output_filename: Path where the JSOn file should be saved.
        """
        if not os.path.exists(output_filename):
        # Initialize file with basic structure if it doesn't exist
            with open(output_filename, 'w') as file:
                json.dump({"images": [], "annotations": [], "categories": []}, file, indent=4)

        with open(output_filename, 'r') as file:
            self.coco_data = json.load(file)

        self.annotations = self.coco_data['annotations']
        self.images = self.coco_data['images']
        self.categories = set(cat['id'] for cat in self.coco_data['categories'])
        self.annotation_id = max([ann['id'] for ann in self.annotations], default=0) + 1
        self.image_id = len(self.images) + 1

    def update(self, image, file_name, boxes, labels, classes):
        """
        Update the log file metrics with the current image or current frame information.

        :param image: The original image/frame.
        :param file_name: image file name.
        :param output: Model outputs.
        :param classes: classes in the model.
        """
        image_info = {
            "file_name": file_name, "width": image.shape[1], "height": image.shape[0]
        }

        # Add image entry
        self.images.append({
            "id": self.image_id,
            "file_name": image_info['file_name'],
            "width": image_info['width'],
            "height": image_info['height']
        })

        boxes = np.array(boxes, dtype=np.float64)
        labels = np.array(labels, dtype=np.float64)

        for box, label in zip(boxes, labels):
            xmin, ymin, xmax, ymax = box
            width = xmax - xmin
            height = ymax - ymin

            annotation = {
                "id": self.annotation_id,
                "image_id": self.image_id,
                "bbox": [xmin, ymin, width, height],
                "area": width * height,
                "category_id": label,
                "iscrowd": 0
            }
            self.annotations.append(annotation)
            self.annotation_id += 1
            self.categories.add(int(label))

        # Update categories
        self.coco_data['categories'] = [{"id": cat_id, "name": classes[cat_id]} for cat_id in self.categories]

    def save(self, output_filename):
        """
        :param output_filename: Path where the JSOn file should be saved.
        """
        with open(output_filename, 'w') as file:
            json.dump(self.coco_data, file, indent=4)
