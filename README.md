# A Simple Pipeline to Train PyTorch Object Detection Models



Train PyTorch models easily on any custom dataset. Choose between official PyTorch models trained on COCO dataset, or choose any backbone from Torchvision classification models, or even write your own custom backbones.

***You can run a Faster RCNN model with Mini Darknet backbone and Mini Detection Head at more than 150 FPS on an RTX 3080***.

![](readme_images/gif_1.gif)

## Get Started

​																								[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1oFxPpBeE8SzSQq7BTUv28IIqQeiHHLdj?usp=sharing) [![Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](https://www.kaggle.com/code/sovitrath/custom-faster-rcnn-training-kaggle/notebook)

## Go To

* [Model naming conventions](#Custom-Model-Naming-Conventions)
* [Setup on Ubuntu](#Setup-for-Ubuntu)
* [Setup on Windows](#Setup-on-Windows)
* [Train on Custom Dataset](#Train-on-Custom-Dataset)
* [Check all available arguments for training](#Check-all-available-arguments-for-training)
* [Distributed training](#Distributed-training)
* [Inference](#Inference)
* [Evaluation](#Evaluation)
* [Available Models](#A-List-of-All-Model-Flags-to-Use-With-the-Training-Script)


## Custom Model Naming Conventions

For this repository, we consider the following representation size in the **Faster RCNN** head and predictor:

| Head size      | Size |
|----------------|------|
| **Small head** |  512 |
| **Tiny head**  |  256 |
| **Nano head**  |  128 |



## Setup on Ubuntu

1. Clone the repository.

   ```bash
   git clone https://github.com/sovit-123/fastercnn-pytorch-training-pipeline.git
   ```

2. Install requirements.

   1. **Method 1**: If you have CUDA and cuDNN set up already, do this in your environment of choice.

      ```bash
      pip install -r requirements.txt
      ```

   2. **Method 2**: If you want to install PyTorch with CUDA Toolkit in your environment of choice.

      ```bash
      conda install pytorch torchvision torchaudio cudatoolkit=10.2 -c pytorch
      ```

      OR

      ```bash
      conda install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch
      ```

      OR install the version with CUDA support as per your choice from **[here](https://pytorch.org/get-started/locally/)**.

      Then install the remaining **[requirements](https://github.com/sovit-123/fasterrcnn-pytorch-training-pipeline/blob/main/requirements.txt)**.





## Setup on Windows

1. **First you need to install Microsoft Visual Studio from [here](https://my.visualstudio.com/Downloads?q=Visual%20Studio%202017)**. Sing In/Sing Up by clicking on **[this link](https://my.visualstudio.com/Downloads?q=Visual%20Studio%202017)** and download the **Visual Studio Community 2017** edition.

   ![](readme_images/vs-2017-annotated.jpg)

   Install with all the default chosen settings. It should be around 6 GB. Mainly, we need the C++ Build Tools.

2. Then install the proper **`pycocotools`** for Windows.

   ```bash
   pip install git+https://github.com/gautamchitnis/cocoapi.git@cocodataset-master#subdirectory=PythonAPI
   ```

3. Clone the repository.

   ```bash
   git clone https://github.com/sovit-123/fastercnn-pytorch-training-pipeline.git
   ```

4. Install PyTorch with CUDA support.

   ```bash
   conda install pytorch torchvision torchaudio cudatoolkit=10.2 -c pytorch
   ```

   OR

   ```bash
   conda install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch
   ```

   OR install the version with CUDA support as per your choice from **[here](https://pytorch.org/get-started/locally/)**.

   Then install the remaining **[requirements](https://github.com/sovit-123/pytorch-efficientdet-api/blob/main/requirements.txt)** except for `pycocotools`.






## Train on Custom Dataset

Taking an exmaple of the [smoke dataset](https://www.kaggle.com/didiruh/smoke-pascal-voc) from Kaggle. Let's say that the dataset is in the `data/smoke_pascal_voc` directory in the following format. And the `smoke.yaml` is in the `configs` directory. Assuming, we store the smoke data in the `data` directory

```bash
├── data
│   ├── smoke_pascal_voc
│   │   ├── archive
│   │   │   ├── train
│   │   │   └── valid
│   └── README.md
├── configs
│   └── smoke.yaml
├── models
│   ├── create_fasterrcnn_model.py
│   ...
│   └── __init__.py
├── outputs
│   ├── inference
│   └── training
│       ...
├── readme_images
│   ...
├── torch_utils
│   ├── coco_eval.py
│   ...
├── utils
│   ├── annotations.py
│   ...
├── datasets.py
├── inference.py
├── inference_video.py
├── __init__.py
├── README.md
├── requirements.txt
└── train.py
```

The content of the `smoke.yaml` should be the following:

```yaml
# Images and labels direcotry should be relative to train.py
TRAIN_DIR_IMAGES: ../../xml_od_data/smoke_pascal_voc/archive/train/images
TRAIN_DIR_LABELS: ../../xml_od_data/smoke_pascal_voc/archive/train/annotations
# VALID_DIR should be relative to train.py
VALID_DIR_IMAGES: ../../xml_od_data/smoke_pascal_voc/archive/valid/images
VALID_DIR_LABELS: ../../xml_od_data/smoke_pascal_voc/archive/valid/annotations

# Class names.
CLASSES: [
    '__background__',
    'smoke'
]

# Number of classes (object classes + 1 for background class in Faster RCNN).
NC: 2

# Whether to save the predictions of the validation set while training.
SAVE_VALID_PREDICTION_IMAGES: True
```

***Note that*** *the data and annotations can be in the same directory as well. In that case, the TRAIN_DIR_IMAGES and TRAIN_DIR_LABELS will save the same path. Similarly for VALID images and labels. The `datasets.py` will take care of that*.

Next, to start the training, you can use the following command.

**Command format:**

```bash
python train.py --config <path to the data config YAML file> --epochs 100 --model <model name (defaults to fasterrcnn_resnet50)> --name <folder name inside output/training/> --batch 16
```

In this case, the exact command would be:

```bash
python train.py --config configs/smoke.yaml --epochs 100 --model fasterrcnn_resnet50_fpn --name smoke_training --batch 16
```


The terminal output should be similar to the following:

```
Number of training samples: 665
Number of validation samples: 72

3,191,405 total parameters.
3,191,405 training parameters.
Epoch     0: adjusting learning rate of group 0 to 1.0000e-03.
Epoch: [0]  [ 0/84]  eta: 0:02:17  lr: 0.000013  loss: 1.6518 (1.6518)  time: 1.6422  data: 0.2176  max mem: 1525
Epoch: [0]  [83/84]  eta: 0:00:00  lr: 0.001000  loss: 1.6540 (1.8020)  time: 0.0769  data: 0.0077  max mem: 1548
Epoch: [0] Total time: 0:00:08 (0.0984 s / it)
creating index...
index created!
Test:  [0/9]  eta: 0:00:02  model_time: 0.0928 (0.0928)  evaluator_time: 0.0245 (0.0245)  time: 0.2972  data: 0.1534  max mem: 1548
Test:  [8/9]  eta: 0:00:00  model_time: 0.0318 (0.0933)  evaluator_time: 0.0237 (0.0238)  time: 0.1652  data: 0.0239  max mem: 1548
Test: Total time: 0:00:01 (0.1691 s / it)
Averaged stats: model_time: 0.0318 (0.0933)  evaluator_time: 0.0237 (0.0238)
Accumulating evaluation results...
DONE (t=0.03s).
IoU metric: bbox
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.001
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.002
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.000
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.000
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.001
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.009
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.007
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.029
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.074
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.028
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.088
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.167
SAVING PLOTS COMPLETE...
...
Epoch: [4]  [ 0/84]  eta: 0:00:20  lr: 0.001000  loss: 0.9575 (0.9575)  time: 0.2461  data: 0.1662  max mem: 1548
Epoch: [4]  [83/84]  eta: 0:00:00  lr: 0.001000  loss: 1.1325 (1.1624)  time: 0.0762  data: 0.0078  max mem: 1548
Epoch: [4] Total time: 0:00:06 (0.0801 s / it)
creating index...
index created!
Test:  [0/9]  eta: 0:00:02  model_time: 0.0369 (0.0369)  evaluator_time: 0.0237 (0.0237)  time: 0.2494  data: 0.1581  max mem: 1548
Test:  [8/9]  eta: 0:00:00  model_time: 0.0323 (0.0330)  evaluator_time: 0.0226 (0.0227)  time: 0.1076  data: 0.0271  max mem: 1548
Test: Total time: 0:00:01 (0.1116 s / it)
Averaged stats: model_time: 0.0323 (0.0330)  evaluator_time: 0.0226 (0.0227)
Accumulating evaluation results...
DONE (t=0.03s).
IoU metric: bbox
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.137
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.313
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.118
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.029
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.175
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.428
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.204
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.306
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.347
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.140
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.424
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.683
SAVING PLOTS COMPLETE...
```



## A List of All Model Flags to Use With the Training Script

The following command expects the `coco` dataset to be present one directory back inside the `input` folder in XML format. You can find the dataset [here on Kaggle](https://www.kaggle.com/datasets/sovitrath/coco-xml-format). Check the `configs/coco.yaml` for more details. You can change the relative dataset path in the YAML file according to your structure.

```bash
# Usage
python train.py --model fasterrcnn_resnet50_fpn_v2 --config configs/coco.yaml
```


OR **see the list of all the available models in the [\_\_INIT\_\_.py](models/__init__.py) file.**



## Check all available arguments for training

```
usage: train.py [-h]
                [-m {fasterrcnn_resnet50_fpn,fasterrcnn_mobilenetv3_large_fpn,fasterrcnn_mobilenetv3_large_320_fpn,fasterrcnn_convnext_tiny,fasterrcnn_convnext_small,fasterrcnn_custom_resnet,fasterrcnn_darknet,fasterrcnn_efficientnet_b0,fasterrcnn_mbv3_small_nano_head,fasterrcnn_mini_darknet,fasterrcnn_mini_darknet_nano_head,fasterrcnn_squeezenet1_0,fasterrcnn_squeezenet1_1,fasterrcnn_mini_squeezenet1_1_small_head,fasterrcnn_mini_squeezenet1_1_tiny_head,fasterrcnn_mobilevit_xxs,fasterrcnn_nano,fasterrcnn_squeezenet1_1_small_head,fasterrcnn_resnet18,fasterrcnn_resnet50_fpn_v2,fasterrcnn_resnet101,fasterrcnn_resnet152,fasterrcnn_vitdet,fasterrcnn_vitdet_tiny,fasterrcnn_regnet_y_400mf,fasterrcnn_vgg16,fcos_mobilinet_v2,retinanet_resnet50,retinanet_mobilenet_v2,ssd_vgg16}]
                [--config CONFIG] [-d DEVICE] [-e EPOCHS] [-j WORKERS] [-b BATCH] [--lr LR] [-ims IMGSZ] [-n NAME] [-vt] [--mosaic MOSAIC] [-uta] [-w WEIGHTS] [-r] [-st]
                [--world-size WORLD_SIZE] [--dist-url DIST_URL] [-we] [-wd] [--sync-bn] [--amp] [--patience PATIENCE] [--optimizer {adam,sgd}] [--momentum MOMENTUM]
                [--weight-decay WEIGHT_DECAY] [-ca] [--seed SEED] [--project-dir PROJECT_DIR]

options:
  -h, --help            show this help message and exit
  -m {fasterrcnn_resnet50_fpn,fasterrcnn_mobilenetv3_large_fpn,fasterrcnn_mobilenetv3_large_320_fpn,fasterrcnn_convnext_tiny,fasterrcnn_convnext_small,fasterrcnn_custom_resnet,fasterrcnn_darknet,fasterrcnn_efficientnet_b0,fasterrcnn_mbv3_small_nano_head,fasterrcnn_mini_darknet,fasterrcnn_mini_darknet_nano_head,fasterrcnn_squeezenet1_0,fasterrcnn_squeezenet1_1,fasterrcnn_mini_squeezenet1_1_small_head,fasterrcnn_mini_squeezenet1_1_tiny_head,fasterrcnn_mobilevit_xxs,fasterrcnn_nano,fasterrcnn_squeezenet1_1_small_head,fasterrcnn_resnet18,fasterrcnn_resnet50_fpn_v2,fasterrcnn_resnet101,fasterrcnn_resnet152,fasterrcnn_vitdet,fasterrcnn_vitdet_tiny,fasterrcnn_regnet_y_400mf,fasterrcnn_vgg16,fcos_mobilinet_v2,retinanet_resnet50,retinanet_mobilenet_v2,ssd_vgg16}, --model {fasterrcnn_resnet50_fpn,fasterrcnn_mobilenetv3_large_fpn,fasterrcnn_mobilenetv3_large_320_fpn,fasterrcnn_convnext_tiny,fasterrcnn_convnext_small,fasterrcnn_custom_resnet,fasterrcnn_darknet,fasterrcnn_efficientnet_b0,fasterrcnn_mbv3_small_nano_head,fasterrcnn_mini_darknet,fasterrcnn_mini_darknet_nano_head,fasterrcnn_squeezenet1_0,fasterrcnn_squeezenet1_1,fasterrcnn_mini_squeezenet1_1_small_head,fasterrcnn_mini_squeezenet1_1_tiny_head,fasterrcnn_mobilevit_xxs,fasterrcnn_nano,fasterrcnn_squeezenet1_1_small_head,fasterrcnn_resnet18,fasterrcnn_resnet50_fpn_v2,fasterrcnn_resnet101,fasterrcnn_resnet152,fasterrcnn_vitdet,fasterrcnn_vitdet_tiny,fasterrcnn_regnet_y_400mf,fasterrcnn_vgg16,fcos_mobilinet_v2,retinanet_resnet50,retinanet_mobilenet_v2,ssd_vgg16}
                        name of the model
  --config CONFIG       path to the data config file
  -d DEVICE, --device DEVICE
                        computation/training device, default is GPU if GPU present
  -e EPOCHS, --epochs EPOCHS
                        number of epochs to train for
  -j WORKERS, --workers WORKERS
                        number of workers for data processing/transforms/augmentations
  -b BATCH, --batch BATCH
                        batch size to load the data
  --lr LR               learning rate for the optimizer
  -ims IMGSZ, --imgsz IMGSZ
                        image size to feed to the network
  -n NAME, --name NAME  training result dir name in outputs/training/, (default res_#)
  -vt, --vis-transformed
                        visualize transformed images fed to the network
  --mosaic MOSAIC       probability of applying mosaic, (default, always apply)
  -uta, --use-train-aug
                        whether to use train augmentation, blur, gray, brightness contrast, color jitter, random gamma all at once
  -w WEIGHTS, --weights WEIGHTS
                        path to model weights if using pretrained weights
  -r, --resume-training
                        whether to resume training, if true, loads previous training plots and epochs and also loads the optimizer state dictionary
  -st, --square-training
                        Resize images to square shape instead of aspect ratio resizing for single image training. For mosaic training, this resizes single images to square shape
                        first then puts them on a square canvas.
  --world-size WORLD_SIZE
                        number of distributed processes
  --dist-url DIST_URL   url used to set up the distributed training
  -we, --enable-wandb   use wandb
  -wd, --disable-wandb  do not use wandb
  --sync-bn             use sync batch norm
  --amp                 use automatic mixed precision
  --patience PATIENCE   number of epochs to wait for when mAP does not increase to trigger early stopping
  --optimizer {adam,adamw,adamax,nadam,sgd}
                        type of optimizer
  --momentum MOMENTUM   optimizer momentum or beta1 in case of Adam opt
  --weight-decay WEIGHT_DECAY
                        optimizer weight decay (L2 penalty)
  -ca, --cosine-annealing
                        use cosine annealing warm restarts
  --seed SEED           global seed for training
  --project-dir PROJECT_DIR
                        save resutls to custom dir instead of `outputs` directory, --project-dir will be named if not already present
```

For more up-to-date list, call `python3 train.py -h`.



## Distributed Training

**Training on 2 GPUs**:

```bash
export CUDA_VISIBLE_DEVICES=0,1
python -m torch.distributed.launch --nproc_per_node=2 --use_env train.py --config configs/smoke.yaml --epochs 100 --model fasterrcnn_resnet50_fpn --name smoke_training --batch 16
```

# Inference

The `inference.py` performs inference on a trained object detection model.
It can be run from the command line using the following command:

```bash
python inference.py --input <input_image_or_directory> --weights <path_to_model_weights> --data <path_to_data_config> --output <output_directory>
```

The script expects the following command-line arguments:
+ `--input`: the input image or directory containing images to perform inference on
+ `--weights`: the path to the trained model weights file
+ `--data`: the path to the data configuration file
+ `--output`: the output directory where the inference results will be saved

The script loads the trained model and data configurations using the provided paths.
It performs inference on the input images and saves the results to the output directory.


### Image Inference on COCO Pretrained Model

By default using **Faster RCNN ResNet50 FPN V2** model.

```bash
python inference.py
```

Use model of your choice with an image input.

```bash
python inference.py --model fasterrcnn_mobilenetv3_large_fpn --input example_test_data/image_1.jpg
```

### Image Inference in Custom Trained Model

In this case you only need to give the weights file path and input file path. The config file and the model name are optional. If not provided they will will be automatically inferred from the weights file.

```bash
python inference.py --input data/inference_data/image_1.jpg --weights outputs/training/smoke_training/last_model_state.pth
```

### Video Inference on COCO Pretrrained Model

```bash
python inference_video.py
```

### Video Inference in Custom Trained Model

```bash
python inference_video.py --input data/inference_data/video_1.mp4 --weights outputs/training/smoke_training/last_model_state.pth
```

### Tracking using COCO Pretrained Models

```bash
# Track all COCO classes (Faster RCNN ResNet50 FPN V2).
python inference_video.py --track --model fasterrcnn_resnet50_fpn_v2 --show

# Track all COCO classes (Faster RCNN ResNet50 FPN V2) using own video.
python inference_video.py --track --model fasterrcnn_resnet50_fpn_v2 --show --input ../inference_data/video_1.mp4

# Tracking only person class (index 1 in COCO pretrained). Check `COCO_91_CLASSES` attribute in `configs/coco.yaml` for more information.
python inference_video.py --track --model fasterrcnn_resnet50_fpn_v2 --show --input ../inference_data/video_4.mp4 --classes 1

# Tracking only person and car classes (indices 1 and 3 in COCO pretrained). Check `COCO_91_CLASSES` attribute in `configs/coco.yaml` for more information.
python inference_video.py --track --model fasterrcnn_resnet50_fpn_v2 --show --input ../inference_data/video_4.mp4 --classes 1 3

# Tracking using custom trained weights. Just provide the path to the weights instead of model name.
python inference_video.py --track --weights outputs/training/fish_det/best_model.pth --show --input ../inference_data/video_6.mp4
```

## Evaluation

Replace the required arguments according to your need.

```bash
python eval.py --model fasterrcnn_resnet50_fpn_v2 --weights outputs/training/trial/best_model.pth --config configs/aquarium.yaml --batch 4
```

You can use the following command to show a table for **class-wise Average Precision** (`--verbose` additionally needed).

```bash
python eval.py --model fasterrcnn_resnet50_fpn_v2 --weights outputs/training/trial/best_model.pth --config configs/aquarium.yaml --batch 4 --verbose
```






# ONNX


The `export.py` script can be used to export a trained model to the ONNX format.
The program imports uses `torch` library to load the trained model and exporting it to ONNX.
The script can be run from the command line using the following command:

```bash
python export.py --weights <path_to_model_weights> --data <path_to_data_config> --out <output_file_name>
```

The script expects three command-line arguments:
   + `--weights`: the path to the trained model weights file (e.g. `outputs/training/fasterrcnn_resnet18_train/best_model.pth`)
   + `--data`: the path to the data configuration file (e.g. `configs/coco.yaml`)
   + `--out`: the output file name for the exported ONNX model (e.g. `model.onnx`)


## Requirements

The script requires `onnx` and `onnxruntime`.


## Inferences

- `onnx_inference_image`
- `onnx_inference_video`

