# README



These scripts have been borrowed/copied from https://github.com/pytorch/vision/tree/main/references/detection.

## engine

`engine.py` is a helper module that contains functions for training and evaluating a model using the PyTorch framework.

* The `train_one_epoch` function trains the model for one epoch and returns the metric logger and lists of losses collected during training. This function takes in several arguments, including:
	+ `model`: The PyTorch model to be trained.
	+ `optimizer`: The optimizer to be used for training.
	+ `data_loader`: The data loader that provides the training data.
	+ `device`: The device (CPU or GPU) to be used for training.
	+ `epoch`: The current epoch number.
	+ `train_loss_hist`: A histogram to store the training losses.
	+ `print_freq`: The frequency at which to print the training losses.
	+ `scaler`: An optional scaler for automatic mixed precision training.
	+ `scheduler`: An optional scheduler for the learning rate.


* The `evaluate` function evaluates the model on the given dataset and returns a tuple containing the evaluation stats and the validation prediction image. The main parameters are:
	+ `model`: The PyTorch model to be evaluated.
	+ `data_loader`: The data loader that provides the evaluation data.
	+ `device`: The device (CPU or GPU) to be used for evaluation.
	+ `save_valid_preds`: An optional flag to save the validation predictions.
	+ `out_dir`: An optional directory to save the validation predictions.
	+ `classes`: An optional list of class names.
	+ `colors`: An optional list of colors to use for the class labels.



## coco_eval

This script implements the CocoEvaluator class, which is a utility class for evaluating the performance of object detection models using the COCO (Common Objects in Context) dataset. It provides methods to update the evaluator with new predictions, synchronize the evaluator across multiple processes, accumulate the evaluation results, and summarize the evaluation results.

The CocoEvaluator class is designed to handle the evaluation of object detection models using the COCO dataset. It provides a convenient way to update the evaluator with new predictions, synchronize the evaluator across multiple processes, and summarize the evaluation results.


## coco_utils

`coco_utils.py` provides utility functions and classes for working with the COCO dataset,
including filtering and remapping categories, converting polygon annotations to masks,
and converting datasets to the COCO API format.

