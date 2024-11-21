from models import *


create_model = {
    'fasterrcnn_resnet50_fpn': fasterrcnn_resnet50_fpn.create_model,  # type: ignore
    'fasterrcnn_mobilenetv3_large_fpn': fasterrcnn_mobilenetv3_large_fpn.create_model,  # type: ignore
    'fasterrcnn_mobilenetv3_large_320_fpn': fasterrcnn_mobilenetv3_large_320_fpn.create_model,  # type: ignore
    'fasterrcnn_convnext_tiny': fasterrcnn_convnext_tiny.create_model,  # type: ignore
    'fasterrcnn_convnext_small': fasterrcnn_convnext_small.create_model,  # type: ignore
    'fasterrcnn_custom_resnet': fasterrcnn_custom_resnet.create_model,  # type: ignore
    'fasterrcnn_darknet': fasterrcnn_darknet.create_model,  # type: ignore
    'fasterrcnn_efficientnet_b0': fasterrcnn_efficientnet_b0.create_model,  # type: ignore
    'fasterrcnn_efficientnet_b1': fasterrcnn_efficientnet_b1.create_model,  # type: ignore
    'fasterrcnn_efficientnet_b2': fasterrcnn_efficientnet_b2.create_model,  # type: ignore
    'fasterrcnn_efficientnet_b3': fasterrcnn_efficientnet_b3.create_model,  # type: ignore
    'fasterrcnn_efficientnet_b4': fasterrcnn_efficientnet_b4.create_model,  # type: ignore
    'fasterrcnn_efficientnet_b5': fasterrcnn_efficientnet_b5.create_model,  # type: ignore
    'fasterrcnn_efficientnet_b6': fasterrcnn_efficientnet_b6.create_model,  # type: ignore
    'fasterrcnn_efficientnet_b7': fasterrcnn_efficientnet_b7.create_model,  # type: ignore
    'fasterrcnn_mbv3_small_nano_head': fasterrcnn_mbv3_small_nano_head.create_model,  # type: ignore
    'fasterrcnn_mini_darknet': fasterrcnn_mini_darknet.create_model,  # type: ignore
    'fasterrcnn_mini_darknet_nano_head': fasterrcnn_mini_darknet_nano_head.create_model,  # type: ignore
    'fasterrcnn_squeezenet1_0': fasterrcnn_squeezenet1_0.create_model,  # type: ignore
    'fasterrcnn_squeezenet1_1': fasterrcnn_squeezenet1_1.create_model,  # type: ignore
    'fasterrcnn_mini_squeezenet1_1_small_head': fasterrcnn_mini_squeezenet1_1_small_head.create_model,  # type: ignore
    'fasterrcnn_mini_squeezenet1_1_tiny_head': fasterrcnn_mini_squeezenet1_1_tiny_head.create_model,  # type: ignore
    'fasterrcnn_mobilevit_xxs': fasterrcnn_mobilevit_xxs.create_model,  # type: ignore  # needs CUDA
    'fasterrcnn_nano': fasterrcnn_nano.create_model,  # type: ignore
    'fasterrcnn_squeezenet1_1_small_head': fasterrcnn_squeezenet1_1_small_head.create_model,  # type: ignore
    'fasterrcnn_resnet18': fasterrcnn_resnet18.create_model,  # type: ignore
    'fasterrcnn_resnet50_fpn_v2': fasterrcnn_resnet50_fpn_v2.create_model,  # type: ignore
    'fasterrcnn_resnet101': fasterrcnn_resnet101.create_model,  # type: ignore
    'fasterrcnn_resnet152': fasterrcnn_resnet152.create_model,  # type: ignore
    'fasterrcnn_vitdet': fasterrcnn_vitdet.create_model,  # type: ignore
    'fasterrcnn_vitdet_tiny': fasterrcnn_vitdet_tiny.create_model,  # type: ignore
    'fasterrcnn_regnet_y_400mf': fasterrcnn_regnet_y_400mf.create_model,  # type: ignore
    'fasterrcnn_vgg16': fasterrcnn_vgg16.create_model,  # type: ignore
    'fcos_mobilinet_v2': fcos_mobilinet_v2.create_model,  # type: ignore
    'retinanet_resnet50': retinanet_resnet50.create_model,  # type: ignone
    'retinanet_mobilenet_v2': retinanet_mobilenet_v2.create_model,  # type: ignone
    'ssd_vgg16': ssd_vgg16.create_model,  # type: ignore
    'ssd_resnet50': ssd_resnet50.create_model,  # type: ignore
}


# Example:
# ========
#
# cd detection_pipeline_finetunning
# python3 -m models.create_detection_model
#
if __name__ == "__main__":
    import argparse
    import torch

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-s', '--show-model',
        action='store_true',
        help='print the model'
    )
    args = parser.parse_args()


    for model_name in create_model.keys():
        build_model = create_model[model_name]
        model = build_model(num_classes=20)
        model.eval().to("cpu")
        x = torch.rand((1, 3, 800, 800))  # batch with 5 elements
        y = model(x)
        print("Model name:", model_name)
        print("Input:", x.shape)
        if args.show_model:
            print(model)
        print("Output:", len(y), "elements ->", y[0].keys())  #
        print()
