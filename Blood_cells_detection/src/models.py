from torchvision.models.detection import (
    ssd300_vgg16, SSD300_VGG16_Weights,
    fasterrcnn_resnet50_fpn_v2, FasterRCNN_ResNet50_FPN_V2_Weights,
    retinanet_resnet50_fpn_v2, RetinaNet_ResNet50_FPN_V2_Weights,
    fcos_resnet50_fpn, FCOS_ResNet50_FPN_Weights
)
from torchvision.models.detection.ssd import SSDClassificationHead
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.retinanet import RetinaNetClassificationHead
from torchvision.models.detection.fcos import FCOSClassificationHead

def get_ssd(num_classes=4, pretrained=True):
    """Инициализация SSD300"""
    weights = SSD300_VGG16_Weights.DEFAULT if pretrained else None
    model = ssd300_vgg16(weights=weights)
    in_channels = [layer.in_channels for layer in model.head.classification_head.module_list]
    num_anchors = model.anchor_generator.num_anchors_per_location()
    model.head.classification_head = SSDClassificationHead(in_channels, num_anchors, num_classes)
    return model

def get_faster_rcnn(num_classes=4, pretrained=True):
    """Инициализация Faster R-CNN"""
    weights = FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT if pretrained else None
    model = fasterrcnn_resnet50_fpn_v2(weights=weights)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model

def get_retinanet(num_classes=4, pretrained=True):
    """Инициализация RetinaNet"""
    weights = RetinaNet_ResNet50_FPN_V2_Weights.DEFAULT if pretrained else None
    model = retinanet_resnet50_fpn_v2(weights=weights)
    in_channels = model.head.classification_head.conv[0].out_channels
    num_anchors = model.head.classification_head.num_anchors
    model.head.classification_head = RetinaNetClassificationHead(in_channels, num_anchors, num_classes)
    return model

def get_fcos(num_classes=4, pretrained=True):
    """Инициализация FCOS"""
    weights = FCOS_ResNet50_FPN_Weights.DEFAULT if pretrained else None
    model = fcos_resnet50_fpn(weights=weights)
    in_channels = model.head.classification_head.conv[0].in_channels
    num_anchors = model.head.classification_head.num_anchors
    model.head.classification_head = FCOSClassificationHead(in_channels, num_anchors, num_classes)
    return model
