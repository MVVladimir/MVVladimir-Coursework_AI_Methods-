from torchvision import models
import torch

from configs import NUM_CLASSES

def shufflenet_v2_x0_5_loader():
    model = models.shufflenet_v2_x0_5()
    model.fc = torch.nn.Linear(in_features=model.fc.in_features, out_features=NUM_CLASSES)
    return model
def resnet18_loader():
    model = models.resnet18()
    model.fc = torch.nn.Linear(in_features=model.fc.in_features, out_features=NUM_CLASSES)
    return model

def resnet34_loader():
    model = models.resnet34()
    model.fc = torch.nn.Linear(in_features=model.fc.in_features, out_features=NUM_CLASSES)
    return model

def resnet50_loader():
    model = models.resnet50()
    model.fc = torch.nn.Linear(in_features=model.fc.in_features, out_features=NUM_CLASSES)
    return model

def alexnet_loader():
    model = models.alexnet()
    model.classifier[6] = torch.nn.Linear(in_features=model.classifier[6].in_features, out_features=NUM_CLASSES)
    return model

def mobilenet_v3_small_loader():
    model = models.mobilenet_v3_small()
    model.classifier[3] = torch.nn.Linear(in_features=model.classifier[3].in_features, out_features=NUM_CLASSES)
    return model

def mobilenet_v3_large_loader():
    model = models.mobilenet_v3_large()
    model.classifier[3] = torch.nn.Linear(in_features=model.classifier[3].in_features, out_features=NUM_CLASSES)
    return model

def efficientnet_v2_s_loader():
    model = models.efficientnet_v2_s()
    model.classifier[1] = torch.nn.Linear(in_features=model.classifier[1].in_features, out_features=NUM_CLASSES)
    return model

def efficientnet_v2_m_loader():
    model = models.efficientnet_v2_m()
    model.classifier[1] = torch.nn.Linear(in_features=model.classifier[1].in_features, out_features=NUM_CLASSES)
    return model

def vgg11_loader():
    model = models.vgg11()
    model.classifier[6] = torch.nn.Linear(in_features=model.classifier[6].in_features, out_features=NUM_CLASSES)
    return model

# === FOR IMAGE TESTING ==========

def shufflenet_v2_x0_5_params(model):
    # return model.conv5[0], model.fc
    return model.conv5[0]

def resnet18_params(model):
    # return model.layer4[1].conv2, model.fc
    return model.layer4[1].conv2

def resnet34_params(model):
    # return model.layer4[2].conv2, model.fc
    return model.layer4[2].conv2

def resnet50_params(model):
    # return model.layer4[2].conv3, model.fc
    return model.layer4[2].conv3

def alexnet_params(model):
    # return model.features[10], model.classifier[6]
    return model.features[10]

def mobilenet_v3_small_params(model):
    # return model.features[12][0], model.classifier[0]
    return model.features[12][0]

def mobilenet_v3_large_params(model):
    # return model.features[16][0], model.classifier[0]
    return model.features[16][0]

def efficientnet_v2_s_params(model):
    # return model.features[7][0], model.classifier[1]
    return model.features[7][0]

def efficientnet_v2_m_params(model):
    # return model.features[8][0], model.classifier[1]
    return model.features[8][0]

def vgg11_params(model):
    # return model.features[18], model.classifier[6]
    return model.features[18]