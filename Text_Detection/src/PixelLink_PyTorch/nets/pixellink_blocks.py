import torch 
import torch.nn as nn 
import torch.utils.model_zoo as model_zoo 
import torch.nn.functional as F 
import math

config = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M']

def make_layers(configs, batch_norm=False):
    layers = []
    in_channels=3
    for v in config:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(
                in_channels= in_channels,
                out_channels= v,
                kernel_size=3, 
                padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)


class VGG(nn.Module):
    def __init__(self, features, num_classes=2, init_weights=True):
        """
        Parameters:
            - features:
            - num_classes:
            - init_weights:
        """

        super(VGG, self).__init__()
        ## modify the model architecture
        self.features = features
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        self.classifier = nn.Sequential(
            nn.Linear(in_features= 7*7*512, out_features=4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(in_features=4096, out_features=4096),
            nn.ReLU(inplace=True), 
            nn.Dropout(), 
            nn.Linear(in_features=4096, out_features=num_classes)
        )

        if init_weights:
            self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        # x = torch.flatten(x, 1)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)

        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weigth, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)


# def _vgg(arch, config, batch_norm, pretrained, progress, **kwargs):
#     """
#     Parameters:
#         - arch: take url to download pretrained
#         - config: config layers of vgg
#         - pretrained: (bool)
#         - progress: 
#     """
#     if pretrained:
#         kwargs['init_weights'] = False
#     model = VGG(make_layers(configs[config], 
#                 batch_norm= batch_norm), 
#                 **kwargs)
#     if pretrained:
#         state_dict = load