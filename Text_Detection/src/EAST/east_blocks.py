import torch 
import torch.nn as nn 
import torch.utils.model_zoo as model_zoo 
import torch.nn.functional as F 
import math

config = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M']

def make_layers(cfg, batch_norm=False):
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


class FeatureExtractor(nn.Module):
    """
    64 -> 128 -> 256 -> 512
    """
    def __init__(self, pretrained):
        """
        - self.features: 
        """
        super(Extractor, self).__init__()
        ### modify architecture 
        vgg16_bn = VGG(make_layers(cfg, batch_norm=True))
        if pretrained:
            vgg16_bn.load_state_dict(torch.load('./pths/vgg16_bn-6c64b313.pth'))
        self.features = vgg16_bn.features
    
    def forward(self, x):
        ### where feature maps after pooling-2 to pooling-5 are extracted
        out = []
        for m in self.features:
            x = m(x)
            if isinstance(m, nn.MaxPool2d):
                out.append(x)
        return out[1:]


class FeatureMerge(nn.Module):
    def __init__(self):
        """
        - self.features: 
        """
        super(Extractor, self).__init__()
        ### modify architecture 
        # take output_feature_extractor*2 ~ input
        self.conv1_1 = nn.Conv2d(512*2, 128, kernel_size=1)
        self.bn1_1 = nn.BatchNorm2d(num_features=128)
        self.relu1_1 = nn.ReLU(inplace=True)
        self.conv1_2 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.bn1_2 = nn.BatchNorm2d(num_features=128)
        self.relu1_2 = nn.ReLU(inplace=True)

        self.conv2_1 = nn.Conv2d(128 + 256, 64, kernel_size=1)
        self.bn2_1 = nn.BatchNorm2d(num_features=64)
        self.relu2_1 = nn.ReLU(inplace=True)
        self.conv2_2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.bn2_2 = nn.BatchNorm2d(num_features=64)
        self.relu2_2 = nn.ReLU(inplace=True)

        self.conv3_1 = nn.Conv2d(64 + 128, 32, kernel_size=1)
        self.bn3_1 = nn.BatchNorm2d(num_features=32)
        self.relu3_1 = nn.ReLU(inplace=True)
        self.conv3_2 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        self.bn3_2 = nn.BatchNorm2d(num_features=32)
        self.relu3_2 = nn.ReLU(inplace=True)

        self.conv4_1 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        self.bn4_1 = nn.BatchNorm2d(num_features=32)
        self.relu4_1 = nn.ReLU(inplace=True)

        if m in self.modules():  # Returns an iterator over all modules in the network
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(
                    m.weight, 
                    mode='fan_out', 
                    nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)  
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)             

    def forward(self, x):
        y = F.interpolate(
                    x[3], 
                    scale_factor=2, 
                    mode='bilinear', 
                    align_corners=True)
        y = torch.cat((y, x[2]), 1)
        y = self.relu1_1(self.bn1_1(self.conv1_1(y)))
        y = self.relu1_2(self.bn1_2(self.conv1_2(y)))

        y = F.interpolate(
                    y, 
                    scale_factor=2, 
                    mode='bilinear',
                    align_corners=True)
        y = torch.cat((y, x[1]), 1)
        y = self.relu2_1(self.bn2_1(self.conv2_1(y)))
        y = self.relu2_2(self.bn2_2(self.conv2_2(y)))

        y = F.interpolate(
                    y, 
                    scale_factor=2, 
                    mode='bilinear',
                    align_corners=True)
        y = torch.cat((y, x[0]), 1)
        y = self.relu3_1(self.bn3_1(self.conv3_1(y)))
        y = self.relu3_2(self.bn3_2(self.conv3_2(y)))

        y = self.relu4_1(self.bn4_1(self.conv4_1(y)))

        return y

class OutputBlock(nn.Module):
    def __init__(self, scope=512):
        super(OutputBlock, self).__init__()
        ### modify architecture 
        self.conv1 = nn.Conv2d(32, 1, 1)
        self.sigmoid1 = nn.Sigmoid()
        self.conv2 = nn.Conv2d(32, 4, 1)
        self.sigmoid2 = nn.Sigmoid()
        self.conv3 = nn.Conv2d(32, 1, 1)
        self.sigmoid3 = nn.Sigmoid()
        # self.conv4 = nn.Conv2d(32, 8, 1)
        # self.sigmoid4 = nn.Sigmoid()
        self.scope = 512
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
        
    def forward(self, x):
		score = self.sigmoid1(self.conv1(x))
		loc   = self.sigmoid2(self.conv2(x)) * self.scope
		angle = (self.sigmoid3(self.conv3(x)) - 0.5) * math.pi
		geo   = torch.cat((loc, angle), 1) 
		return score, geo

        

