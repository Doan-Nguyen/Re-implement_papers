from ctpn_blocks import * 
from torchvision import models


class CTPN_Model(nn.Module):
    def __init__(self):
        """
        Architectures:
            - base_layers: Vgg16 remove conv5
            - rpn: region proposal network
            - brnn: 
        """

        super(CTPN_Model, self).__init__()

        base_model = models.vgg16(pretrained=False)
        layers = list(base_model.features)[:-1]  # through the last convolutional maps (conv5 )
        self.base_layers = nn.Sequential(*layers)
        self.rpn = BasicConvBlock(512, 512, 3, 1, 1, batch_norm=False)
        self.brnn = nn.GRU(input_size=512, 
                        hidden_size=128, # each window is use the 256-D BLSTM (include 2 128-D LSTM)
                        bidirectional=True, 
                        batch_first=True)
        self.lstm_fc = BasicConvBlock(
                        in_channels=256,
                        out_channels=512, 
                        kernel_size=1, 
                        stride=1, 
                        relu=True, 
                        batch_norm=False)
        self.rpn_class = BasicConvBlock(
                        in_channels=512,
                        out_channels=10*2, 
                        kernel_size=1, 
                        stride=1, 
                        relu=False, 
                        batch_norm=False)
        self.rpn_regress = BasicConvBlock(
                        in_channels=512,
                        out_channels=10*2, 
                        kernel_size=1, 
                        stride=1, 
                        relu=False, 
                        batch_norm=False)
        
    def forward(self, x):
        x = self.base_layers(x)
        ## rpn
        x = self.rpn(x)

        x1 = x.permute(0, 2, 3, 1).contiguous() # channels last
        b = x1.size() 
        x1 = x1.view(b[0]*b[1], b[2], b[3])

        x2, _ = self.brnn(x1)

        xsz = x.size()
        x3 = x2.view(xsz[0], xsz[2], xsz[3], 256) # 

        x3 = x3.permute(0, 3, 1, 2).contiguous()
        x3 = self.lstm_fc(x3)
        x = x3 

        cls = self.rpn_class(x)
        regr = self.rpn_regress(x)

        cls = cls.view(cls.size(0), cls.size(1)*cls.size(2)*10, 2)
        regr = regr.view(regr.size(0), regr.size(1)*regr.size(2)*10, 2)

        return cls, regr


