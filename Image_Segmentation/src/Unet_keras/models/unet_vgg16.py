##          Basic packages
import sys 
##          Import files
import blocks
sys.path.append('../')
import configs


class UNet(keras.Model):
    def __init__(self):
        super(UNet, self).__init__()
        ###
        self.input_block = blocks.InputBlock(filters=64)
        self.bottleneck = blocks.BottleneckBlock(filters=1024)
        self.output_block = blocks.OutputBlock(filters=64, n_classes=2)

        self.down_blocks = [blocks.DownsampleBlock(filters, idx)
                            for idx, filters in enumerate([128, 256, 512])]
        self.up_blocks= [blocks.UpsampleBlock(filters, idx)
                            for idx, filters in enumerate([512, 256, 128])]

    def call(self, inputs):
        skip_connections = []
        out, residual = self.input_block(inputs)
        skip_connections.append(residual)

        for down_block in self.down_blocks:
            out, residual = down_block(out)
            skip_connections.append(residual)

        out = self.bottleneck(out, training)

        for up_block in self.up_blocks:
            out, residual = up_block(out)
        
        out = self.output_block(out, skip_connections.pop())
        
