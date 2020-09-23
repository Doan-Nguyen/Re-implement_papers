from collections import defaultdict, OrderedDict
from tensorflow.keras.models import Model
from tensorflow.keras.applications.vgg16 import VGG16 as PreModel

layer_size_dict = defaultdict(list)
inputs = []

base_pretrained_model = PreModel(
                            input_shape=(512, 512, 3), 
                            include_top=False, 
                            weights='imagenet')
base_pretrained_model.trainable = False
base_pretrained_model.summary()

for lay_idx, c_layer in enumerate(base_pretrained_model.layers):
    if not c_layer.__class__.__name__ == 'InputLayer':
        layer_size_dict[c_layer.get_output_shape_at(0)[1:3]] += [c_layer]
    else:
        inputs += [c_layer]

###     freeze dict
layer_size_dict = OrderedDict(layer_size_dict.items())
for k, v in layer_size_dict.items():
    print(k, [w.__class__.__name__ for w in v])

###     Take the last layer of each shape & make it into 
pretrained_encoder = Model(
                        inputs=base_pretrained_model.get_input_at(0),
                        outputs=[v[-1]].get_output_at(0) for k, v in layer_size_dict.items()]
                    )

pretrained_encoder.trainable = False