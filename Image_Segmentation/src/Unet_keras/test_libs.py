import segmentation_models as sm 

BACKBONE = 'vgg16'
preprocess_input = sm.get_preprocessing(BACKBONE)

###         Load datasets
x_train, y_train, x_val, y_val = load_data()