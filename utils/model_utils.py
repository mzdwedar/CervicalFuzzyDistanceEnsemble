import tensorflow as tf
from tensorflow.keras.models import Model

import os
import pandas as pd

def create_model(model_name,IMG_SIZE = 256, output = 6):


    IMG_SHAPE = (IMG_SIZE, IMG_SIZE, 3)  # IMG_SIZE = 256
    if(model_name == "MobileNetV2" ):

        model = tf.keras.applications.MobileNetV2(input_shape=IMG_SHAPE,
                                                            include_top=False,
                                                            weights='imagenet')
    elif(model_name == "InceptionV3"):
        model = tf.keras.applications.inception_v3.InceptionV3(input_shape=IMG_SHAPE,
                                                                        include_top=False,
                                                                        weights='imagenet')
        
    elif(model_name == "InceptionResNetV2"):
        model = tf.keras.applications.inception_resnet_v2.InceptionResNetV2(input_shape=IMG_SHAPE,
                                                                                        include_top=False,
                                                                                        weights='imagenet')
    else:
        return        

    x = tf.keras.layers.Conv2D(128, (3, 3), activation='relu')(model.output)
    x = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(x)
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(100, activation='relu')(x)
    x = tf.keras.layers.Dense(output, activation='softmax')(x)

    model = Model(inputs=model.input, outputs=x)

    my_model = tf.keras.models.clone_model(model)
    return my_model

def save_model(model, model_name, history):
    ''' save the model and the metrics'''
    os.makedirs('saved_models', exist_ok=True)

    model_saved_name = model_name + ".h5"
    
    model.save("saved_model/" + model_saved_name)

    hist_df = pd.DataFrame(history.history) 
    hist_csv_file =  "history_" + model_name + ".csv"
    filepath = "saved_models/" + hist_csv_file 
    with open(filepath, mode='w') as f:
        hist_df.to_csv(f)

    print(f'{model_saved_name} saved')
    print(f'{hist_csv_file} saved')

def load_hdf5_model(model_name, path='saved_models'):
    '''load saved hdf5 model'''
    model_path = path + "/" + model_name + ".h5"
    model = tf.keras.models.load_model(model_path)

    print(f'{model_name} is loaded')

    return model