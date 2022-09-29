from utils.model_utils import load_hdf5_model
from utils.model_eval import predict
from utils.fuzzy_dist_ensemble import fuzzy_dist

import numpy as np
import pandas as pd

def inference(path):
    '''
    1. process the dataset
    2. load the models
    3. make inference
    4. save the result to a csv file
    '''
    # process the data from the directory
    #wrap it with TF object
    
    model1 = load_hdf5_model("InceptionV3")
    model2 = load_hdf5_model("MobileNetV2")
    model3 = load_hdf5_model("InceptionResNetV2")

    preds1 = predict(model1, dataset, num_examples_test)
    preds2 = predict(model2, dataset, num_examples_test)
    preds3 = predict(model3, dataset, num_examples_test)

    ensem_preds = fuzzy_dist(preds1, preds2, preds3)

    y_preds=[]
    for pred in ensem_preds: 
        y_preds.append(np.argmax(pred))

    df = pd.DataFrame(path)
    df['predictions'] = y_preds
    df.to_csv('path/predictions')