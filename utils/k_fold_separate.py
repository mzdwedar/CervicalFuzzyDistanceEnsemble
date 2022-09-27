import tensorflow as tf 
import tensorflow_addons as tfa
import numpy as np 
from sklearn.metrics import confusion_matrix , accuracy_score, roc_auc_score
from sklearn.metrics import precision_score, recall_score, f1_score
# from sklearn.preprocessing import LabelEncoder
from utils.create_model import create_model
from utils.fuzzy_dist_ensemble import fuzzy_dist 

import pandas as pd


def encode_y(y):
  
  label2id = {'NILM':0, 'ASC-US':1, 'ASC-H':2, 'LSIL':3, 'HSIL':4, 'SCC':5}
  return label2id[y]   

def normalize(input_image):
  """normalizes the input image pixel values to be [0,1] """
  input_image = tf.cast(input_image, tf.float32)
  input_image /= 255.0
  return input_image

def parse_function(datapoint):
  """
  return a resized and normalized pair of image and mask
  args
    datapoint: a single image and its corresponding segmentation mask

  1. load the image from its path, decode it to jpeg, normalize it to [0,1]
  2. decode the run-length encoding to pixels, then project the mask onto canvas with same size as image
  3. resize both the image and segmentation mask, to math the input size of the network i.e (128,128)
  """
  input_image, label = datapoint

  input_image = tf.io.read_file(input_image)
  input_image = tf.image.decode_image(input_image, channels=3)
  input_image = tf.image.resize(input_image, (256, 256), method='nearest')
  input_image = normalize(input_image)

  label = encode_y(label)

  return input_image, label

def evaluate_model(model, val_dataset, validation_batch, num_examples_val):
  X = []
  Y = []
  y_preds = []
  
  ds = val_dataset.unbatch()
  ds = ds.batch(num_examples_val)  

  for image, annotation in ds.take(1):
    Y = annotation.numpy()
    X = image.numpy()

  preds = model.predict(X, batch_size=validation_batch )
  for pred in preds : 
    y_preds.append(np.argmax(pred))
  print('Accuracy Score: ',accuracy_score(Y,y_preds))
  n = len(precision_score(Y,y_preds , average= None ))
  print('Precision Score(Class wise): ',precision_score(Y, y_preds, average=None ), " mean- " , sum(precision_score(Y, y_preds, average= None ))/n)
  print('Recall Score(Class wise): ',recall_score(Y, y_preds, average=None ), " mean- " , sum(recall_score(Y, y_preds, average= None ))/n)
  print('F1 Score(Class wise): ',f1_score(Y, y_preds, average=None), " mean- " , sum(f1_score(Y, y_preds, average= None))/n)
  print('Conf Matrix Score(Class wise):\n ',confusion_matrix(Y, y_preds))    
  print('AUC ROC(Class wise): ',roc_auc_score(Y, y_preds, average= 'weighted'), " mean- " , sum(roc_auc_score(Y,y_preds , average= 'weighted'))/n)

  return preds

def save_model(model, model_name, history):
    ''' save the model and the metrics'''
    model_saved_name = model_name + "_weights" + "_" + ".h5"
    
    model.save("./" + model_saved_name)

    hist_df = pd.DataFrame(history.history) 
    hist_csv_file =  "history_" + model_name + "_weights" + "_" + ".csv"
    filepath = "./" + hist_csv_file 
    with open(filepath, mode='w') as f:
        hist_df.to_csv(f)


    print(f'{model_saved_name} saved')
    print(f'{hist_csv_file} saved')

def train(train_paths, val_paths, model_name1, model_name2, model_name3, NUM_EPOCHS=70, train_batch=16, validation_batch=16, lr=1e-4):
    num_examples_val = len(val_paths)

    train = tf.data.Dataset.from_tensor_slices(train_paths.values)
    val = tf.data.Dataset.from_tensor_slices(val_paths.values)

    # shuffle and group the train set into batches
    AUTOTUNE = tf.data.AUTOTUNE
    train_dataset = (train
                      .cache()
                      .shuffle(buffer_size=len(train_paths))
                      .map(parse_function, num_parallel_calls=AUTOTUNE)
                      .batch(train_batch)
                      .repeat()
                      .prefetch(buffer_size=AUTOTUNE)
                    )

    # group the test set into batches
    val_dataset = (val
                    .map(parse_function)
                    .batch(validation_batch)
                    .prefetch(buffer_size=AUTOTUNE)
                  )

    # -----------------------------------Model 1 -------------------------------------------------

    print()
    print(model_name1)
    print()
    
    model1 = create_model(model_name1)

    model1.compile(loss='sparse_categorical_crossentropy',
                  optimizer= tf.keras.optimizers.Adam(learning_rate=lr, beta_1=0.9, beta_2=0.999, decay=0.0001),
                  metrics=['accuracy'])

    history1 = model1.fit(x = train_dataset,
                         validation_data= val_dataset,
                         epochs=NUM_EPOCHS
                         ) 

    save_model(model1, model_name1, history1)

    # Generate generalization metrics
    preds1 = evaluate_model(model1, val_dataset, validation_batch, num_examples_val)

    # -----------------------------------Model 2 -------------------------------------------------
    print()
    print(model_name2)
    print()

    model2 = create_model(model_name2)

    model2.compile(loss='sparse_categorical_crossentropy',
                  optimizer= tf.keras.optimizers.Adam(learning_rate=0.0001, beta_1=0.9, beta_2=0.999, decay=0.0001),
                  metrics=['accuracy'])

    history2 = model2.fit(x = train_dataset ,
                         validation_data= val_dataset,
                         epochs=NUM_EPOCHS
                         )

    save_model(model2, model_name2, history2)

    # Generate generalization metrics
    preds2 = evaluate_model(model2, val_dataset, validation_batch, num_examples_val)
   
    # -----------------------------------Model 3 -------------------------------------------------
    model3 = create_model(model_name3)
    
    print()
    print(model_name3)
    print()

    model3.compile(loss='sparse_categorical_crossentropy',
                  optimizer= tf.keras.optimizers.Adam(learning_rate=0.0001, beta_1=0.9, beta_2=0.999, decay=0.0001),
                  metrics=['accuracy'])

    history3 = model3.fit(x=train_dataset,
                         validation_data=val_dataset,
                         epochs=NUM_EPOCHS
                         )

    save_model(model3, model_name3, history3)

    # Generate generalization metrics
    preds3 = evaluate_model(model3, val_dataset, validation_batch, num_examples_val)

    # -------------------------------------------fuzzy distance ----------------------------------------
    ensem_pred=fuzzy_dist(preds1,preds2,preds3)
    y_val = val_paths['class'].numpy()

    print('Post Ensemble Accuracy Score: ',accuracy_score(y_val,ensem_pred))
    n = len(precision_score(y_val, ensem_pred , average= None ))
    print('Post Ensemble Precision Score(Class wise): ',precision_score(y_val,ensem_pred , average= None ) , " mean- " , sum(precision_score(y_val,ensem_pred , average= None ))/n )
    print('Post Ensemble Recall Score(Class wise): ',recall_score(y_val,ensem_pred , average= None ), " mean- " , sum(recall_score(y_val,ensem_pred , average= None ))/n)
    print('Post Ensemble F1 Score(Class wise): ',f1_score(y_val,ensem_pred , average= None), " mean- " , sum(f1_score(y_val,ensem_pred , average= None))/n)
    print('Post Ensemble Conf Matrix Score(Class wise):\n ',confusion_matrix(y_val,ensem_pred ))
    print('Post EnsembleAUC ROC(Class wise): ',roc_auc_score(y_val,ensem_pred , average= 'weighted'), " mean- " , sum(roc_auc_score(y_val,ensem_pred , average= 'weighted'))/n)
