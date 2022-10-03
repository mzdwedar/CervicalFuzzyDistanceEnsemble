import numpy as np

from sklearn.metrics import confusion_matrix , accuracy_score, roc_auc_score
from sklearn.metrics import precision_score, recall_score, f1_score

def compute_metrics(model_name, Y, preds):
  y_preds=[]

  for pred in preds: 
    y_preds.append(np.argmax(pred))

  print(f'{model_name} Accuracy Score: ',accuracy_score(Y,y_preds))
  n = len(precision_score(Y,y_preds , average= None))
  print(f'{model_name} Precision Score(Class wise): ',precision_score(Y, y_preds, average=None), " mean- " , sum(precision_score(Y, y_preds, average= None ))/n)
  print(f'{model_name} Recall Score(Class wise): ',recall_score(Y, y_preds, average=None), " mean- " , sum(recall_score(Y, y_preds, average= None ))/n)
  print(f'{model_name} F1 Score(Class wise): ',f1_score(Y, y_preds, average=None), " mean- " , sum(f1_score(Y, y_preds, average= None))/n)
  print(f'{model_name} Conf Matrix Score(Class wise):\n ',confusion_matrix(Y, y_preds))    
  print(f'{model_name} AUC ROC(Class wise): ',roc_auc_score(Y, y_preds, average= 'weighted'), " mean- " , sum(roc_auc_score(Y,y_preds , average= 'weighted'))/n)

def predict(model, dataset, num_examples, batch_size):
  
  ds = dataset.unbatch()
  ds = ds.batch(num_examples)  
  
  X = []
  for images, _ in ds.take(1):
    X = images.numpy()

  preds = model.predict(X, batch_size=batch_size)
  
  return preds