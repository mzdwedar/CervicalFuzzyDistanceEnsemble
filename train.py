import tensorflow as tf 
from sklearn.metrics import classification_report, accuracy_score

from utils.fuzzy_dist_ensemble import fuzzy_dist 
from utils.generate_datasets import get_training_dataset
from utils.model_utils import create_model, save_model
from utils.data_pipeline import parse_function, encode_y
from utils.model_eval import predict, compute_metrics

import argparse



def train(train_paths, val_paths, model_name1, model_name2, model_name3, train_batch, validation_batch, NUM_EPOCHS=70, lr=1e-4):

    num_examples_val = len(val_paths)
    num_examples_train = len(train_paths)
    y_true_val = val_paths['class'].apply(encode_y).to_numpy()

    train = tf.data.Dataset.from_tensor_slices(train_paths.values)
    val = tf.data.Dataset.from_tensor_slices(val_paths.values)

    AUTOTUNE = tf.data.AUTOTUNE
    train_dataset = (train
                      .cache()
                      .shuffle(buffer_size=len(train_paths))
                      .map(parse_function, num_parallel_calls=AUTOTUNE)
                      .batch(train_batch)
                      .repeat()
                      .prefetch(buffer_size=AUTOTUNE)
                    )

    val_dataset = (val
                    .map(parse_function)
                    .batch(validation_batch)
                    .prefetch(buffer_size=AUTOTUNE)
                  )
    steps_per_epoch = num_examples_train // train_batch
    validation_steps = num_examples_val // validation_batch

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
                         epochs=NUM_EPOCHS,
                         steps_per_epoch=steps_per_epoch,
                         validation_steps=validation_steps
                         ) 

    save_model(model1, model_name1, history1)

    # Generate generalization metrics
    preds1 = predict(model1, val_dataset, num_examples_val, validation_batch)
    compute_metrics(model_name1, y_true_val, preds1)

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
                         epochs=NUM_EPOCHS,
                         steps_per_epoch=steps_per_epoch,
                         validation_steps=validation_steps
                         )

    save_model(model2, model_name2, history2)

    # Generate generalization metrics
    preds2 = predict(model2, val_dataset, num_examples_val, validation_batch)
    compute_metrics(model_name2, y_true_val, preds2)

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
                         epochs=NUM_EPOCHS,
                         steps_per_epoch=steps_per_epoch,
                         validation_steps=validation_steps
                         )

    save_model(model3, model_name3, history3)

    # Generate generalization metrics
    preds3 = predict(model3, val_dataset, num_examples_val, validation_batch)
    compute_metrics(model_name3, y_true_val, preds3)

    # -------------------------------------------fuzzy distance ----------------------------------------
    ensem_preds=fuzzy_dist(preds1,preds2,preds3)
    
    print('Accuracy score: ', accuracy_score(y_true_val, ensem_preds))
    print(classification_report(y_true_val, ensem_preds, digits=4))
    

if __name__ == '__main__':

  parser = argparse.ArgumentParser()
  parser.add_argument('--num_epochs', type=int, default=70, 
                      help='number of epochs for training')
  parser.add_argument('--path', type=str, default='./',
                      help='Path where the image data is stored')
  parser.add_argument('--batch_size', type=int, default=16,
                      help='Batch Size for Mini Batch Training')
  parser.add_argument('--lr', type=float, default = 1e-4,
                      help='Learning rate for training')
  args = parser.parse_args()

  train_df, val_df = get_training_dataset(args.path)

  train(train_df, val_df, 
        "InceptionV3" , "MobileNetV2" ,"InceptionResNetV2",
        train_batch=args.batch_size, validation_batch = args.batch_size,
        NUM_EPOCHS = args.num_epochs, lr=args.lr)
  