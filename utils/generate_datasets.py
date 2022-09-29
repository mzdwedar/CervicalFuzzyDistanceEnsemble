import os 
import pandas as pd 

def get_training_dataset(path):
    """
    the expected directory structure

            +-- train
            |   +-- NILM
            |   +-- ASC-US
            |   +-- ASC-H
            |   +-- LSIL
            |   +-- HSIL
            |   +-- SCC
            +-- val
            |   +-- NILM
            |   +-- ASC-US
            |   +-- ASC-H
            |   +-- LSIL
            |   +-- HSIL
            |   +-- SCC
    """
    print("generating training dataset")

    uniques = ["NILM" , "ASC-US" , "ASC-H" , "LSIL" , "HSIL", "SCC"]
    dirs = ["Train" , "Test"]
    data = {'train':[], 'val':[]}

    for dir in dirs :
        for unique in uniques:
            directory = path + "/" + dir + "/" + unique

            for filename in os.listdir(directory):
                paths = directory + "/" + filename
                data[dir].append([paths, unique])

    train_df = pd.DataFrame(data['Train'], columns = ["path", "class"])
    val_df = pd.DataFrame(data['val'], columns=["path", "class"])

    return train_df, val_df

def get_testing_dataset(path):
    '''
    the expected directory structure

            +-- test
            |   +-- NILM
            |   +-- ASC-US
            |   +-- ASC-H
            |   +-- LSIL
            |   +-- HSIL
            |   +-- SCC
    '''
    print("generating testing dataset")

    uniques = ["NILM" , "ASC-US" , "ASC-H" , "LSIL" , "HSIL", "SCC"]
    dirs = ["test"]
    data = []

    for dir in dirs:
        for unique in uniques:
            directory = os.path.join(path, dir, unique)

            for filename in os.listdir(directory):
                paths = os.path.join(directory, filename)
                data.append([paths, unique])
    
    test_df = pd.DataFrame(data, columns = ["path", "class"])

    return test_df