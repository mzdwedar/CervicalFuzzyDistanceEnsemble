import os 
import pandas as pd 
from sklearn.utils import shuffle

def generate_csv(path):
    print("CSV being generated")
    uniques = ["NILM" , "ASC-US" , "ASC-H" , "LSIL" , "HSIL", "SCC"]
    dirs = ["Train" , "Test"]

    """
            +-- train
            |   +-- NILM
            |   +-- ASC-US
            |   +-- ASC-H
            |   +-- LSIL
            |   +-- HSIL
            |   +-- SCC

            +-- test
            |   +-- NILM
            |   +-- ASC-US
            |   +-- ASC-H
            |   +-- LSIL
            |   +-- HSIL
            |   +-- SCC

    
    """
    #Above is the expected directory structure

    # data = []
    data = {'Train':[], 'Test':[]}
    for dir in dirs :
        for unique in uniques:
            directory = path + "/" + dir + "/" + unique    #required path 

            for filename in os.listdir(directory):
               
                paths = directory + "/" + filename  #required path 
                data[dir].append([paths, unique])

    train_df = pd.DataFrame(data['Train'], columns = ["path", "class"])
    test_df = pd.DataFrame(data['Test'], columns=["path", "class"])

    # df = shuffle(df)
    # name = "csv_files/" + "Data-full"        #required path 
    # df.to_csv(name, index = False)
    # print("Generation Complete")
    return train_df, test_df
