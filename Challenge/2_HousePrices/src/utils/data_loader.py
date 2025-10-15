import pandas as pd

def load_data(data_config):
    df_train = pd.read_csv(data_config['train'])
    df_test = pd.read_csv(data_config['test'])
    return df_train, df_test
