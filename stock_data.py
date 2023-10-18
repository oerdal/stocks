import pandas as pd
import numpy as np
import torch
import os
import logging, sys
from torch.utils.data import Dataset, DataLoader

from stock_language import ts_to_string, BPE, tokenize

logging.basicConfig(stream=sys.stderr, level=logging.DEBUG)

# Dataset of ALL stocks from a particular index (ex: NASDAQ)
class StockDataset(Dataset):
    def __init__(self, data_dir, close_only=True, as_tensor=True):
        self.data_dir = data_dir
        self.paths = os.listdir(self.data_dir)
        self.close_only = close_only
        self.as_tensor = as_tensor
        
    def __len__(self):
        return len(self.paths)
    
    def __getitem__(self, idx):
        ticker_path = os.path.join(self.data_dir, self.paths[idx])
        data = pd.read_csv(ticker_path)
        if self.close_only:
            data = data.loc[:,'Close']
        if self.as_tensor:
            data = torch.tensor(data)
        else:
            data = data.to_numpy()
        return self.paths[idx].split('.')[0], data


class StockLanguageDataset(Dataset):
    def __init__(self, data_dir, preprocess=True, close_only=True, as_language=True, mini_test=False):
        self.data_dir = data_dir
        self.paths = os.listdir(self.data_dir)
        self.close_only = close_only
        self.as_language = as_language

        if mini_test:
            self.paths = self.paths[:min(5, len(self.paths))]

        if as_language:
            self.SL_strings = []
            for path in self.paths:
                ##
                logging.info(f'Processing path {path}')
                ##
                path = os.path.join(self.data_dir, path)
                data = pd.read_csv(path)

                if preprocess:
                    data = data.dropna()
                
                if self.close_only:
                    data = data.loc[:,'Close']
                else:
                    # operation not yet supported
                    ...
                
                data = ts_to_string(data)
                self.SL_strings.append(data)
        else:
            self.SL_strings = []

    
    def __len__(self):
        return len(self.SL_strings)
    

    def __getitem__(self, idx):
        return self.SL_strings[idx]


def basicSLDatasetDemo():
    nasdaq = './stock_market_data/nasdaq/csv'
    nasdaq_dataset = StockLanguageDataset(nasdaq, mini_test=True)
    nasdaq_dataloader = DataLoader(nasdaq_dataset, batch_size=1, shuffle=False)

    # need to unbatch
    SL_strings = [SL_string[0] for SL_string in nasdaq_dataloader]
    tokenizer = BPE(SL_strings, n=500)
    print(tokenizer)

    
def main():
    basicSLDatasetDemo()


if __name__ == '__main__':
    main()