import pandas as pd
import numpy as np
import torch
import utils.pre_process
from torch.utils.data import DataLoader, Dataset


class CustomDataset(Dataset):
    def __init__(self, features, labels):
        self.features = features
        self.labels = labels

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        #feature = torch.tensor(self.features[idx], dtype=torch.float32)
        feature = torch.from_numpy(np.array(self.features[idx])).float()
        #label = torch.tensor(self.labels[idx], dtype=torch.float32)
        label = torch.from_numpy(np.array(self.labels[idx])).float()
        return feature, label


class CustomDataLoader:
    def __init__(self, **kwargs):
        self._data_kwargs = kwargs.get('data')
        self.batch_size = int(self._data_kwargs.get('batch_size', 1))
        self.path = str(kwargs.get('base_dir'))
        self.file_name = str(kwargs.get('file_name'))
        self.city = str(kwargs.get('city'))

    def load_dataset(self, **kwargs):

        df = pd.read_csv(self.path + self.file_name)

        if self.city == 'LA':
            feature, label = utils.pre_process.choose_target_generate_fllist_LA(df)
        else:
            feature, label = utils.pre_process.choose_target_generate_fllist_CHI(df)

        num_samples = len(feature)

        # Calculate the sizes for training, validation, test, and global test sets
        num_train = round(num_samples * 0.5417)  # 65% for training
        num_val = round(num_samples * 0.0417)  # 5% for validation
        num_test = round(num_samples * 0.0833)  # 10% for testing

        # Training set
        x_train, target_train = feature[:num_train], label[:num_train]

        # Validation set
        x_val, target_val = feature[num_train:num_train + num_val], label[num_train:num_train + num_val]

        # Test set
        x_test, target_test = feature[num_train + num_val: num_train + num_val + num_test], label[
                                                                                            num_train + num_val: num_train + num_val + num_test]

        train_dataset = CustomDataset(x_train, target_train)
        val_dataset = CustomDataset(x_val, target_val)
        test_dataset = CustomDataset(x_test, target_test)

        trainloaders = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=False)
        valloaders = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False)
        testloaders = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False)

        return trainloaders, valloaders, testloaders
