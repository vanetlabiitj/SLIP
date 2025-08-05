import torch
import sklearn.metrics as metrics_sk
import numpy as np
from dataset.dataloader import CustomDataLoader
from models.shared_bottom.backbone.SharedBottomNet import SharedBottomNet
from models.shared_bottom.loss.RFL import RFL


class SLIPSupervisor:
    def __init__(self, **kwargs):
        self._data_kwargs = kwargs.get('data')
        self._model_kwargs = kwargs.get('model')
        self._train_kwargs = kwargs.get('train')
        self.input_dim = int(self._model_kwargs.get('input_dim', 1))
        self.output_dim = int(self._model_kwargs.get('output_dim', 1))
        self.loader = CustomDataLoader(**kwargs)
        self.trainloader, self.valloader, self.testloader = self.loader.load_dataset()
        SLIP_model = SharedBottomNet(**self._model_kwargs)
        print(SLIP_model)
        self.SLIP_model = SLIP_model
        self._epoch_num = self._train_kwargs.get('epoch', 0)

    def train(self, **kwargs):
        kwargs.update(self._train_kwargs)
        return self._train(**kwargs)

    def evaluate(self, valloader, **kwargs):

        with torch.no_grad():
            self.SLIP_model = self.SLIP_model.eval()
            all_predictions, all_labels, macro, micro = [], [], [], []
            loss_history, epoch_loss = [], []
            loss_t = 0
            criterion = RFL(gamma_neg=4, gamma_pos=0, clip=0.05, disable_torch_grad_focal_loss=True, **kwargs)
            for batch in valloader:
                x, labels = batch[0], batch[1]
                x_shape = x.shape
                x = x.reshape(x_shape[0], x_shape[1], -1)
                labels_shape = labels.shape
                labels = labels.reshape(labels_shape[0], -1)
                outputs = self.SLIP_model(x)
                loss = criterion(outputs, labels)
                loss_t += loss.item()
                predicted_labels = (outputs > 0.5).float()
                all_predictions.extend(predicted_labels.tolist())
                all_labels.extend(labels.tolist())

            loss_t /= len(valloader.dataset)
            loss_history.append(loss)
            all_labels_reshape = np.reshape(all_labels, (-1, 8))
            all_predictions_reshape = np.reshape(all_predictions, (-1, 8))
            macro_f1 = metrics_sk.f1_score(all_labels_reshape, all_predictions_reshape, average='macro')
            micro_f1 = metrics_sk.f1_score(all_labels_reshape, all_predictions_reshape, average='micro')

            return loss_t, macro_f1, micro_f1

    def _train(self, base_lr = 0.001, epochs=100, epsilon=1e-3, **kwargs):
        print('Start training ...')
        optimizer = torch.optim.Adam(self.SLIP_model.parameters(), lr=base_lr, eps=epsilon)
        criterion = RFL(gamma_neg=4, gamma_pos=0, clip=0.05, disable_torch_grad_focal_loss=True, **kwargs)
        all_predictions, all_labels, loss_history = [], [], []
        for epoch_num in range(self._epoch_num, epochs):
            self.SLIP_model = self.SLIP_model.train()
            correct_train, total_train, epoch_loss_train = 0, 0, 0.0

            for batch in self.trainloader:
                x, labels = batch[0], batch[1]
                optimizer.zero_grad()
                x_shape = x.shape
                x = x.reshape(x_shape[0], x_shape[1], -1)
                labels_shape = labels.shape
                labels = labels.reshape(labels_shape[0], -1)
                outputs = self.SLIP_model(x)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                epoch_loss_train += loss.item()
                predicted_labels = (outputs > 0.5).float()
                all_predictions.extend(predicted_labels.tolist())
                all_labels.extend(labels.tolist())

            loss_history.append(epoch_loss_train)
            epoch_loss_train /= len(self.trainloader)
            all_labels_reshape = np.reshape(all_labels, (-1, 8))
            all_predictions_reshape = np.reshape(all_predictions, (-1, 8))
            train_macro_f1 = metrics_sk.f1_score(all_labels_reshape, all_predictions_reshape, average='macro')
            train_micro_f1 = metrics_sk.f1_score(all_labels_reshape, all_predictions_reshape, average='micro')

            print(f"Epoch {epoch_num + 1}: train loss {epoch_loss_train}, macro F1 {train_macro_f1}, micro F1 {train_micro_f1}")

            val_loss, val_macro_f1, val_micro_f1 = self.evaluate(self.valloader, **kwargs)
            print(f"      : val loss {val_loss}, macro F1 {val_macro_f1}, micro F1 {val_micro_f1}")

        test_loss, test_macro, test_micro = self.evaluate(self.testloader, **kwargs)
        print(f"test loss {test_loss}, macro F1 {test_macro}, micro F1 {test_micro}")
