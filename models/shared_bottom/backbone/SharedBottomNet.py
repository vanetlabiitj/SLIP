import torch.nn as nn
from models.shared_bottom.backbone.TCN import TCNBlock
import torch


class SharedBottomNet(nn.Module):
    def __init__(self, **model_kwargs):
        super().__init__()
        self.num_inputs = int(model_kwargs.get('input_dim', 1))  # number of features
        self.num_outputs = int(model_kwargs.get('output_dim', 1))
        self.hidden_units = int(model_kwargs.get('hidden_units', 1))
        self.tower_hidden_units = int(model_kwargs.get('tower_hidden_units', 1))
        self.dropout_rate = float(model_kwargs.get('dropout_rate', 1))
        self.num_layers = int(model_kwargs.get('num_layers', 1))
        self.kernel_size = int(model_kwargs.get('kernel_size', 1))
        self.num_regions = int(model_kwargs.get('num_nodes', 1))
        self.e1 = int(model_kwargs.get('e1_size', 1))
        self.e2 = int(model_kwargs.get('e2_size', 1))
        self.e3 = int(model_kwargs.get('e3_size', 1))

        # Shared Bottom LSTM Layer
        # self.lstm = nn.GRU(
        #     input_size=num_inputs,
        #     hidden_size=hidden_units,
        #     batch_first=True,
        #     num_layers=self.num_layers
        # )

        layers = []
        for i in range(self.num_layers):
            in_channels = self.num_inputs if i == 0 else self.hidden_units
            layers.append(TCNBlock(in_channels, self.hidden_units, self.kernel_size, dilation=2**i))  # Increasing dilation
        self.tcn = nn.Sequential(*layers)

        # Tower Networks (3 Towers)
        self.tower1 = nn.Sequential(
            nn.Linear(self.hidden_units, self.tower_hidden_units),
            nn.ReLU(),
            nn.Dropout(self.dropout_rate),
            nn.Linear(self.tower_hidden_units, self.e1*8),
            nn.ReLU()
        )

        self.tower2 = nn.Sequential(
            nn.Linear(self.hidden_units, self.tower_hidden_units),
            nn.ReLU(),
            nn.Dropout(self.dropout_rate),
            nn.Linear(self.tower_hidden_units, self.e2*8),
            nn.ReLU()
        )

        self.tower3 = nn.Sequential(
            nn.Linear(self.hidden_units, self.tower_hidden_units),
            nn.ReLU(),
            nn.Dropout(self.dropout_rate),
            nn.Linear(self.tower_hidden_units, self.e3*8),
            nn.ReLU()
        )

        # Final Linear Layer
        self.linear_final = nn.Linear(in_features=(self.e1*8 + self.e2*8 + self.e3*8), out_features=self.num_outputs)
        self.sigmoid = nn.Sigmoid()
        #self.linear_final = CosineClassifier(in_features=(113*8 + 37*8 + 38*8), out_features=904, scale=30)


        # Weight Initialization
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LSTM):
                for name, param in m.named_parameters():
                    if 'weight' in name:
                        nn.init.xavier_uniform_(param)
                    elif 'bias' in name:
                        nn.init.zeros_(param)

    def forward(self, x):
        x = x.permute(0, 2, 1)  # Change to (batch, channels, sequence_length) for TCN
        tcn_out = self.tcn(x)
        shared_out = tcn_out[:, :, -1]

        # Tower Networks
        out1 = self.tower1(shared_out)
        out2 = self.tower2(shared_out)
        out3 = self.tower3(shared_out)

        # Concatenation and Final Output
        concatenated_logits = torch.cat((out1, out2, out3), dim=1)
        out_final = self.linear_final(concatenated_logits)
        #out_final = self.sigmoid(out_final)
        return out_final