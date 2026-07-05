import torch.nn as nn
import torch.nn.functional as F

class FeatureExtractorTransformer(nn.Module):
    def __init__(self, dropout_rate=0):
        super(FeatureExtractorTransformer, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv1d(
                in_channels=2, out_channels=64, kernel_size=3, stride=1, padding=1
            ),
            nn.ReLU(),
            nn.Dropout1d(dropout_rate),
            nn.Conv1d(
                in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1
            ),
            nn.ReLU(),
            nn.Dropout1d(dropout_rate),
            nn.Conv1d(
                in_channels=128, out_channels=512, kernel_size=3, stride=1, padding=1
            ),
            nn.ReLU(),
            nn.Dropout1d(dropout_rate),
        )

        self.max_pool = nn.AdaptiveMaxPool1d(output_size=1)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=512,
            nhead=8,
            dim_feedforward=1024,
            dropout=dropout_rate,
            batch_first=True,
        )
        # num layers may be adjusted:
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=1)

        self.final_projection = nn.Linear(512, 128)

    def forward(self, x):
        x = self.conv_layers(x)

        x = self.max_pool(x)
        x = x.squeeze(-1)

        x = self.transformer(x)
        x = self.final_projection(x)
        x_norm = F.normalize(x, p=2, dim=1)
        return x_norm


class PRO_DSC(nn.Module):
    def __init__(self, hidden_dim, z_dim,  dropout_rate=0):
        super().__init__()
        self.pre_feature = FeatureExtractorTransformer(dropout_rate)
        self.subspace = nn.Sequential(
            nn.Linear(hidden_dim, z_dim)
        )
        self.cluster = nn.Sequential(
            nn.Linear(hidden_dim, z_dim)
        )

    def forward(self, x):
        pre_feature = self.pre_feature(x)
        pre_feature = pre_feature.view(pre_feature.shape[0], -1)
        Z = self.subspace(pre_feature)
        logits = self.cluster(pre_feature).float()
        Z = F.normalize(Z, 2)
        logits = F.normalize(logits, 2)

        return Z, logits

class PRO_DSC_Transformer(nn.Module):
    def __init__(self,input_dim, hidden_dim, z_dim, dropout_rate=0.2):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=512,
            nhead=8,
            dim_feedforward=1024,
            dropout=dropout_rate,
            batch_first=True,
        )
        self.pre_feature = nn.Sequential(nn.Linear(input_dim, hidden_dim),
                                         nn.BatchNorm1d(hidden_dim),
                                         nn.ReLU(),
                                         nn.TransformerEncoder(encoder_layer, num_layers=1),
                                         nn.Linear(512, hidden_dim),
                                         nn.ReLU(),
                                         )
        self.subspace = nn.Sequential(
            nn.Linear(hidden_dim, z_dim)
        )
        self.cluster = nn.Sequential(
            nn.Linear(hidden_dim, z_dim)
        )


    def forward(self, x):
        
        pre_feature = self.pre_feature(x)
        Z = self.subspace(pre_feature)
        logits = self.cluster(pre_feature).float()
        Z = F.normalize(Z, 2)
        logits = F.normalize(logits, 2)
        
        return Z, logits

    