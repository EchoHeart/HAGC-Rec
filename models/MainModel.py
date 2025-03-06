import torch
import torch.nn as nn


from models.Downstream.MLP import MLP
from models.UserModel import UserModel


class MainModel(nn.Module):
    def __init__(self, nurseModel_config, patientModel_config, mlp_config):
        super().__init__()

        self.nurseModel = UserModel(
            embeddingLayer_config=nurseModel_config['emb_config'],
            useHAT=nurseModel_config['useHAT'],
            hat_config=nurseModel_config['hat_config'],
            useGAT=nurseModel_config['useGAT'],
            gat_config=nurseModel_config['gat_config'],
            useGCN=nurseModel_config['useGCN'],
            gcn_config=nurseModel_config['gcn_config']
        )

        self.patientModel = UserModel(
            embeddingLayer_config=patientModel_config['emb_config'],
            useHAT=patientModel_config['useHAT'],
            hat_config=patientModel_config['hat_config'],
            useGAT=patientModel_config['useGAT'],
            gat_config=patientModel_config['gat_config'],
            useGCN=patientModel_config['useGCN'],
            gcn_config=patientModel_config['gcn_config']
        )

        self.MLP = MLP(
            mlp_config['input_features'],
            mlp_config['hidden_features'],
        )

    def forward(self, n_data, p_data, indexes):
        n_after_model = self.nurseModel(n_data)
        p_after_model = self.patientModel(p_data)

        n = n_after_model.index_select(dim=0, index=indexes[0])
        p = p_after_model.index_select(dim=0, index=indexes[1])

        return self.MLP(torch.cat((n, p), dim=1))
