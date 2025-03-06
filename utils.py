import torch
import numpy as np
import scipy.sparse as sp

from models.MainModel import MainModel


def setSeed(seed):
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    else:
        torch.manual_seed(seed)


def getData(df, columns, isToken, d):
    return torch.tensor(df[columns].values, dtype=torch.long if isToken else torch.float, device=d)


def getVocabulary(filePath):
    vocabulary = {}
    with open(filePath, 'r') as file:
        for line in file.readlines():
            line = line.replace('\n', '')
            [token, tokenID] = line.split(': ')
            tokenID = int(tokenID)

            vocabulary[token] = tokenID
    return len(vocabulary)


# 读取标签数据
def getLabels(filePath, d):
    nurseIndexes = []
    patientIndexes = []
    scoreList = []
    with open(filePath, 'r') as f:
        for line in f.readlines():
            line = line.replace('\n', '')
            [nurse2Patient, score] = line.split(': ')

            [nurse, patient] = nurse2Patient.split('-')
            score = float(score)

            nurseIndex = int(nurse)
            patientIndex = int(patient)

            nurseIndexes.append(nurseIndex)
            patientIndexes.append(patientIndex)
            scoreList.append(score)

    indexData = torch.tensor(np.row_stack((nurseIndexes, patientIndexes)), dtype=torch.long, device=d)
    scoreData = torch.tensor(np.array(scoreList), dtype=torch.float, device=d)

    return indexData, scoreData


def getEdges(filePath, d):
    srcIds, tarIds = [], []
    with open(filePath, 'r') as f:
        for line in f.readlines()[:-1]:
            line = line.replace('\n', '')
            [srcNode, tarNodes] = line.split(': ')

            srcNode = int(srcNode)
            tarNodes = map(int, tarNodes.split(', '))

            for tarNode in tarNodes:
                srcIds.append(srcNode)
                tarIds.append(tarNode)

    return torch.tensor(np.row_stack((srcIds, tarIds)), dtype=torch.long, device=d)


def getAdj(filePath, count, d, basedDist=False, filePath_dist=None):
    m = {}
    with open(filePath, 'r') as file:
        lines = file.readlines()[:-1]
        for line in lines:
            line = line.replace('\n', '')
            [src, tars] = line.split(': ')

            srcIndex = int(src)
            tarIndexes = list(map(int, tars.split(', ')))
            m[srcIndex] = tarIndexes

    # 邻接矩阵
    adj_matrix = []
    for idx in range(count):
        tmp = [0] * count
        if idx not in m.keys():
            adj_matrix.append(tmp)
        else:
            for j in m[idx]:
                tmp[j] = 1
            adj_matrix.append(tmp)

    # 添加自环
    for idx in range(count):
        adj_matrix[idx][idx] = 1

    # 度矩阵
    degree_matrix = [[0] * count for _ in range(count)]
    for idx in range(count):
        degree = sum(adj_matrix[idx])
        degree_matrix[idx][idx] = degree

    degree_matrix_np = np.array(degree_matrix)
    degree_matrix_sqrt_inv = np.linalg.inv(np.sqrt(degree_matrix_np))
    degree_matrix_sqrt_inv[np.isinf(degree_matrix_sqrt_inv)] = 0

    adj_matrix_np = np.array(adj_matrix)
    if basedDist:
        dist_matrix = []
        with open(filePath_dist, 'r') as distFile:
            lines = distFile.readlines()
            for line in lines:
                line = line.replace('\n', '')
                [_, dists] = line.split(': ')

                dists = list(map(float, dists.split(', ')))
                dist_matrix.append(dists)
        dist_matrix_np = np.array(dist_matrix)
        adj_matrix_np = np.multiply(dist_matrix_np, adj_matrix_np)

    normalized_adj_matrix_np = degree_matrix_sqrt_inv @ adj_matrix_np @ degree_matrix_sqrt_inv
    return torch.tensor(normalized_adj_matrix_np, dtype=torch.float, device=d)


def getAdj_NGCF(filePath, d):
    R = np.zeros((200, 1000), dtype=np.float32)
    with open(filePath, 'r') as file:
        for line in file.readlines():
            line = line.replace('\n', '')
            n2p = [int(i) for i in line.split(' ')]

            for p in n2p[1:]:
                R[n2p[0], p] = 1.

    adj_matrix = np.zeros((200 + 1000, 200 + 1000), dtype=np.float32)
    adj_matrix[:200, 200:] = R
    adj_matrix[200:, :200] = R.T

    # 归一化邻接矩阵
    def normalize(adj):
        row_sum = np.sum(adj, axis=1)

        d_inv = np.power(row_sum, -1)
        d_inv[np.isinf(d_inv)] = 0.
        d_mat_inv = np.diag(d_inv)

        norm_adj = np.dot(d_mat_inv, adj)
        return norm_adj

    norm_adj_matrix = normalize(adj_matrix + np.eye(adj_matrix.shape[0]))

    return torch.from_numpy(norm_adj_matrix).float().to(d)


def getModelConfig(
        n_count_nurse,
        n_count_patient,
        t_count,
        embedded_size,
        useHAT,
        hat_config_nurse,
        hat_config_patient,
        useGAT,
        gat_config_nurse,
        gat_config_patient,
        useGCN,
        gcn_config_nurse,
        gcn_config_patient):
    nurseModel_config = {
        'emb_config': {
            'n_count': n_count_nurse,
            't_count': t_count,
            'embedded_size': embedded_size
        },
        'useHAT': useHAT,
        'hat_config': hat_config_nurse,
        'useGAT': useGAT,
        'gat_config': gat_config_nurse,
        'useGCN': useGCN,
        'gcn_config': gcn_config_nurse
    }
    patientModel_config = {
        'emb_config': {
            'n_count': n_count_patient,
            't_count': t_count,
            'embedded_size': embedded_size
        },
        'useHAT': useHAT,
        'hat_config': hat_config_patient,
        'useGAT': useGAT,
        'gat_config': gat_config_patient,
        'useGCN': useGCN,
        'gcn_config': gcn_config_patient
    }
    return nurseModel_config, patientModel_config


def getModel(nurseModel_config, patientModel_config, isOnlyHAT):
    return MainModel(nurseModel_config, patientModel_config, mlp_config={
        'input_features': 96 + 96 if isOnlyHAT else 8 + 8,
        'hidden_features': 32,
    })
