import pandas as pd
import torch.nn as nn
import torch.optim as optim

from utils import *
from models.NGCF.NGCF import NGCF
from models.MF.MF import MF

from matplotlib import pyplot as plt
from datetime import datetime

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    time = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')

    # 读取护工、患者数据
    nurseData_train = pd.read_csv('data/inductive/train/nurseData.csv')
    patientData_train = pd.read_csv('data/inductive/train/patientData.csv')

    nurseData_val = pd.read_csv('data/inductive/val/nurseData.csv')
    patientData_val = pd.read_csv('data/inductive/val/patientData.csv')

    nurseData = pd.read_csv('data/nurseData_200.csv')
    patientData = pd.read_csv('data/patientData_1000.csv')

    tokenCount = getVocabulary('data/token2Id.txt')

    # 标签数据
    indexData_train_inductive, scoreData_train_inductive = getLabels('data/inductive/train/labels.txt', device)
    indexData_val_inductive, scoreData_val_inductive = getLabels('data/inductive/val/labels.txt', device)

    indexData_train, scoreData_train = getLabels('data/train/trainLabels_200_1000_90.txt', device)
    indexData_val, scoreData_val = getLabels('data/val/valLabels_200_1000_10.txt', device)

    """
    离散型、连续型特征分类
    """
    columnToken_common = ['性别', '省', '民族', '宗教信仰', '学历', '体型']
    columnToken_nurse = ['常驻医院', '主要负责病区', '其他负责病区']
    columnToken_patient = ['过敏史', '是否残疾', '暴露史', '是否吸烟', '是否饮酒', '常饮酒类', '所属医院', '所在病区']

    columnNumber_common = ['年龄', '身高', '体重', 'BMI指数']
    columnNumber_nurse = ['普通话程度', '方言程度', '在岗时间', '擅长科室种类', '接单数', '活跃度', '评分']
    columnNumber_patient = ['烟龄', '平均每日吸烟次数', '平均每日饮酒次数', '锻炼频率', '刷牙频率', '重症等级']

    nurseColumns = columnNumber_common + columnNumber_nurse + columnToken_common + columnToken_nurse
    patientColumns = columnNumber_common + columnNumber_patient + columnToken_common + columnToken_patient

    nurseData_token_train = getData(nurseData_train, columnToken_common + columnToken_nurse, True, device)
    nurseData_number_train = getData(nurseData_train, columnNumber_common + columnNumber_nurse, False, device)
    patientData_token_train = getData(patientData_train, columnToken_common + columnToken_patient, True, device)
    patientData_number_train = getData(patientData_train, columnNumber_common + columnNumber_patient, False, device)

    nurseData_token_val = getData(nurseData_val, columnToken_common + columnToken_nurse, True, device)
    nurseData_number_val = getData(nurseData_val, columnNumber_common + columnNumber_nurse, False, device)
    patientData_token_val = getData(patientData_val, columnToken_common + columnToken_patient, True, device)
    patientData_number_val = getData(patientData_val, columnNumber_common + columnNumber_patient, False, device)

    nurseData_token = getData(nurseData, columnToken_common + columnToken_nurse, True, device)
    nurseData_number = getData(nurseData, columnNumber_common + columnNumber_nurse, False, device)
    patientData_token = getData(patientData, columnToken_common + columnToken_patient, True, device)
    patientData_number = getData(patientData, columnNumber_common + columnNumber_patient, False, device)

    # 获取边数据
    edgeData_nurse_train = getEdges('data/inductive/train/edge_nurse_0.9_dist.txt', device)
    edgeData_nurse_val = getEdges('data/inductive/val/edge_nurse_0.9_dist.txt', device)
    edgeData_patient_train = getEdges('data/inductive/train/edge_patient_0.9_dist.txt', device)
    edgeData_patient_val = getEdges('data/inductive/val/edge_patient_0.9_dist.txt', device)

    adjData_nurse = getAdj('data/edge/200_1000/edge_nurse_0.9_dist.txt', nurseData.shape[0], device)
    adjData_patient = getAdj('data/edge/200_1000/edge_patient_0.9_dist.txt', patientData.shape[0], device)
    # adjData_nurse_dist = getAdj('data/edge/200_1000/edge_nurse_0.9_dist.txt', nurseData.shape[0], device,
    #                             basedDist=True, filePath_dist='data/dist_matrix_nurse.txt')
    # adjData_patient_dist = getAdj('data/edge/200_1000/edge_patient_0.9_dist.txt', patientData.shape[0], device,
    #                               basedDist=True, filePath_dist='data/dist_matrix_patient.txt')

    adjData = getAdj_NGCF('data/n2p.txt', device)

    HAT_config_nurse = {
        'num_HAT_layers': 3,
        'in_features': [8, 16, 16],
        'out_features': [16, 16, 32],
        'num_softmax_layers': [5, 3, 1],
        'all_dims': [len(nurseColumns), 5, 3],
        'softmax_dims': [
            [[0, 1, 2, 3, 11, 12, 13, 14, 15, 16], [0, 3, 16], [4, 5], [6, 7, 8, 9, 10, 15], [17, 18, 19]],
            [[0], [1, 2, 3], [4]],
            [[0, 1, 2]]
        ]
    }
    HAT_config_patient = {
        'num_HAT_layers': 3,
        'in_features': [8, 16, 16],
        'out_features': [16, 16, 32],
        'num_softmax_layers': [5, 3, 1],
        'all_dims': [len(patientColumns), 5, 3],
        'softmax_dims': [
            [[0, 1, 2, 3, 10, 11, 12, 13, 14, 15], [0, 3, 9, 15, 17], [16, 18], [4, 5, 6, 7, 8, 19, 20, 21], [22, 23]],
            [[0], [1, 2, 3], [4]],
            [[0, 1, 2]]
        ]
    }

    GAT_config = {
        'num_of_layers': 2,
        'num_heads_per_layer': [8, 1],
        'num_features_per_layer': [96, 8, 8],
        'dropout_prob': 0.6,
        'bias': True,
        'add_skip_connection': False
    }
    GAT_config_nurse = {
        'num_of_layers': 2,
        'num_heads_per_layer': [8, 1],
        'num_features_per_layer': [len(nurseColumns) * 8, 8, 8],
        'dropout_prob': 0.6,
        'bias': True,
        'add_skip_connection': False
    }
    GAT_config_patient = {
        'num_of_layers': 2,
        'num_heads_per_layer': [8, 1],
        'num_features_per_layer': [len(patientColumns) * 8, 8, 8],
        'dropout_prob': 0.6,
        'bias': True,
        'add_skip_connection': False
    }

    GCN_config = {
        'input_features': 96,
        'hidden_features': 16,
        'out_features': 8,
        'dropout': 0.5
    }
    GCN_config_nurse = {
        'input_features': len(nurseColumns) * 8,
        'hidden_features': 16,
        'out_features': 8,
        'dropout': 0.5
    }
    GCN_config_patient = {
        'input_features': len(patientColumns) * 8,
        'hidden_features': 16,
        'out_features': 8,
        'dropout': 0.5
    }

    models, optimizers, lossFuncs = [], [], []

    # # only HAT
    # setSeed(2024)
    # onlyHAT = getModelConfig(n_count_nurse=nurseData_number.shape[1], n_count_patient=patientData_number.shape[1],
    #                          t_count=tokenCount, embedded_size=8,
    #                          useHAT=True, hat_config_nurse=HAT_config_nurse, hat_config_patient=HAT_config_patient,
    #                          useGAT=False, gat_config_nurse=None, gat_config_patient=None,
    #                          useGCN=False, gcn_config_nurse=None, gcn_config_patient=None)
    # model = getModel(onlyHAT[0], onlyHAT[1], isOnlyHAT=True).to(device)
    # optimizer = optim.Adam(model.parameters(), lr=5e-3, weight_decay=5e-4)
    # criterion = nn.L1Loss()
    #
    # models.append(model)
    # optimizers.append(optimizer)
    # lossFuncs.append(criterion)

    # # only GAT
    # setSeed(2024)
    # onlyGAT = getModelConfig(n_count_nurse=nurseData_number_train.shape[1],
    #                          n_count_patient=patientData_number_train.shape[1],
    #                          t_count=tokenCount, embedded_size=8,
    #                          useHAT=False, hat_config_nurse=None, hat_config_patient=None,
    #                          useGAT=True, gat_config_nurse=GAT_config_nurse, gat_config_patient=GAT_config_patient,
    #                          useGCN=False, gcn_config_nurse=None, gcn_config_patient=None)
    # model = getModel(onlyGAT[0], onlyGAT[1], isOnlyHAT=False).to(device)
    # optimizer = optim.Adam(model.parameters(), lr=5e-3, weight_decay=5e-4)
    # criterion = nn.L1Loss()
    #
    # models.append(model)
    # optimizers.append(optimizer)
    # lossFuncs.append(criterion)

    # # only GCN
    # setSeed(2024)
    # onlyGCN = getModelConfig(n_count_nurse=nurseData_number.shape[1], n_count_patient=patientData_number.shape[1],
    #                          t_count=tokenCount, embedded_size=8,
    #                          useHAT=False, hat_config_nurse=None, hat_config_patient=None,
    #                          useGAT=False, gat_config_nurse=None, gat_config_patient=None,
    #                          useGCN=True, gcn_config_nurse=GCN_config_nurse, gcn_config_patient=GCN_config_patient)
    # model = getModel(onlyGCN[0], onlyGCN[1], isOnlyHAT=False).to(device)
    # optimizer = optim.Adam(model.parameters(), lr=5e-3, weight_decay=5e-4)
    # criterion = nn.L1Loss()
    #
    # models.append(model)
    # optimizers.append(optimizer)
    # lossFuncs.append(criterion)

    # # HAT + GAT
    # setSeed(2024)
    # HGARec = getModelConfig(n_count_nurse=nurseData_number_train.shape[1],
    #                         n_count_patient=patientData_number_train.shape[1],
    #                         t_count=tokenCount, embedded_size=8,
    #                         useHAT=True, hat_config_nurse=HAT_config_nurse, hat_config_patient=HAT_config_patient,
    #                         useGAT=True, gat_config_nurse=GAT_config, gat_config_patient=GAT_config,
    #                         useGCN=False, gcn_config_nurse=None, gcn_config_patient=None)
    # model = getModel(HGARec[0], HGARec[1], isOnlyHAT=False).to(device)
    # optimizer = optim.Adam(model.parameters(), lr=5e-3, weight_decay=5e-4)
    # criterion = nn.L1Loss()
    #
    # models.append(model)
    # optimizers.append(optimizer)
    # lossFuncs.append(criterion)

    # # HAT + GCN
    # setSeed(2024)
    # HGCRec = getModelConfig(n_count_nurse=nurseData_number.shape[1], n_count_patient=patientData_number.shape[1],
    #                         t_count=tokenCount, embedded_size=8,
    #                         useHAT=True, hat_config_nurse=HAT_config_nurse, hat_config_patient=HAT_config_patient,
    #                         useGAT=False, gat_config_nurse=None, gat_config_patient=None,
    #                         useGCN=True, gcn_config_nurse=GCN_config, gcn_config_patient=GCN_config)
    # model = getModel(HGCRec[0], HGCRec[1], isOnlyHAT=False).to(device)
    # optimizer = optim.Adam(model.parameters(), lr=5e-3, weight_decay=5e-4)
    # criterion = nn.L1Loss()
    #
    # models.append(model)
    # optimizers.append(optimizer)
    # lossFuncs.append(criterion)

    # # NGCF
    # setSeed(2024)
    # model = NGCF(n_users=200, n_items=1000, embedding_dim=8, weight_size=[64], dropout_list=[0.1], mlp_config={
    #     'input_features': 72 + 72,
    #     'hidden_features': 32
    # }).to(device)
    # optimizer = optim.Adam(model.parameters(), lr=5e-3, weight_decay=5e-4)
    # criterion = nn.L1Loss()
    #
    # models.append(model)
    # optimizers.append(optimizer)
    # lossFuncs.append(criterion)

    # MF
    setSeed(2024)
    model = MF(n_users=200, n_items=1000, embedding_dim=8, mlp_config={
        'input_features': 8 + 8,
        'hidden_features': 32
    }).to(device)
    optimizer = optim.Adam(model.parameters(), lr=5e-3, weight_decay=5e-4)
    criterion = nn.L1Loss()

    models.append(model)
    optimizers.append(optimizer)
    lossFuncs.append(criterion)

    losses_val = []
    needSave = True
    epochs, step, MAX_PATIENCE_CNT = 10000, 200, 1000
    modelNames = ['HAT', 'GAT', 'GCN', 'HGA-Rec', 'HGC-Rec']
    for i in range(len(models)):
        BEST_VAL_LOSS, PATIENCE_CNT = 5, 0
        lossValues_val = []

        for epoch in range(1, epochs + 1):
            models[i].train()
            # outputs_train = models[i]((nurseData_number, nurseData_token, adjData_nurse),
            #                           (patientData_number, patientData_token, adjData_patient), indexData_train)
            # loss_train = lossFuncs[i](outputs_train, scoreData_train.view(-1, 1))
            # outputs_train = models[i]((nurseData_number_train, nurseData_token_train, edgeData_nurse_train),
            #                           (patientData_number_train, patientData_token_train, edgeData_patient_train),
            #                           indexData_train_inductive)
            # loss_train = lossFuncs[i](outputs_train, scoreData_train_inductive.view(-1, 1))
            if i == 1 or i == 3:
                """
                GAT, HAT + GAT
                """
                outputs_train = models[i]((nurseData_number_train, nurseData_token_train, edgeData_nurse_train),
                                          (patientData_number_train, patientData_token_train, edgeData_patient_train),
                                          indexData_train_inductive)
                loss_train = lossFuncs[i](outputs_train, scoreData_train_inductive.view(-1, 1))
            elif i == 0:
                """
                HAT
                """
                # outputs_train = models[i]((nurseData_number, nurseData_token), (patientData_number, patientData_token),
                #                           indexData_train)
                # loss_train = lossFuncs[i](outputs_train, scoreData_train.view(-1, 1))

                # # for NGCF
                # outputs_train = models[i](adjData, indexData_train)
                # loss_train = lossFuncs[i](outputs_train, scoreData_train.view(-1, 1))

                # for MF
                outputs_train = models[i](indexData_train)
                loss_train = lossFuncs[i](outputs_train, scoreData_train.view(-1, 1))
            else:
                """
                GCN, HAT + GCN
                """
                outputs_train = models[i]((nurseData_number, nurseData_token, adjData_nurse),
                                          (patientData_number, patientData_token, adjData_patient), indexData_train)
                loss_train = lossFuncs[i](outputs_train, scoreData_train.view(-1, 1))
            lossValue_train = loss_train.item()

            optimizers[i].zero_grad()
            loss_train.backward()
            optimizers[i].step()

            with torch.no_grad():
                # models[i].eval()
                # outputs_val = models[i]((nurseData_number, nurseData_token, adjData_nurse),
                #                         (patientData_number, patientData_token, adjData_patient), indexData_val)
                # loss_val = lossFuncs[i](outputs_val, scoreData_val.view(-1, 1))
                # outputs_val = models[i]((nurseData_number_val, nurseData_token_val, edgeData_nurse_val),
                #                         (patientData_number_val, patientData_token_val, edgeData_patient_val),
                #                         indexData_val_inductive)
                # loss_val = lossFuncs[i](outputs_val, scoreData_val_inductive.view(-1, 1))
                if i == 1 or i == 3:
                    """
                    GAT, HAT + GAT
                    """
                    outputs_val = models[i]((nurseData_number_val, nurseData_token_val, edgeData_nurse_val),
                                            (patientData_number_val, patientData_token_val, edgeData_patient_val),
                                            indexData_val_inductive)
                    loss_val = lossFuncs[i](outputs_val, scoreData_val_inductive.view(-1, 1))
                elif i == 0:
                    """
                    HAT
                    """
                    # outputs_val = models[i]((nurseData_number, nurseData_token),
                    #                         (patientData_number, patientData_token),
                    #                         indexData_val)
                    # loss_val = lossFuncs[i](outputs_val, scoreData_val.view(-1, 1))

                    # # for NGCF
                    # outputs_val = models[i](adjData, indexData_val)
                    # loss_val = lossFuncs[i](outputs_val, scoreData_val.view(-1, 1))

                    # for MF
                    outputs_val = models[i](indexData_val)
                    loss_val = lossFuncs[i](outputs_val, scoreData_val.view(-1, 1))
                else:
                    """
                    GCN, HAT + GCN
                    """
                    outputs_val = models[i]((nurseData_number, nurseData_token, adjData_nurse),
                                            (patientData_number, patientData_token, adjData_patient), indexData_val)
                    loss_val = lossFuncs[i](outputs_val, scoreData_val.view(-1, 1))
                lossValue_val = loss_val.item()

                if lossValue_val < BEST_VAL_LOSS:
                    if needSave:
                        torch.save(models[i].state_dict(), f'bestModel_{modelNames[i]}_{time}.pth')
                    BEST_VAL_LOSS = lossValue_val
                    PATIENCE_CNT = 0
                else:
                    PATIENCE_CNT += 1

                lossValues_val.append(lossValue_val)
                if epoch % step == 0:
                    print(
                        f'Model {modelNames[i]}, '
                        f'Epoch {epoch}/{epochs}, '
                        f'Train Loss: {lossValue_train:.4f}, '
                        f'Val Loss: {lossValue_val:.4f}, '
                        f'Best Val Loss: {BEST_VAL_LOSS}')

                if PATIENCE_CNT >= MAX_PATIENCE_CNT:
                    break

        losses_val.append(lossValues_val)
        print()

    # 创建折线图
    fig = plt.figure(figsize=(10, 6))  # 设置图形大小

    colors = ['blue', 'orange', 'red', 'green', 'purple']
    for i in range(len(models)):
        plt.plot(losses_val[i], color=colors[i], linestyle='-', label=f'Model {modelNames[i]} Val Loss')

    plt.title('Loss Change with Epoch 10000')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.legend()

    if needSave:
        plt.savefig(f'loss_val_{time}.png')
    plt.show()
