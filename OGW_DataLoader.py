import numpy as np
from sklearn import preprocessing
from scipy.io import loadmat
import pandas as pd


class OGW_data():
    """
    For all the data, each 500 samples are in one working condition
    """
    def __init__(self, source, target, known_index, outlier_index, transfer_A_B):
        self.source = source
        self.target = target

        path = './OGW_mat_data/'

        data1 = loadmat(path + 'data_ow_bubble.mat')['data_ow_bubble']
        data2 = loadmat(path + 'data_ow_plug.mat')['data_ow_plug']
        data3 = loadmat(path + 'data_ow_slug.mat')['data_ow_slug']
        data4 = loadmat(path + 'data_ow_wave.mat')['data_ow_wave']
        data5 = loadmat(path + 'data_ow_st.mat')['data_ow_st']
        data6 = loadmat(path + 'data_ow_ann.mat')['data_ow_ann']
        data7 = loadmat(path + 'data_wo_bubble.mat')['data_wo_bubble']
        data8 = loadmat(path + 'data_wo_plug.mat')['data_wo_plug']
        data9 = loadmat(path + 'data_wo_slug.mat')['data_wo_slug']
        data10 = loadmat(path + 'data_wo_ann.mat')['data_wo_ann']

        attribute_matrix = pd.read_excel(path + 'OGW_attribute.xlsx', index_col='no')
        attribute_matrix = attribute_matrix.values

        known_index.sort()
        data_list = [data1, data2, data3, data4, data5, data6, data7, data8, data9, data10]

        # Domain A
        length_A = 400
        att_label_A_known = []
        label_A_known = []
        data_A_known = []
        att_label_A_outlier = []
        label_A_outlier = []
        data_A_outlier = []
        
        # known classes
        for item in known_index:
            label_index = known_index.index(item)
            att_label_A_known += [attribute_matrix[item, :]] * length_A
            label_A_known.append([[label_index]] * length_A)
            data_A_known.append(data_list[item][0:length_A])
        att_label_A_known = np.row_stack(att_label_A_known)
        label_A_known = np.row_stack(label_A_known)
        data_A_known = np.row_stack(data_A_known)
        
        # outlier classes
        for item in outlier_index:
            att_label_A_outlier += [attribute_matrix[item, :]] * length_A
            label_A_outlier.append([[len(known_index)]] * length_A)
            data_A_outlier.append(data_list[item][500:500 + length_A])
        att_label_A_outlier = np.row_stack(att_label_A_outlier)
        label_A_outlier = np.row_stack(label_A_outlier)
        data_A_outlier = np.row_stack(data_A_outlier)

        att_label_A = np.row_stack([att_label_A_known, att_label_A_outlier])
        label_A = np.row_stack([label_A_known, label_A_outlier])
        data_A = np.row_stack([data_A_known, data_A_outlier])
        
        # Domain B
        length_B = 300
        att_label_B_known = []
        label_B_known = []
        data_B_known = []
        att_label_B_outlier = []
        label_B_outlier = []
        data_B_outlier = []

        # known classes
        for item in known_index:
            label_index = known_index.index(item)
            att_label_B_known += [attribute_matrix[item, :]] * length_B
            label_B_known.append([[label_index]] * length_B)
            data_B_known.append(data_list[item][500:500 + length_B])
        att_label_B_known = np.row_stack(att_label_B_known)
        label_B_known = np.row_stack(label_B_known)
        data_B_known = np.row_stack(data_B_known)

        # outlier classes
        for item in outlier_index:
            att_label_B_outlier += [attribute_matrix[item, :]] * length_B
            label_B_outlier.append([[len(known_index)]] * length_B)
            data_B_outlier.append(data_list[item][500:500 + length_B])
        att_label_B_outlier = np.row_stack(att_label_B_outlier)
        label_B_outlier = np.row_stack(label_B_outlier)
        data_B_outlier = np.row_stack(data_B_outlier)

        att_label_B = np.row_stack([att_label_B_known, att_label_B_outlier])
        label_B = np.row_stack([label_B_known, label_B_outlier])
        data_B = np.row_stack([data_B_known, data_B_outlier])

        # Determine the source domain and target domain
        if transfer_A_B == 1:
            self.Source_X = data_A_known
            self.Source_y = label_A_known
            self.Source_z = att_label_A_known
            self.Target_X = data_B
            self.Target_y = label_B
            self.Target_z = att_label_B

        elif transfer_A_B == 2:
            self.Source_X = data_B_known
            self.Source_y = label_B_known
            self.Source_z = att_label_B_known
            self.Target_X = data_A
            self.Target_y = label_A
            self.Target_z = att_label_A

        # StandardScaler
        combined_data = np.vstack([self.Source_X, self.Target_X])
        scaler = preprocessing.StandardScaler().fit(combined_data)
        self.Source_X = scaler.transform(self.Source_X)
        self.Target_X = scaler.transform(self.Target_X)

    def __getitem__(self, index):
        if self.source:
            data = self.Source_X[index, :]
            label = self.Source_y[index]
            att_label = self.Source_z[index, :]
        else:
            data = self.Target_X[index, :]
            label = self.Target_y[index]
            att_label = self.Target_z[index, :]
        return data, int(label), att_label

    def __len__(self):
        if self.source:
            return len(self.Source_y)
        return len(self.Target_y)