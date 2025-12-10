import pandas as pd
import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import pyinform
from OGW_DataLoader import OGW_data

"""
Conduct attribute causality mining before network training
After determining the attribute causality, the MANHAC structure is fixed
"""

def create_attribute_data(attribute_no, train_attributelabel, traindata):   # Consider the low value 0/1 and the high value 4/5 of each attribute
    label_attribute = train_attributelabel[:, attribute_no - 1]
    indices_attribute_5 = np.where(label_attribute == 5)[0]
    indices_attribute_4 = np.where(label_attribute == 4)[0]
    indices_attribute_pos = np.hstack((indices_attribute_5, indices_attribute_4))

    indices_attribute_0 = np.where(label_attribute == 0)[0]
    indices_attribute_1 = np.where(label_attribute == 1)[0]
    indices_attribute_neg = np.hstack((indices_attribute_0, indices_attribute_1))

    label_attribute_pos = label_attribute[indices_attribute_pos]
    label_attribute_pos = np.array([1] * label_attribute_pos.shape[0])

    label_attribute_neg = label_attribute[indices_attribute_neg]
    label_attribute_neg = np.array([0] * label_attribute_neg.shape[0])

    data_attribute_pos = traindata[indices_attribute_pos]
    data_attribute_neg = traindata[indices_attribute_neg]

    x = np.vstack((data_attribute_pos, data_attribute_neg))      # positive data + negative data
    y = np.hstack([label_attribute_pos, label_attribute_neg])    # labels

    return x, y

def discretize_data(data, bins=10):
    """
    Discretize continuous data into non-negative integers (states).
    """
    return np.digitize(data, bins=np.linspace(np.min(data), np.max(data), bins)) - 1

def Fea_LDA_extraction(attribute_no, train_attributelabel, traindata):
    lda = LinearDiscriminantAnalysis(n_components=1)
    x, y = create_attribute_data(attribute_no, train_attributelabel, traindata)
    Fea_LDA = lda.fit_transform(x, y)
    Fea_LDA = discretize_data(Fea_LDA)

    return Fea_LDA

def TE(Source_data):
    data = Source_data.Source_X
    attribute = Source_data.Source_z

    num_series = np.size(attribute, axis=1)
    te_matrix = np.zeros((num_series, num_series))
    te_diff_matrix = np.zeros((num_series, num_series))

    for i in range(num_series):
        for j in range(num_series):
            if i != j:
                feature1 = Fea_LDA_extraction(i, attribute, data)
                feature2 = Fea_LDA_extraction(j, attribute, data)
                feature_size = np.min([np.size(feature1, axis=0), np.size(feature2, axis=0)])
                feature1 = feature1[0:feature_size, :]
                feature2 = feature2[0:feature_size, :]
                te_matrix[i, j] = pyinform.transfer_entropy(feature1.T, feature2.T, k=1)

                TE_i_to_j = pyinform.transfer_entropy(feature1.T, feature2.T, k=1)
                TE_j_to_i = pyinform.transfer_entropy(feature2.T, feature1.T, k=1)
                te_diff_matrix[i, j] = TE_i_to_j - TE_j_to_i

    return te_matrix, te_diff_matrix