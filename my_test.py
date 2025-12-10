import torch
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
import numpy as np
import torch.nn.functional as F
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

use_gpu = False     # True

def plot_normalized_confusion_matrix(ground_truth, predictions, output_size, title_suffix=""):

    class_names = ['State5', 'State6', 'State8', 'State9']
    class_names.append('Unknown')  # The last is the outlier class

    # confusion matrix
    cm = confusion_matrix(ground_truth, predictions)
    cm_normalized = cm.astype('float') / np.maximum(cm.sum(axis=1)[:, np.newaxis], 1)
    cm_normalized_df = pd.DataFrame(cm_normalized, index=class_names, columns=class_names)

    sns.set_context("notebook", font_scale=1.2)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm_normalized_df, annot=True, fmt='.4f', cmap='Blues',
                cbar=True, square=True,
                linewidths=0.5, linecolor='gray')

    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.xticks(rotation=0)
    plt.yticks(rotation=0)
    plt.tight_layout()

    plt.show()

    return cm, cm_normalized, class_names

def per_class_accuracy(y_true, y_pred):
    unique_classes = np.unique(y_true)
    class_accuracies = {}

    for class_label in unique_classes:
        class_true = (y_true == class_label)
        class_pred = (y_pred == class_label)

        # the accuracy rate of this category
        correct_predictions = np.sum(class_true & class_pred)
        total_class_samples = np.sum(class_true)

        if total_class_samples > 0:
            class_accuracy = correct_predictions / total_class_samples
        else:
            class_accuracy = 0.0

        class_accuracies[class_label] = class_accuracy

    return class_accuracies

def test(epoch, feature_extractor_global, classifier, AFE1,AFE2, AFE3, ATA2, ATA3, AP1, AP2, AP3, source_dataloader, target_dataloader, para, output_size):
    """
    Test the performance of the model
    :param feature_extractor_global: Network used to extract global feature from target samples
    :param classifier: Network used to classify samples
    :param AFE: Attribute feature extractor
    :param AP: Attribute predictor
    :param ATA: Attribute causality assistance
    :param source_dataloader: Test dataloader of source domain
    :param target_dataloader: Test dataloader of target domain
    :return: None
    """
    # setup the network
    feature_extractor_global.eval()
    classifier.eval()
    AFE1.eval()
    AFE2.eval()
    AFE3.eval()
    ATA2.eval()
    ATA3.eval()
    AP1.eval()
    AP2.eval()
    AP3.eval()

    # Source domain
    ground_truth_source = []
    predict_source = []
    for batch_idx, sdata in enumerate(source_dataloader):
        input1, label1, label_att_A = sdata
        input1 = input1.unsqueeze(1)
        input1 = input1.float()

        # Extract the global features of the source domain
        src_feature_global = feature_extractor_global(input1)

        # Extract the attribute features of the source domain
        src_feature_att1 = AFE1(input1)
        src_feature_att2 = AFE2(input1)
        src_feature_att3 = AFE3(input1)

        # ACA module
        src_ATA2 = ATA2(src_feature_att1)
        src_ATA3 = ATA3(src_feature_att2)

        # Attribute prediction of the source domain
        src_pred_att1 = AP1(src_feature_att1)
        src_pred_att2 = AP2(src_feature_att2 + para*src_ATA2)
        src_pred_att3 = AP3(src_feature_att3 + para*src_ATA3)
        src_attribute_result = torch.cat((src_pred_att1, src_pred_att2, src_pred_att3), dim=1)

        # Calcultate class logits for identifying source classes
        src_class_logits = classifier(src_feature_global.type(torch.float), src_attribute_result.type(torch.float))
        src_pred_class = F.log_softmax(src_class_logits, 1)
        pred1_class = src_pred_class.data.max(1, keepdim = True)[1].squeeze().tolist()

        ground_truth_source += list(label1.data.cpu().numpy().squeeze())
        predict_source += pred1_class

    # Target domain
    ground_truth_target = []
    predict_target = []
    for batch_idx, tdata in enumerate(target_dataloader):
        input2, label2, label_att_B = tdata
        input2 = input2.unsqueeze(1)
        input2 = input2.float()

        # Extract the global features of the target domain
        tgt_feature_global = feature_extractor_global(input2)

        # Extract the attribute features of the target domain
        tgt_feature_att1 = AFE1(input2)
        tgt_feature_att2 = AFE2(input2)
        tgt_feature_att3 = AFE3(input2)

        # ACA module
        tgt_ATA2 = ATA2(tgt_feature_att1)
        tgt_ATA3 = ATA3(tgt_feature_att2)

        # Attribute prediction of the target domain
        tgt_pred_att1 = AP1(tgt_feature_att1)
        tgt_pred_att2 = AP2(tgt_feature_att2 + para*tgt_ATA2)
        tgt_pred_att3 = AP3(tgt_feature_att3 + para*tgt_ATA3)
        tgt_attribute_result = torch.cat((tgt_pred_att1, tgt_pred_att2, tgt_pred_att3), dim=1)

        # Calcultate class logits for identifying target classes
        tgt_class_logits = classifier(tgt_feature_global.type(torch.float), tgt_attribute_result.type(torch.float))
        tgt_pred_class = F.log_softmax(tgt_class_logits, 1)
        pred2_class = tgt_pred_class.data.max(1, keepdim = True)[1].squeeze().tolist()

        ground_truth_target += list(label2.data.cpu().numpy().squeeze())
        predict_target += pred2_class

    source_acc = accuracy_score(ground_truth_source, predict_source)
    source_f1 = f1_score(ground_truth_source, predict_source, average='weighted')

    # The prediction accuracy of the known classes and the outlier class
    list1, list2 = zip(*[(x, y) for x, y in zip(ground_truth_target, predict_target) if x != output_size-1])
    new_list1, new_list2 = zip(*[(x, y) for x, y in zip(ground_truth_target, predict_target) if x == output_size-1])

    ground_truth_target_known = list(list1)
    predict_target_known = list(list2)
    ground_truth_target_outlier = list(new_list1)
    predict_target_outlier = list(new_list2)

    target_acc_known = accuracy_score(ground_truth_target_known, predict_target_known)
    target_f1_known = f1_score(ground_truth_target_known, predict_target_known, average='weighted')

    target_acc_outlier = accuracy_score(ground_truth_target_outlier, predict_target_outlier)
    target_f1_outlier = f1_score(ground_truth_target_outlier, predict_target_outlier, average='weighted')

    overall_acc = (target_acc_known * len(predict_target_known) + target_acc_outlier * len(
        ground_truth_target_outlier)) / (len(predict_target_known) + len(ground_truth_target_outlier))

    print('Target Accuracy_known: ({:.4f}%)\nTarget Accuracy_outlier: ({:.4f}%)\nOverall_Accuracy: ({:.4f}%)'.format(
        100. * target_acc_known, 100. * target_acc_outlier, 100. * overall_acc))

    class_accuracies_known = per_class_accuracy(np.array(ground_truth_target_known), np.array(predict_target_known))
    for class_label, accuracy in class_accuracies_known.items():
        print(f"class {class_label}: accuracy = {100. * accuracy:.4f}")

    # if epoch == 299:
    #     print("\n=== Normalized Confusion Matrix with Outlier Class ===")
    #     cm_complete, cm_normalized, class_names = plot_normalized_confusion_matrix(
    #         ground_truth_target, predict_target, output_size, " - Target Domain")

    return source_acc, source_f1, target_acc_known, target_f1_known, target_acc_outlier, target_f1_outlier