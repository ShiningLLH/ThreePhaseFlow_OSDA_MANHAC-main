import torch.nn as nn
import torch.optim as optim
import networks
from my_test import test
from utils import *
from torch.utils.data import DataLoader
import torch
from OGW_DataLoader import OGW_data
import random
import numpy as np
import torch.nn.functional as F
import matplotlib.pyplot as plt


random.seed(6)
random_integers = [random.randint(1, 100) for _ in range(5)]
print(random_integers)
manual_seed = random_integers[4]

random.seed(manual_seed)
np.random.seed(manual_seed)
torch.manual_seed(manual_seed)
torch.cuda.manual_seed(manual_seed)
torch.backends.cudnn.deterministic = True

num_epochs = 300
batch_size = 128

init_lr = 0.001
gamma = 10      # GRL weighting
theta = 1       # weights for losses
para = 0.5      # weights for attribute causality assistance
base_t = 0.6    # initial value of dynamic t
delta = 0.2     # change rate of dynamic t

task = 1
transfer_A_B = 1  # 1: Domain A → Domain B, 2: Domain B → Domain A

if task == 1:
    source_class = [0, 3, 5]
    target_class = [0, 3, 5, 6, 7]
elif task == 2:
    source_class = [2, 3, 4, 5]
    target_class = [2, 3, 4, 5, 7, 8]
elif task == 3:
    source_class = [2, 3, 4, 5]
    target_class = [2, 3, 4, 5, 7, 8, 9]
elif task == 4:
    source_class = [1, 2, 3, 4, 5, 6]
    target_class = [1, 2, 3, 4, 5, 6, 7, 8]
elif task == 5:
    source_class = [5, 6, 8, 9]
    target_class = [0, 2, 4, 5, 6, 8, 9]
elif task == 6:
    source_class = [3, 4, 5, 6, 8, 9]
    target_class = [0, 1, 2, 3, 4, 5, 6, 8, 9]

outlier_index = [item for item in target_class if item not in source_class]
output_size = len(source_class) + 1

# DataLoader
Source_data = OGW_data(source=True, target=False, known_index=source_class, outlier_index=outlier_index, transfer_A_B=transfer_A_B)
Target_data = OGW_data(source=False, target=True, known_index=source_class, outlier_index=outlier_index, transfer_A_B=transfer_A_B)
source_dataloader = DataLoader(Source_data, batch_size=batch_size, shuffle=True)
target_dataloader = DataLoader(Target_data, batch_size=batch_size, shuffle=True)

# Model initialization
feature_extractor_global = networks.fea_Extractor_global(in_channels=1, out_channels=64)    # global feature extractor

classifier = networks.Classifier(output_size=output_size)   # state classifier

AFE1 = networks.fea_Extractor_att1(in_channels=1, out_channels=10)    # hierarchical attribute feature extractor
AFE2 = networks.fea_Extractor_att2(in_channels=1, out_channels=10)
AFE3 = networks.fea_Extractor_att3(in_channels=1, out_channels=10)

AP1 = networks.Att_predictor1()         # attribute predictor
AP2 = networks.Att_predictor2()
AP3 = networks.Att_predictor3()

ATA2 = networks.Att_transmission2()     # attribute causality assisted (ACA) module
ATA3 = networks.Att_transmission3()

domain_classifier_att1 = networks.Domain_classifier_att1()      # attribute domain discriminator
domain_classifier_att2 = networks.Domain_classifier_att2()
domain_classifier_att3 = networks.Domain_classifier_att3()

class_criterion = nn.NLLLoss()
domain_criterion = nn.NLLLoss()
outlier_criterion = nn.BCELoss()
attribute_loss = nn.MSELoss()

optimizer = optim.Adam([{'params': feature_extractor_global.parameters()},
                        {'params': classifier.parameters()},
                        {'params': AFE1.parameters()},
                        {'params': AFE2.parameters()},
                        {'params': AFE3.parameters()},
                        {'params': ATA2.parameters()},
                        {'params': ATA3.parameters()},
                        {'params': AP1.parameters()},
                        {'params': AP2.parameters()},
                        {'params': AP3.parameters()},
                        {'params': domain_classifier_att1.parameters()},
                        {'params': domain_classifier_att2.parameters()},
                        {'params': domain_classifier_att3.parameters()}], lr=init_lr)

source_acc_list = []
target_acc_known_list = []
target_acc_outlier_list = []
loss_sum_list = []
loss_att_list = []
t_list = []

# training process
for epoch in range(num_epochs):
    feature_extractor_global.train()
    classifier.train()
    AFE1.train()
    AFE2.train()
    AFE3.train()
    ATA2.train()
    ATA3.train()
    AP1.train()
    AP2.train()
    AP3.train()
    domain_classifier_att1.train()
    domain_classifier_att2.train()
    domain_classifier_att3.train()

    # steps
    start_steps = epoch*len(source_dataloader)
    total_steps = num_epochs*len(source_dataloader)

    for batch_idx, (sdata, tdata) in enumerate(zip(source_dataloader, target_dataloader)):

        p = float(batch_idx+start_steps)/total_steps
        constant = 2./(1.+np.exp(-gamma*p))-1

        input1, label1, label_att_A = sdata
        input2, label2, label_att_B = tdata
        input1 = input1.unsqueeze(1)
        input2 = input2.unsqueeze(1)
        input1 = input1.float()
        input2 = input2.float()
        label_att_A = label_att_A.float()

        # domain labels
        source_labels = torch.zeros((input1.size()[0])).type(torch.LongTensor)
        target_labels = torch.ones((input2.size()[0])).type(torch.LongTensor)

        optimizer = optimizer_scheduler(optimizer, p, init_lr)
        optimizer.zero_grad()

        # Extract global features from the source and target domains
        src_feature_global = feature_extractor_global(input1)
        tgt_feature_global = feature_extractor_global(input2)

        # Extract attribute features from the source and target domains
        src_feature_att1 = AFE1(input1)
        src_feature_att2 = AFE2(input1)
        src_feature_att3 = AFE3(input1)
        
        tgt_feature_att1 = AFE1(input2)
        tgt_feature_att2 = AFE2(input2)
        tgt_feature_att3 = AFE3(input2)

        # ACA, first layer: attribute A1, A2, A3; second layer: A4, A5; third layer: A6, A7, A8
        # ACA structure depends on the attribute causality mining
        src_ATA2 = ATA2(src_feature_att1)
        src_ATA3 = ATA3(src_feature_att2)

        tgt_ATA2 = ATA2(tgt_feature_att1)
        tgt_ATA3 = ATA3(tgt_feature_att2)

        # Source domain attribute prediction
        # The inputs of AP2 and AP3 include the outputs of the previous layers
        src_pred_att1 = AP1(src_feature_att1)
        src_pred_att2 = AP2(src_feature_att2 + para*src_ATA2)
        src_pred_att3 = AP3(src_feature_att3 + para*src_ATA3)
        src_attribute_result = torch.cat((src_pred_att1, src_pred_att2, src_pred_att3), dim=1)

        # Target domain attribute prediction
        tgt_pred_att1 = AP1(tgt_feature_att1)
        tgt_pred_att2 = AP2(tgt_feature_att2 + para*tgt_ATA2)
        tgt_pred_att3 = AP3(tgt_feature_att3 + para*tgt_ATA3)
        tgt_attribute_result = torch.cat((tgt_pred_att1, tgt_pred_att2, tgt_pred_att3), dim=1)

        # Calculate losses
        # 1. attribute prediction loss in the source domain
        src_att_loss = attribute_loss(src_attribute_result, label_att_A)

        # 2. class logits and classification loss
        src_class_logits = classifier(src_feature_global, src_attribute_result)
        src_pred_class = F.log_softmax(src_class_logits, 1)
        src_pred_class_loss = class_criterion(src_pred_class, label1)

        # 3. loss between the probability of the target unknown class and the dynamic threshold t
        tgt_class_logits = classifier(tgt_feature_global, tgt_attribute_result, constant, adaption=True)
        outlier_prob = F.softmax(tgt_class_logits, dim=1)[:, -1]

        t = compute_dynamic_threshold(tgt_class_logits, base_t=base_t, delta=delta, output_size=output_size)
        outlier_thresholds = t * torch.ones((input2.size()[0])).type(torch.float)
        outlier_thresholds = outlier_thresholds.unsqueeze(1)
        outlier_loss_adv = outlier_criterion(outlier_prob.unsqueeze(1), outlier_thresholds)

        # 4. fine-grained domain discrimination loss of attributes at each layer
        tgt_domain_att1_pred = domain_classifier_att1(tgt_feature_att1.view(tgt_feature_att1.size(0), -1), constant)
        src_domain_att1_pred = domain_classifier_att1(src_feature_att1.view(src_feature_att1.size(0), -1), constant)
        tgt_domain_att1_loss = domain_criterion(tgt_domain_att1_pred, target_labels)
        src_domain_att1_loss = domain_criterion(src_domain_att1_pred, source_labels)
        domain_att1_loss = tgt_domain_att1_loss + src_domain_att1_loss
        tgt_domain_att2_pred = domain_classifier_att2(tgt_feature_att2.view(tgt_feature_att2.size(0), -1), constant)
        src_domain_att2_pred = domain_classifier_att2(src_feature_att2.view(src_feature_att2.size(0), -1), constant)
        tgt_domain_att2_loss = domain_criterion(tgt_domain_att2_pred, target_labels)
        src_domain_att2_loss = domain_criterion(src_domain_att2_pred, source_labels)
        domain_att2_loss = tgt_domain_att2_loss + src_domain_att2_loss
        tgt_domain_att3_pred = domain_classifier_att3(tgt_feature_att3.view(tgt_feature_att3.size(0), -1), constant)
        src_domain_att3_pred = domain_classifier_att3(src_feature_att3.view(src_feature_att3.size(0), -1), constant)
        tgt_domain_att3_loss = domain_criterion(tgt_domain_att3_pred, target_labels)
        src_domain_att3_loss = domain_criterion(src_domain_att3_pred, source_labels)
        domain_att3_loss = tgt_domain_att3_loss + src_domain_att3_loss

        finegrained_domain_loss = (domain_att1_loss + domain_att2_loss + domain_att3_loss)/3

        # total loss
        loss = src_att_loss + theta * src_pred_class_loss + theta * outlier_loss_adv + theta * finegrained_domain_loss

        loss.backward()
        optimizer.step()

    print('\nEpoch: {}'.format(epoch))
    print('Total Loss: {:.4f}\tAttribute Loss: {:.4f}\tSrc_class Loss: {:.4f}\tOutlier_class Loss: {:.4f}\tFine-grained Domain Loss: {:.4f}'.format(
         loss.item(), src_att_loss.item(), src_pred_class_loss.item(), outlier_loss_adv.item(), finegrained_domain_loss.item()))

    # test after each epoch
    source_acc, source_f1, target_acc_known, target_f1_known, target_acc_outlier, target_f1_outlier = test(epoch, feature_extractor_global, classifier,
                                                                                                           AFE1,AFE2, AFE3, ATA2, ATA3, AP1, AP2, AP3,
                                                                                                           source_dataloader, target_dataloader, para, output_size)
    source_acc_list.append(source_acc)
    target_acc_known_list.append(target_acc_known)
    target_acc_outlier_list.append(target_acc_outlier)
    loss_sum_list.append(loss.item())
    loss_att_list.append(src_att_loss.item())
    t_list.append(t)

# plot
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

ax1.plot(t_list, label='Dynamic t', color='mediumseagreen')
ax1.set_ylabel('t Value', fontsize=18)
ax1.set_xlabel('Training Iteration', fontsize=18)
ax1.legend(fontsize=16)
ax1.tick_params(axis='x', labelsize=14)
ax1.tick_params(axis='y', labelsize=14)

ax2.plot(loss_sum_list, label='Total Loss', color='royalblue')
ax2.set_xlabel('Training Iteration', fontsize=18)
ax2.set_ylabel('Loss Value', color='royalblue', fontsize=18)
ax2.tick_params(axis='x', labelsize=14)
ax2.tick_params(axis='y', labelcolor='royalblue', labelsize=14)

ax3 = ax2.twinx()
ax3.plot(source_acc_list, label='Source Accuracy', color='lightcoral')
ax3.set_ylabel('Accuracy', color='lightcoral', fontsize=18)
ax3.tick_params(axis='x', labelsize=14)
ax3.tick_params(axis='y', labelcolor='lightcoral', labelsize=14)

handles, labels = ax2.get_legend_handles_labels()
handles2, labels2 = ax3.get_legend_handles_labels()
handles.extend(handles2)
labels.extend(labels2)
ax2.legend(handles, labels, loc='right', fontsize=16)

fig.tight_layout()
plt.show()