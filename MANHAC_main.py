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


# ================= 1. 环境配置与种子设置 =================
device = torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))
print(f"Using device: {device}")

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)


# ================= 2. 参数设置 =================
num_epochs = 300
batch_size = 128
init_lr = 0.001
gamma = 10      # GRL 权重系数
theta = 1       # 损失函数权重
para = 0.5      # 属性因果辅助权重
base_t = 0.6    # 动态阈值初始值
delta = 0.2     # 动态阈值变化率

task = 1
transfer_A_B = 1  # 1: Domain A → Domain B, 2: Domain B → Domain A

# 任务设置
task_configs = {
    1: ([0, 3, 5], [0, 3, 5, 6, 7]),
    2: ([2, 3, 4, 5], [2, 3, 4, 5, 7, 8]),
    3: ([2, 3, 4, 5], [2, 3, 4, 5, 7, 8, 9]),
    4: ([1, 2, 3, 4, 5, 6], [1, 2, 3, 4, 5, 6, 7, 8]),
    5: ([5, 6, 8, 9], [0, 2, 4, 5, 6, 8, 9]),
    6: ([3, 4, 5, 6, 8, 9], [0, 1, 2, 3, 4, 5, 6, 8, 9])
}
source_class, target_class = task_configs.get(task)
outlier_index = [item for item in target_class if item not in source_class]
output_size = len(source_class) + 1


# ================= 3. 数据加载 =================
Source_data = OGW_data(source=True, target=False, known_index=source_class, outlier_index=outlier_index, transfer_A_B=transfer_A_B)
Target_data = OGW_data(source=False, target=True, known_index=source_class, outlier_index=outlier_index, transfer_A_B=transfer_A_B)

source_dataloader = DataLoader(Source_data, batch_size=batch_size, shuffle=True)
target_dataloader = DataLoader(Target_data, batch_size=batch_size, shuffle=True)


# ================= 4. 模型初始化 =================
models = {
    "FE_global": networks.fea_Extractor_global(in_channels=1, out_channels=64),
    "classifier": networks.Classifier(output_size=output_size),
    "AFE1": networks.fea_Extractor_att1(in_channels=1, out_channels=10),
    "AFE2": networks.fea_Extractor_att2(in_channels=1, out_channels=10),
    "AFE3": networks.fea_Extractor_att3(in_channels=1, out_channels=10),
    "AP1": networks.Att_predictor1(),
    "AP2": networks.Att_predictor2(),
    "AP3": networks.Att_predictor3(),
    "ATA2": networks.Att_transmission2(),
    "ATA3": networks.Att_transmission3(),
    "DC1": networks.Domain_classifier_att1(),
    "DC2": networks.Domain_classifier_att2(),
    "DC3": networks.Domain_classifier_att3()
}

# 将所有模型移至设备并收集参数
all_params = []
for name, model in models.items():
    model.to(device)
    all_params.append({'params': model.parameters()})

optimizer = optim.Adam(all_params, lr=init_lr)

class_criterion = nn.NLLLoss()
domain_criterion = nn.NLLLoss()
outlier_criterion = nn.BCELoss()
attribute_loss = nn.MSELoss()


# ================= 5. 训练循环 =================
history = {
    "source_acc": [], "target_acc_k": [], "target_acc_o": [],
    "loss_total": [], "loss_att": [], "t_value": []
}

for epoch in range(num_epochs):
    for m in models.values(): m.train()

    epoch_loss, epoch_att_loss = 0, 0
    start_steps = epoch * len(source_dataloader)
    total_steps = num_epochs * len(source_dataloader)

    for batch_idx, (sdata, tdata) in enumerate(zip(source_dataloader, target_dataloader)):
        p = float(batch_idx + start_steps) / total_steps
        constant = 2. / (1. + np.exp(-gamma * p)) - 1

        optimizer = optimizer_scheduler(optimizer, p, init_lr)
        optimizer.zero_grad()

        # 数据准备
        s_in, s_label, s_att = [x.to(device).float() if i != 1 else x.to(device) for i, x in enumerate(sdata)]
        t_in, t_label, _ = [x.to(device).float() if i != 1 else x.to(device) for i, x in enumerate(tdata)]

        s_in, t_in = s_in.unsqueeze(1), t_in.unsqueeze(1)
        s_domain_labels = torch.zeros(s_in.size(0), dtype=torch.long).to(device)
        t_domain_labels = torch.ones(t_in.size(0), dtype=torch.long).to(device)

        # 特征提取与属性预测
        s_fea_g = models["FE_global"](s_in)
        t_fea_g = models["FE_global"](t_in)

        s_f1, s_f2, s_f3 = models["AFE1"](s_in), models["AFE2"](s_in), models["AFE3"](s_in)
        t_f1, t_f2, t_f3 = models["AFE1"](t_in), models["AFE2"](t_in), models["AFE3"](t_in)

        # ACA因果传递
        s_pred_att1 = models["AP1"](s_f1)
        s_pred_att2 = models["AP2"](s_f2 + para * models["ATA2"](s_f1))
        s_pred_att3 = models["AP3"](s_f3 + para * models["ATA3"](s_f2))
        s_att_res = torch.cat((s_pred_att1, s_pred_att2, s_pred_att3), dim=1)

        t_pred_att1 = models["AP1"](t_f1)
        t_pred_att2 = models["AP2"](t_f2 + para * models["ATA2"](t_f1))
        t_pred_att3 = models["AP3"](t_f3 + para * models["ATA3"](t_f2))
        t_att_res = torch.cat((t_pred_att1, t_pred_att2, t_pred_att3), dim=1)

        loss_att_val = attribute_loss(s_att_res, s_att)

        s_logits = models["classifier"](s_fea_g, s_att_res)
        loss_cls = class_criterion(F.log_softmax(s_logits, 1), s_label)

        t_logits = models["classifier"](t_fea_g, t_att_res, constant, adaption=True)
        outlier_prob = F.softmax(t_logits, dim=1)[:, -1]

        # 动态阈值计算
        t_val = compute_dynamic_threshold(t_logits, base_t=base_t, delta=delta, output_size=output_size)
        loss_outlier = outlier_criterion(outlier_prob, torch.full_like(outlier_prob, t_val))

        # 细粒度领域对抗损失
        def get_dom_loss(dc, src_f, tgt_f, const):
            s_d = dc(src_f.view(src_f.size(0), -1), const)
            t_d = dc(tgt_f.view(tgt_f.size(0), -1), const)
            return domain_criterion(s_d, s_domain_labels) + domain_criterion(t_d, t_domain_labels)


        loss_dom = (get_dom_loss(models["DC1"], s_f1, t_f1, constant) +
                    get_dom_loss(models["DC2"], s_f2, t_f2, constant) +
                    get_dom_loss(models["DC3"], s_f3, t_f3, constant)) / 3

        total_loss = loss_att_val + theta * (loss_cls + loss_outlier + loss_dom)

        total_loss.backward()
        optimizer.step()

        epoch_loss += total_loss.item()
        epoch_att_loss += loss_att_val.item()

    # 记录每个 Epoch 的平均值
    avg_loss = epoch_loss / len(source_dataloader)
    history["loss_total"].append(avg_loss)
    history["loss_att"].append(epoch_att_loss / len(source_dataloader))
    history["t_value"].append(t_val)

    # 评估
    res = test(epoch, models["FE_global"], models["classifier"],
               models["AFE1"], models["AFE2"], models["AFE3"],
               models["ATA2"], models["ATA3"], models["AP1"], models["AP2"], models["AP3"],
               source_dataloader, target_dataloader, para, output_size)

    history["source_acc"].append(res[0])
    history["target_acc_k"].append(res[2])
    history["target_acc_o"].append(res[4])

    print(f"Epoch {epoch} | Loss: {avg_loss:.4f} | Att_Loss: {history['loss_att'][-1]:.4f} | t: {t_val:.4f}\n")


# ================= 6. 绘图可视化 =================
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']  # 根据系统调整
plt.rcParams['axes.unicode_minus'] = False
# 全局字号设置
plt.rcParams.update({
    'font.size': 16
})

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

ax1.plot(history["t_value"], label='Dynamic t', color='mediumseagreen', lw=2)
ax1.set_xlabel('Epoch')

ax2.plot(history["loss_total"], label='Total Loss', color='royalblue', alpha=0.7)
ax2.set_ylabel('Loss Value', color='royalblue')
ax2.set_xlabel('Epoch')

ax3 = ax2.twinx()
ax3.plot(history["source_acc"], label='Source Acc', color='lightcoral', lw=2)
ax3.set_ylabel('Accuracy', color='darkred')

lines1, labels1 = ax2.get_legend_handles_labels()
lines2, labels2 = ax3.get_legend_handles_labels()
ax2.legend(lines1 + lines2, labels1 + labels2, frameon=False, fontsize=12, loc='right')

plt.tight_layout()
plt.show()