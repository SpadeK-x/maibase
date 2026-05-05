import math

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, random_split
from glob import glob
from tqdm import tqdm
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from CLAM import CLAM_SB


class ChartDataset(Dataset):
    def __init__(self):
        self.map = pd.read_csv('./labels.csv', index_col=None)
        self.map = self.map.set_index('chart')['level'].to_dict()  # 设置 'name' 列为索引，然后取 'age' 列并转为字典
        self.charts = []
        self.labels = []
        # self.label_map = ['10', '10+', '11', '11+', '12', '12+', '13', '13+', '14', '14+', '15']
        self.label_map = ['13', '13+', '14', '14+']
        self.label_map = {l: i for i, l in enumerate(self.label_map)}

        for chart_name, label in self.map.items():
            if label < 13.0:
                continue
            try:
                chart = torch.load(f'./charts/parsed_charts/{chart_name}.pt', map_location='cpu')
                if label==15.0:
                    continue
                self.charts.append(chart)
                # self.labels.append(label * 10 - 70)
                integer_part, fraction_part = divmod(label, 1)
                if round(fraction_part, ndigits=1) <= 0.5 + 1e-6:
                    self.labels.append(self.label_map[f'{int(integer_part)}'])
                else:
                    self.labels.append(self.label_map[f'{int(integer_part)}+'])
            except Exception as e:
                print(e)
                exit()
        # for chart_name, _ in self.map.items():
        #     try:
        #         chart = torch.load(f'./charts/parsed_charts/{chart_name}.pt', map_location='cpu')
        #         self.charts.append(chart)
        #         label = 0 if chart_name.split('_')[-1] == 'expert' else 1
        #         self.labels.append(label)
        #     except Exception as e:
        #         print(e)
        #         continue
        print(len(self.charts))
        super(ChartDataset, self).__init__()

    def __len__(self):
        return len(self.charts)

    def __getitem__(self, idx):
        return self.charts[idx], self.labels[idx]

    def get_n(self):
        return 4


def collate_fn(batch):
    # batch 是一个 list，包含 N 个样本
    # 每个样本是 (num_measures, num_notes, 5)
    y = torch.tensor([i[1] for i in batch], dtype=torch.long)
    batch = [i[0] for i in batch]

    # 1. 找到当前 batch 里最大的 measure 数
    max_m = max([s.shape[0] for s in batch])
    # 2. 找到当前 batch 里最大的 note 数 (或者直接用全局设定的 72)
    max_n = 32

    # 初始化全 0 张量
    B = len(batch)
    padded_x = torch.zeros(B, max_m, max_n, 5)
    note_mask = torch.ones(B, max_m, max_n, dtype=torch.bool)  # True表示Pad
    measure_mask = torch.ones(B, max_m, dtype=torch.bool)  # True表示Pad

    for i, song in enumerate(batch):
        m_len = song.shape[0]
        # 填充 measure mask (真实部分设为 False)
        measure_mask[i, :m_len] = False

        for j in range(m_len):
            n_len = song[j].shape[0]
            if n_len > max_n: n_len = max_n  # 截断

            # 填数据
            padded_x[i, j, :n_len, :] = song[j][:n_len, :]
            # 填充 note mask
            note_mask[i, j, :n_len] = False

    return padded_x, note_mask, measure_mask, y


# batch_data shape: [Batch, Max_Notes_Per_Bar, 5]
# 5个维度分别是: [time_idx, type_idx, lane_idx, dur_val, shape_idx]

class TimeEncoding(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.d_model = d_model

        # 定义一系列可学习的频率 (Frequencies)
        # 也可以固定为类似于 Transformer 的频率，但让它可学习通常更适应数据
        self.linear = nn.Linear(1, d_model // 2)

    def forward(self, time_values):
        # time_values: [Batch, Seq, 1] 或者是具体的秒数/毫秒数

        # 1. 投影到频率空间: weights * time + bias
        freqs = self.linear(time_values)

        # 2. 生成正弦和余弦
        # output: [Batch, Seq, d_model]
        sin_emb = torch.sin(freqs)
        cos_emb = torch.cos(freqs)

        return torch.cat([sin_emb, cos_emb], dim=-1)


class MusicEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        # 定义查表层
        self.emb_type = nn.Embedding(17, dim, padding_idx=0)
        self.emb_lane = nn.Embedding(43, dim, padding_idx=0)
        self.emb_shape = nn.Embedding(16, dim, padding_idx=0)
        self.time_encoder = TimeEncoding(dim)

        # 数值层
        self.proj_time = nn.Linear(1, dim)
        self.proj_dur = nn.Linear(1, dim)
        # 激活函数（可选，让线性层非线性化）
        self.activation = nn.GELU()

    def forward(self, x):
        # x: [Batch, Seq, 5]

        # 查表
        h_type = self.emb_type(x[:, :, 1].long())
        h_lane = self.emb_lane(x[:, :, 2].long())
        h_shape = self.emb_shape(x[:, :, 4].long())

        # 数值映射
        # 把 duration 维度单独拿出来变为 [Batch, Seq, 1]
        raw_time = x[:, :, 0].float()
        raw_dur = x[:, :, 3].float()

        # 示例方案：除以一个大致的最大值，限制在 0-1 左右
        # 或者使用 torch.log1p(raw_time) 如果跨度很大
        time_input = torch.log1p(raw_time).unsqueeze(-1).float().contiguous()
        dur_input = raw_dur.unsqueeze(-1).float().contiguous()  # duration 通常较小，可能不需要处理

        h_time = self.activation(self.time_encoder(raw_time.unsqueeze(-1)))
        h_dur = self.activation(self.proj_dur(dur_input))

        # 融合：相加 (Add) 是最常用的方式，保留各方面属性md

        final_embedding = h_time + h_type + h_lane + h_dur + h_shape

        return final_embedding


class HierarchicalRhythmModel(nn.Module):
    def __init__(self, dim, num_classes, cluster_num=100):
        super().__init__()

        # --- 1. 小节级编码器 (Note Level) ---
        self.note_embedding = MusicEmbedding(dim)
        # [Level 1 CLS] 用于聚合小节信息
        self.measure_cls_token = nn.Parameter(torch.randn(1, 1, dim))

        encoder_layer = nn.TransformerEncoderLayer(d_model=dim, nhead=4, batch_first=True, dropout=0.2, norm_first=True)
        self.measure_encoder = nn.TransformerEncoder(encoder_layer, num_layers=1)

        # model_dict = {"dropout": 0.4,
        #               'n_classes': 2,
        #               "embed_dim": dim}
        # self.measure_encoder = CLAM_SB(**model_dict)
        self.pos_encoder = TimeEncoding(dim)
        # --- 2. 歌曲级编码器 (Song Level) ---
        # [Level 2 CLS] 新增：用于聚合整首歌信息
        self.song_cls_token = nn.Parameter(torch.randn(1, 1, dim))

        song_layer = nn.TransformerEncoderLayer(d_model=dim, nhead=4, batch_first=True, dropout=0.2, norm_first=True)
        self.song_encoder = nn.TransformerEncoder(song_layer, num_layers=2)

        self.classifier = nn.Linear(dim, num_classes)

    def forward(self, x, note_mask, measure_mask):
        B, M, N, F = x.shape
        # ====================================================
        # 第一阶段：Note -> Measure (已修改部分)
        # ====================================================
        x_flat = x.view(B * M, N, F)
        note_mask_flat = note_mask.view(B * M, N)

        emb_notes = self.note_embedding(x_flat)
        data = emb_notes.detach().cpu().flatten().numpy()

        # 1. 拼接 Measure CLS
        curr_batch = emb_notes.size(0)
        measure_cls_tokens = self.measure_cls_token.expand(curr_batch, -1, -1)
        emb_notes_with_cls = torch.cat((measure_cls_tokens, emb_notes), dim=1)

        # 2. 更新 Mask
        cls_mask_note = torch.zeros((curr_batch, 1), dtype=torch.bool, device=x.device)
        new_note_mask = torch.cat((cls_mask_note, note_mask_flat), dim=1)

        # 3. 编码 & 提取
        encoded_notes = self.measure_encoder(emb_notes_with_cls, src_key_padding_mask=new_note_mask)
        measure_vectors_flat = encoded_notes[:, 0, :]  # [B*M, dim]

        # ====================================================
        # 第二阶段：Measure -> Song (新增修改部分)
        # ====================================================

        # 4. 还原维度: [Batch, Max_Measures, dim]
        measure_vectors = measure_vectors_flat.view(B, M, -1)
        # 添加位置编码
        raw_time = x[:, :, 0, 0].float()
        measure_vectors = measure_vectors + self.pos_encoder(raw_time.unsqueeze(-1))
        # 5. 拼接 Song CLS
        # 扩展到当前 Batch 大小 -> [Batch, 1, dim]
        song_cls_tokens = self.song_cls_token.expand(B, -1, -1)

        # 拼接到序列头部 -> [Batch, M+1, dim]
        song_input = torch.cat((song_cls_tokens, measure_vectors), dim=1)

        # 6. 更新 Mask (Song Level)
        # 给 Song CLS 一个 False (不Mask) -> [Batch, 1]
        cls_mask_song = torch.zeros((B, 1), dtype=torch.bool, device=x.device)
        # 拼接 -> [Batch, M+1]
        new_measure_mask = torch.cat((cls_mask_song, measure_mask), dim=1)

        # 7. 编码
        # 输出: [Batch, M+1, dim]
        song_context_all = self.song_encoder(song_input, src_key_padding_mask=new_measure_mask)

        # 8. 提取结果

        # 结果 A: 整首歌的全局向量 (Global Embedding)
        # 取第 0 个位置 -> [Batch, dim]
        whole_song_embedding = song_context_all[:, 0, :]

        # 结果 B: 每个小节的上下文向量 (Sequence Embedding)
        # 取后面 M 个位置 -> [Batch, M, dim]
        measure_sequence_output = song_context_all[:, 1:, :]

        # 根据你的任务返回你需要的部分
        # 如果是分类任务，返回 whole_song_embedding
        # 如果是生成任务或序列标注，返回 measure_sequence_output
        # 或者都返回
        logits = self.classifier(whole_song_embedding)
        return logits, whole_song_embedding, measure_sequence_output


device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
# 创建完整数据集
full_data = ChartDataset()
num_classes = full_data.get_n()
# 计算训练集和验证集的大小（9:1比例）
total_size = len(full_data)
train_size = int(0.8 * total_size)
eval_size = int(0.1 * total_size)
test_size = total_size - train_size - eval_size
# 随机分割数据集
train_data, eval_data, test_data = random_split(full_data, [train_size, eval_size, test_size])
# 创建数据加载器
train_loader = DataLoader(train_data,
                          shuffle=True,
                          batch_size=16,
                          collate_fn=collate_fn)
eval_loader = DataLoader(eval_data,
                         shuffle=False,
                         batch_size=32,
                         collate_fn=collate_fn)
test_loader = DataLoader(test_data,
                         shuffle=False,
                         batch_size=32,
                         collate_fn=collate_fn)
model = HierarchicalRhythmModel(64, num_classes).to(device)
lr = 5e-6
optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
criterion = nn.CrossEntropyLoss()

MAX_EPOCH = 100
# 早停策略参数
patience = 100
best_eval_loss = float('inf')
patience_counter = 0

# 记录训练历史
history = {
    'train_loss': [],
    'train_acc': [],
    'eval_loss': [],
    'eval_acc': []
}

for epoch in range(MAX_EPOCH):
    train_loss, eval_loss = 0, 0
    train_acc, eval_acc = 0, 0
    model.train()
    for batch in tqdm(train_loader, desc=f'epoch={epoch + 1}, train'):
        padded_x, note_mask, measure_mask, y = batch
        padded_x, note_mask, measure_mask, y = padded_x.to(device), note_mask.to(device), measure_mask.to(device), y.to(
            device)
        logits, _, _ = model(padded_x, note_mask, measure_mask)

        preds = torch.argmax(logits, dim=1)
        acc = (preds == y.long()).float().mean()
        train_acc += acc.item()

        loss = criterion(logits, y)
        loss.backward()
        # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        train_loss += loss.detach().item()
        optimizer.step()
        optimizer.zero_grad()

    train_loss = train_loss / len(train_loader)
    train_acc = train_acc / len(train_loader)
    print(f'train loss:{train_loss}')
    print(f'train acc:{train_acc}')

    model.eval()
    with torch.no_grad():
        for batch in tqdm(eval_loader, desc=f'epoch={epoch + 1}, eval'):
            padded_x, note_mask, measure_mask, y = batch
            padded_x, note_mask, measure_mask, y = padded_x.to(device), note_mask.to(device), measure_mask.to(
                device), y.to(device)
            logits, _, _ = model(padded_x, note_mask, measure_mask)

            preds = torch.argmax(logits, dim=1)
            acc = (preds == y.long()).float().mean()
            eval_acc += acc.item()

            loss = criterion(logits, y)
            eval_loss += loss.detach().item()

        eval_loss = eval_loss / len(eval_loader)
        eval_acc = eval_acc / len(eval_loader)
        print(f'eval loss:{eval_loss}')
        print(f'eval acc:{eval_acc}')

    # 记录历史
    history['train_loss'].append(train_loss)
    history['train_acc'].append(train_acc)
    history['eval_loss'].append(eval_loss)
    history['eval_acc'].append(eval_acc)

    # 早停检查
    if eval_loss < best_eval_loss:
        best_eval_loss = eval_loss
        patience_counter = 0
        # 保存最佳模型
        torch.save(model.state_dict(), 'best_model.pth')
        print(f'模型已保存，当前最佳验证损失: {best_eval_loss:.4f}')
    else:
        patience_counter += 1
        print(f'验证损失未改善，patience: {patience_counter}/{patience}')

    # 早停判断
    if patience_counter >= patience:
        print(f'早停触发！在epoch {epoch + 1}停止训练')
        break

# 加载最佳模型
print('加载最佳模型...')
model.load_state_dict(torch.load('best_model.pth'))

# 绘制训练曲线
plt.figure(figsize=(12, 5))

# Loss曲线
plt.subplot(1, 2, 1)
plt.plot(history['train_loss'], label='Train Loss', marker='o')
plt.plot(history['eval_loss'], label='Eval Loss', marker='s')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Validation Loss')
plt.legend()
plt.grid(True)

# Accuracy曲线
plt.subplot(1, 2, 2)
plt.plot(history['train_acc'], label='Train Acc', marker='o')
plt.plot(history['eval_acc'], label='Eval Acc', marker='s')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Training and Validation Accuracy')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig('training_curves.png', dpi=300, bbox_inches='tight')
print('训练曲线已保存到 training_curves.png')
plt.close()

# test
model.eval()
test_acc = 0
with torch.no_grad():
    for batch in tqdm(test_loader, desc=f'test'):
        padded_x, note_mask, measure_mask, y = batch
        padded_x, note_mask, measure_mask, y = padded_x.to(device), note_mask.to(device), measure_mask.to(device), y.to(
            device)
        logits, _, _ = model(padded_x, note_mask, measure_mask)

        preds = torch.argmax(logits, dim=1)
        acc = (preds == y.long()).float().mean()
        test_acc += acc.item()
    test_acc = test_acc / len(test_loader)
    print(f'Test Accuracy: {test_acc:.4f}')
