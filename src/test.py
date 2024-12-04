import torch

num_examples = 3
pos_neg_k_results = torch.Tensor([[1,0,1],[1,0,0]])

# 初始化索引
pos_indices = torch.arange(num_examples)
neg_indices = torch.arange(num_examples)

# Step 1: 确定 flag 值 (是否 pos_neg_k_results[0][i] == 1)
flag = pos_neg_k_results[0] == 1  # [num_examples]

# Step 2: 构建掩码，分别寻找符合条件的 k
# 找到第一个满足条件的 k 对应的掩码
pos_mask = (pos_neg_k_results == 1) & (~flag.unsqueeze(0))  # 负样本中寻找第一个 1
neg_mask = (pos_neg_k_results == 0) & (flag.unsqueeze(0))   # 正样本中寻找第一个 0

# Step 3: 找到第一个满足条件的位置 k
# 对每个样本，找到满足掩码的第一个 k 索引
first_pos_k = torch.argmax(pos_mask.long(), dim=0)  # [num_examples]
first_neg_k = torch.argmax(neg_mask.long(), dim=0)  # [num_examples]

# Step 4: 检查是否有合法匹配项
valid_pos = pos_mask.any(dim=0)  # [num_examples] 是否找到合法的 k
valid_neg = neg_mask.any(dim=0)  # [num_examples]

# 没有合法匹配项的样本，设置 k 为 0（保持原值）
first_pos_k[~valid_pos] = 0
first_neg_k[~valid_neg] = 0

# Step 5: 更新索引
pos_indices[~flag] += first_pos_k[~flag] * num_examples  # flag == 0 时更新 pos_indices
neg_indices[flag] += first_neg_k[flag] * num_examples    # flag == 1 时更新 neg_indices


print(pos_indices, neg_indices)
