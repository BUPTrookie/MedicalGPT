# RLOO / PPO 策略优化学习笔记

基于 `ppo_training.py` 及 TRL 源码的逐步解析，涵盖数据处理、核心变量、损失函数、RLOO 简化思路等。

---

## 1. 数据预处理：为什么只取偶数项

### 代码

```python
for dialog in get_dialog(examples):
    for i in range(len(dialog) // 2):
        source_txt = dialog[2 * i]
        new_examples["prompt"].append(source_txt)
```

### 原因

`prompt_template.get_dialog()` 返回的 dialog 是 prompt 和 response 交替排列的扁平列表：

```
dialog = [prompt_0, response_0, prompt_1, response_1, ...]
          索引0      索引1       索引2      索引3
```

- 偶数索引（0, 2, 4...）= prompt（包含历史对话 + 当前问题，已套好模板）
- 奇数索引（1, 3, 5...）= response（模型的回答）

PPO/RLOO 训练只需要 prompt，不需要 response。因为 PPO 的流程是让策略模型**自己生成**回答，再由 reward model 打分，训练数据里的原始 response 没用。

---

## 2. 整体流程

```
┌──────────────────────────────────────────────────────┐
│ 1. 当前策略模型对每个 prompt 生成 K 个回答            │
│ 2. 生成时记录 logps_old（每个回答的对数概率）         │
│ 3. Reward model 给每个回答打分                       │
│ 4. 算 advantage（每个回答比平均水平好多少）           │
│                                                      │
│ 5. PPO 更新循环（多个 epoch）：                       │
│    a. 用当前参数的模型重新算 logps_new                │
│    b. ratio = exp(logps_new - logps_old)             │
│    c. 算 clipped loss                                │
│    d. 反向传播，更新参数                              │
│    e. 回到 a（logps_new 随参数变化，其他不变）        │
│                                                      │
│ 6. 这批数据用完，回到第 1 步，用新策略重新生成        │
└──────────────────────────────────────────────────────┘
```

每次回到第 1 步，模型已经稍微变好了一点：更倾向于生成高分回答。周而复始，策略逐步改进。

---

## 3. 核心变量详解

### 3.1 logps_old — 旧策略的对数概率

**含义：** 旧策略（参数更新前的模型）生成这个回答序列的对数概率。在生成回答时就算好并**存下来**，后续不再变化。

**计算方式：** 模型在每个位置输出下一个 token 的概率分布，取实际生成 token 的概率，所有 token 的对数概率求和。

```
假设生成了 "多喝水休息"，token 化为 [多, 喝, 水, 休, 息]

位置0（看到 prompt）     → P(多)  = 0.03  → log = -3.51
位置1（看到 prompt+多）  → P(喝)  = 0.15  → log = -1.90
位置2（看到 ...+喝）     → P(水)  = 0.40  → log = -0.92
位置3（看到 ...+水）     → P(休)  = 0.08  → log = -2.53
位置4（看到 ...+休）     → P(息)  = 0.70  → log = -0.36

logps_old = -3.51 + (-1.90) + (-0.92) + (-2.53) + (-0.36) = -9.22
```

### 3.2 logps_new — 新策略的对数概率

**含义：** 当前策略（参数已更新若干步的模型）对**同一个回答序列**重新计算的对数概率。

回答文本是固定的（旧模型生成的），不是让新模型重新生成，只是拿同样的文本重新算概率。每更新一次参数，logps_new 就会变化。

```
同样的 "多喝水休息"，用更新后的模型重新算：

位置0 → P_new(多) = 0.05  → log = -3.00
位置1 → P_new(喝) = 0.20  → log = -1.61
位置2 → P_new(水) = 0.35  → log = -1.05
位置3 → P_new(休) = 0.10  → log = -2.30
位置4 → P_new(息) = 0.75  → log = -0.29

logps_new = -3.00 + (-1.61) + (-1.05) + (-2.30) + (-0.29) = -8.25
```

### 3.3 ratio — 新旧概率比

```python
log_ratio = logps_new - logps_old    # -8.25 - (-9.22) = 0.97
ratio = exp(log_ratio)               # exp(0.97) ≈ 2.64
```

**含义：** `P_new(回答) / P_old(回答)`。ratio = 2.64 意味着新策略把这个回答的概率提高到了原来的 2.64 倍。

| ratio 值 | 含义 |
|---|---|
| 1.0 | 新旧策略对这个回答的评价完全一样 |
| 2.64 | 新策略把概率提高到 2.64 倍 |
| 0.3 | 新策略把概率压低到原来的 30% |

### 3.4 advantages — 优势值

**含义：** 这个回答比平均水平好多少。正值 = 好回答，负值 = 差回答。

具体计算方式取决于用 PPO 还是 RLOO（见第 5 节）。

---

## 4. PPO-Clip 损失函数

### 公式

```python
coef_1 = ratio * advantages                           # 无约束目标
coef_2 = clamp(ratio, 1-ε, 1+ε) * advantages          # 截断目标，ε 通常为 0.2
loss = -min(coef_1, coef_2).mean()
```

### loss 如何引导模型

关键在于 loss 对 logps_new 的梯度方向。优化器的目标是让 loss 变小。

**好回答（advantage > 0）：**

```
loss = -exp(logps_new - logps_old) * advantage
```

advantage 是正常数。要让 loss 变小（更负），就要让 exp(...) 变大，即让 logps_new 变大。logps_new 变大 = 模型认为这个回答的概率提高了 → **鼓励好回答**。

**差回答（advantage < 0）：**

```
loss = -exp(logps_new - logps_old) * (负数) = exp(...) * |advantage|
```

正值。要让 loss 变小，就要让 exp(...) 变小，即让 logps_new 变小 → **抑制差回答**。

### clamp 截断的作用

没有 clamp 的话，好回答的概率会被无限推高，差回答被无限压低，策略会崩。

clamp 把 ratio 限制在 `[0.8, 1.2]`（ε=0.2 时），效果是：

- 好回答概率提高到 1.2 倍后，loss 不再因继续提高而减小 → 梯度趋近 0 → 停止推
- 差回答概率降低到 0.8 倍后，loss 不再因继续降低而减小 → 梯度趋近 0 → 停止压

**每一步只允许小幅调整，然后重新采样，再来下一步。这就是 "Proximal"（近端）的含义。**

### min 的截断逻辑：只截断过度优化的方向

| | advantage > 0（好回答） | advantage < 0（差回答） |
|---|---|---|
| ratio > 1+ε | min 选 coef_2，**封顶收益** | min 选 coef_1，不截断 |
| ratio 在范围内 | coef_1 = coef_2，无截断 | coef_1 = coef_2，无截断 |
| ratio < 1-ε | min 选 coef_1，不截断 | min 选 coef_2，**封底惩罚** |

### 具体例子

**差回答（advantage = -1.65），模型已经压了一些（ratio = 0.7）：**

```
coef_1 = 0.7 × (-1.65) = -1.155
coef_2 = clamp(0.7, 0.8, 1.2) × (-1.65) = 0.8 × (-1.65) = -1.32
min(-1.155, -1.32) = -1.32     ← 选更负的
loss = -(-1.32) = +1.32        ← 截断值兜底，不让 loss 继续下降
```

如果用 coef_1，loss = 1.155，更小，模型会被鼓励继续压。但 min 选了 coef_2 = -1.32，loss 固定在 1.32 不再下降 → 告诉模型：压到 0.8 就够了。

**好回答（advantage = +1.07），模型已经提升了（ratio = 1.5）：**

```
coef_1 = 1.5 × 1.07 = 1.605
coef_2 = clamp(1.5, 0.8, 1.2) × 1.07 = 1.2 × 1.07 = 1.284
min(1.605, 1.284) = 1.284      ← 选更小的
loss = -(1.284) = -1.284       ← 封顶，不让 ratio 继续贪
```

---

## 5. RLOO vs 标准 PPO

### 标准 PPO：4 个模型

| 模型 | 作用 |
|---|---|
| **Policy model** | 当前策略，生成回答，参数在更新 |
| **Reference model** | 旧策略的冻结副本，用于算 KL 散度惩罚，防止策略跑太远 |
| **Reward model** | 给回答打分 |
| **Value model (Critic)** | 估计每个状态的"基线值"，用于算 advantage = reward - baseline |

PPO 的 advantage 依赖 Value model：`advantage = reward - V(state)`。Value model 本身也需要训练，多了一个模型的显存和计算开销。

### RLOO 的简化：去掉 Value model

RLOO（REINFORCE Leave-One-Out）的核心思想：**不需要 Value model 来估计基线，用同一个 prompt 的其他回答的平均 reward 当基线。**

假设一个 prompt 生成了 K=4 个回答：

```
回答A: reward = 3.5
回答B: reward = 4.2
回答C: reward = 2.8
回答D: reward = 0.5
```

算回答 A 的 advantage 时，用**其余回答**（Leave-One-Out）的平均 reward 当基线：

```
baseline_A = (4.2 + 2.8 + 0.5) / 3 = 2.5
advantage_A = 3.5 - 2.5 = +1.0
```

算回答 B 的 advantage：

```
baseline_B = (3.5 + 2.8 + 0.5) / 3 = 2.27
advantage_B = 4.2 - 2.27 = +1.93
```

每个回答的基线排除了自己，用剩下 K-1 个回答的均值。这样就不需要单独训练一个 Value model 来估计基线了。

### 对比总结

| | PPO | RLOO |
|---|---|---|
| 模型数量 | 4 | 2（Policy + Reward） |
| baseline 来源 | Value model 预测 | 同 prompt 其他回答的均值 |
| 要求 | 每个 prompt 生成 1 个回答即可 | 每个 prompt 需要生成 **K 个回答**（K≥2） |
| 显存 | 大（多一个 Value model） | 小 |
| advantage 质量 | Value model 训好的话更精确 | K 越大越稳定，K 小时方差大 |

### Reference model 在 RLOO 中的处理

RLOO 也可以加 Reference model 来算 KL 惩罚，但 KL 惩罚是直接加在 reward 上的：

```
adjusted_reward = reward - β * KL(policy || reference)
```

不需要 Reference model 作为单独的模型参与 loss 计算。有些实现直接用 logps_old 近似 reference，进一步省掉这个模型。
