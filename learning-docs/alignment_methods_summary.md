# RLOO / GRPO / DPO 对齐方法综合总结

> 基于 MedicalGPT 项目代码的学习笔记，结合原理与实现。

---

## 一、为什么需要对齐？

SFT（监督微调）让模型学会了"像人一样说话"，但它只是模仿训练数据，不知道什么回答是好的、什么是差的。对齐（Alignment）的目标是：**让模型学会区分好坏，主动生成更好的回答。**

三种方法解决同一个问题，思路不同：

| 方法 | 一句话总结 |
|------|----------|
| **RLOO** | 让模型生成回答 → 奖励模型打分 → 好的鼓励、差的抑制 |
| **GRPO** | 让模型生成回答 → 规则函数判对错 → 对的鼓励、错的抑制 |
| **DPO** | 直接拿现成的好/差回答 → 提高好回答概率、降低差回答概率 |

---

## 二、核心对比表

| | RLOO | GRPO | DPO |
|---|---|---|---|
| **本质** | 强化学习 | 强化学习 | 监督学习 |
| **训练时要生成？** | 是 | 是 | 否 |
| **奖励来源** | 神经网络（奖励模型） | 规则函数（判对错 + 查格式） | 隐式（概率比即奖励） |
| **前置依赖** | SFT 模型 + RM 模型 | 仅 SFT 模型 | 仅 SFT 模型 |
| **数据格式** | 只需 prompt | prompt + 标准答案 | prompt + chosen + rejected |
| **内存中模型数** | 2（策略 + 奖励） | 1（策略） | 1（策略，LoRA 关闭 = 参考） |
| **Baseline 计算** | 留一法（去掉自己算均值） | 组均值 + 标准化（z-score） | 无（参考模型充当锚点） |
| **更新方式** | PPO-Clip | PPO-Clip | 单一损失函数梯度下降 |
| **训练速度** | 慢（要生成 + 打分） | 慢（要生成） | 快（纯前向传播） |
| **适用场景** | 通用对话 | 数学/代码等可验证任务 | 通用对话 |
| **代表工作** | InstructGPT | DeepSeek-R1 | Llama 2/3 |

---

## 三、RLOO 详解

### 3.1 完整流程

```
阶段 1：训练奖励模型（reward_modeling.py）
  SFT 模型 → 替换 lm_head 为 score 层（输出标量）
  数据：{prompt, chosen, rejected} 偏好对
  损失：Bradley-Terry loss = -log σ(r_chosen - r_rejected)

阶段 2：RLOO 策略训练（ppo_training.py）
  加载：策略模型（SFT 训好的） + 奖励模型（阶段1训好的）
  数据：只需 prompt
```

### 3.2 训练循环

```
for each batch of prompts:
  1. 策略模型对每个 prompt 生成 K 个回答（num_generations >= 2）
  2. 奖励模型对每个 (prompt + 回答) 打分 → r_1, r_2, ..., r_K
  3. Leave-One-Out baseline:
       baseline_i = mean(所有 r_j, j != i)
       advantage_i = r_i - baseline_i
  4. PPO-Clip 更新:
       ratio = π_new(回答|prompt) / π_old(回答|prompt)
       loss = -min(ratio * adv, clip(ratio, 1±ε) * adv)
  5. 可选 KL 惩罚: reward -= β * KL(π || π_ref)
```

### 3.3 Leave-One-Out 示例

4 个回答的分数为 `[3.2, 1.5, 2.8, 2.0]`：

```
baseline_1 = (1.5 + 2.8 + 2.0) / 3 = 2.1    → adv_1 = +1.1  ← 鼓励
baseline_2 = (3.2 + 2.8 + 2.0) / 3 = 2.67   → adv_2 = -1.17 ← 抑制
baseline_3 = (3.2 + 1.5 + 2.0) / 3 = 2.23   → adv_3 = +0.57 ← 鼓励
baseline_4 = (3.2 + 1.5 + 2.8) / 3 = 2.5    → adv_4 = -0.5  ← 抑制
```

### 3.4 关键代码位置

- 奖励模型架构：`training/reward_modeling.py` — `AutoModelForSequenceClassification(num_labels=1)`
- 奖励模型损失：`training/reward_modeling.py` — `RewardTrainer.compute_loss()`
- 策略训练入口：`training/ppo_training.py` — `RLOOTrainer`
- 运行脚本：`scripts/run_rm.sh`（阶段1）+ `scripts/run_ppo.sh`（阶段2）

### 3.5 和真正 PPO 的区别

| | PPO | RLOO |
|---|---|---|
| 内存中模型 | 4 个（策略、参考、奖励、价值） | 2 个（策略 + 奖励） |
| Baseline 来源 | 价值模型（需额外训练） | 同 prompt 其他回答的平均分 |
| 每个 prompt 生成数 | 1 个 | >= 2 个 |
| 显存需求 | 极高 | 约一半 |

本项目文件名叫 `ppo_training.py`，但实际实现完全是 RLOO。

---

## 四、GRPO 详解

### 4.1 核心创新：规则函数代替奖励模型

GRPO 不需要训练奖励模型，用**代码函数**直接判断回答质量：

**奖励函数 1 — 准确度奖励（accuracy_reward）：**

```python
# 从标准答案解析正确数字（如 "#### 72" → 72）
gold_parsed = parse(sol.split("####", 1)[-1].strip())
# 从模型输出的 <answer> 标签提取答案
answer_parsed = parse(extract_answer(content))
# 数学验证：对=1.0，错=0.0
reward = float(verify(answer_parsed, gold_parsed))
```

**奖励函数 2 — 格式奖励（format_reward）：**

```python
# 检查是否符合 <think>...</think><answer>...</answer> 格式
pattern = r"<think>.*?</think><answer>.*?</answer>$"
reward = 1.0 if re.match(pattern, content) else 0.0
```

最终奖励 = 两个函数分数的加权和。模型同时学会"答对"和"按格式输出"。

### 4.2 System Prompt（R1 风格）

```
The assistant first thinks about the reasoning process in the mind and then provides
the user with the answer. The reasoning process and answer are enclosed within
<think> </think> and <answer> </answer> tags.
```

用 system prompt 指定输出格式，用奖励函数强化这个格式 → 模型自己学会"先推理再回答"。

### 4.3 训练循环

```
for each batch of prompts:
  1. 策略模型对每个 prompt 生成 K 个回答（num_generations=4）
  2. 规则函数打分（不需要奖励模型）
       accuracy_reward: [1.0, 0.0, 1.0, 1.0]
       format_reward:   [1.0, 1.0, 1.0, 0.0]
       总奖励:          [2.0, 1.0, 2.0, 1.0]
  3. 组内相对优势（Group Relative）
       mean = 1.5, std = 0.577
       advantage_i = (reward_i - mean) / std   ← z-score 标准化
  4. PPO-Clip 策略更新（和 RLOO 一样的公式）
  5. KL 惩罚: reward -= β * KL（β=0.001，比 RLOO 小）
```

### 4.4 和 RLOO 的 Baseline 对比

```
RLOO（留一法）:
  分数 [3.2, 1.5, 2.8, 2.0]
  baseline_1 = (1.5+2.8+2.0)/3 = 2.1          ← 去掉自己算其余均值

GRPO（组内相对）:
  分数 [2.0, 1.0, 2.0, 1.0]
  mean = 1.5, std = 0.577
  advantage_i = (reward_i - mean) / std        ← 全组均值 + 除以标准差
```

GRPO 多了除以标准差的归一化，让不同 prompt 之间的 advantage 量纲一致。

### 4.5 关键代码位置

- 奖励函数：`training/grpo_training.py` — `accuracy_reward()`, `format_reward()`
- 数据预处理：`training/grpo_training.py:208-219` — 构造 `{prompt, answer}` 格式
- Trainer 初始化：`training/grpo_training.py:392-402` — `GRPOTrainer(reward_funcs=[...])`
- 运行脚本：`scripts/run_grpo.sh`

---

## 五、DPO 详解

### 5.1 核心思想：跳过奖励模型

DPO 的数学推导证明：**策略模型对 chosen/rejected 的对数概率差，本身就隐含了一个奖励函数。**

```
传统 RLHF: SFT → 训奖励模型 → 用奖励模型做 RL → 最终策略
DPO:       SFT → 直接用偏好数据优化 → 最终策略
```

### 5.2 损失函数

```
L_DPO = -log σ(β * (Δlog_chosen - Δlog_rejected))

其中:
  Δlog_chosen  = log π(chosen|prompt) - log π_ref(chosen|prompt)
  Δlog_rejected = log π(rejected|prompt) - log π_ref(rejected|prompt)
```

直觉理解：

- `Δlog_chosen` 大 → 策略模型比参考模型更倾向于 chosen → 好
- `Δlog_rejected` 大 → 策略模型比参考模型更倾向于 rejected → 坏
- 损失驱动：**提高 chosen 概率，降低 rejected 概率，同时不偏离参考模型太远**

### 5.3 参考模型的巧妙处理

```python
trainer = DPOTrainer(
    model,
    ref_model=None if args.use_peft else deepcopy(model),
    peft_config=peft_config if args.use_peft else None,
)
```

| 场景 | ref_model | 原因 |
|------|-----------|------|
| 用 LoRA | `None` | 关闭 adapter = 原始 base model = 天然参考模型 |
| 全参数训练 | `deepcopy(model)` | 需要显式拷贝一份冻结副本 |

### 5.4 数据预处理

```python
# 输出三个纯文本字段，DPOTrainer 内部自动 tokenize
{
    "prompt":   "<|im_start|>user\n问题...<|im_end|>\n<|im_start|>assistant\n",
    "chosen":   "好回答文本",
    "rejected": "差回答文本",
}
```

### 5.5 训练循环

```
for each batch:
  1. 策略模型（LoRA ON）计算:
       log π(chosen|prompt),  log π(rejected|prompt)
  2. 参考模型（LoRA OFF）计算:
       log π_ref(chosen|prompt),  log π_ref(rejected|prompt)
  3. DPO 损失:
       loss = -log σ(β * (Δlog_chosen - Δlog_rejected))
  4. 反向传播 → 只更新 LoRA 参数
```

**关键特点：没有生成步骤。** chosen 和 rejected 来自数据集，模型只算概率，不自己生成。

### 5.6 关键代码位置

- 数据预处理：`training/dpo_training.py:293-328` — `return_prompt_and_responses()`
- 参考模型处理：`training/dpo_training.py:503-511` — `ref_model=None` 的 LoRA 技巧
- LoRA 配置：`training/dpo_training.py:493-500` — `TaskType.CAUSAL_LM`
- 运行脚本：`scripts/run_dpo.sh`

---

## 六、损失函数对比

三个方法的损失函数都基于 Bradley-Terry 偏好模型，形式惊人地相似：

```
RM:   loss = -log σ(r_chosen - r_rejected)
DPO:  loss = -log σ(β * (Δlog_chosen - Δlog_rejected))
RLOO: loss = -min(ratio * adv, clip(ratio, 1±ε) * adv)
GRPO: loss = -min(ratio * adv, clip(ratio, 1±ε) * adv)
```

- RM 和 DPO 都是直接优化偏好概率
- RLOO 和 GRPO 的损失形式一样（PPO-Clip），区别只在 advantage 怎么算

---

## 七、LoRA 配置对比

| | SFT | RM | DPO | GRPO | RLOO |
|---|---|---|---|---|---|
| TaskType | CAUSAL_LM | **SEQ_CLS** | CAUSAL_LM | CAUSAL_LM | CAUSAL_LM |
| 排除层 | lm_head | lm_head + **score** | lm_head | lm_head | — (由 TRL 管理) |

只有奖励模型用 `SEQ_CLS`（序列分类），其余都是 `CAUSAL_LM`（因果语言模型）。

---

## 八、如何选择

- **有偏好数据、想简单快速** → **DPO**（一步到位，训练最快）
- **有偏好数据、想要更精细控制** → **RLOO**（先训 RM 再做 RL，可解释性好）
- **任务有明确正确答案（数学/代码）** → **GRPO**（规则函数最直接，不需要人工标注偏好）

---

## 九、项目文件对应关系

```
training/reward_modeling.py    → 奖励模型训练（RLOO 的前置步骤）
training/ppo_training.py       → RLOO 策略训练（名字叫 PPO，实际是 RLOO）
training/grpo_training.py      → GRPO 训练
training/dpo_training.py       → DPO 训练

scripts/run_rm.sh              → 奖励模型训练脚本
scripts/run_ppo.sh             → RLOO 训练脚本
scripts/run_grpo.sh            → GRPO 训练脚本
scripts/run_dpo.sh             → DPO 训练脚本

data/reward/                   → 偏好数据（RM 和 DPO 共用）
data/grpo/                     → GRPO 数据（或默认用 openai/gsm8k）
```
