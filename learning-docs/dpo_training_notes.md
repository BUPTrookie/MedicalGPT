# DPO 训练学习笔记

基于 `training/dpo_training.py` 及 TRL `DPOTrainer` 源码的逐步解析，涵盖数据预处理、参考模型处理、损失函数与反向传播。

---

## 1. DPO 是什么

DPO（Direct Preference Optimization）是一种**跳过 Reward Model 的对齐方法**。传统 RLHF 路径是：

```
SFT → 训练 Reward Model → PPO/RLOO 强化学习
```

DPO 把中间两步合成一步：

```
SFT → DPO（直接用偏好数据训练策略模型）
```

核心思想：不需要先训练一个打分模型，直接从"人类觉得 A 比 B 好"的偏好对中学习，通过一个巧妙的数学变换把 RL 优化问题转成了监督学习问题。

---

## 2. 数据预处理（第 293-328 行）

### 原始数据格式

```json
{
    "system": "",
    "history": [],
    "question": "20个关于新鲜果汁菜单的口号...",
    "response_chosen": "这里是一个名为Dishes的餐厅的20个口号...",
    "response_rejected": "1. \"与菜肴一起品尝新鲜！\"..."
}
```

与 Reward Modeling 数据格式完全一样。区别：RM 用偏好对训练打分模型，DPO 直接用偏好对训练策略模型。

### 预处理函数：return_prompt_and_responses

**目标**：把 5 字段转成 DPOTrainer 需要的 3 字段。

```
输入: {"system", "history", "question", "response_chosen", "response_rejected"}
输出: {"prompt", "chosen", "rejected"}
```

#### 两条路径构造 prompt

**路径 1：自定义模板**（`template_name` 有值时）

```python
if prompt_template:
    history_with_question = history + [[question, '']] if history else [[question, '']]
    prompts.append(prompt_template.get_prompt(messages=history_with_question, system_prompt=system_prompt))
```

把 question 和空回答 `''` 组成 pair 附加到 history 末尾。空回答表示"这轮还没回答"，模板只输出 prompt 部分。

**路径 2：tokenizer 内置 chat_template**（默认方式）

```python
else:
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    if history:
        for h_q, h_a in history:
            messages.append({"role": "user", "content": h_q})
            messages.append({"role": "assistant", "content": h_a})
    messages.append({"role": "user", "content": question})
    prompts.append(tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    ))
```

关键参数：
- `tokenize=False`：返回字符串（DPOTrainer 内部自己 tokenize）
- `add_generation_prompt=True`：末尾加 assistant 开头标记，如 `<|im_start|>assistant\n`

#### 具体例子：有历史对话

```json
{
    "system": "你是一个非常聪明的AI助手",
    "history": [["什么是有机农业？", "有机农业是一种不使用化学肥料的农业方式。"]],
    "question": "它有什么优点？"
}
```

构造出的 messages：

```python
[
    {"role": "system",    "content": "你是一个非常聪明的AI助手"},
    {"role": "user",      "content": "什么是有机农业？"},
    {"role": "assistant", "content": "有机农业是一种不使用化学肥料的农业方式。"},
    {"role": "user",      "content": "它有什么优点？"},
]
```

经 `apply_chat_template` 后（以 Qwen 为例）：

```
<|im_start|>system
你是一个非常聪明的AI助手<|im_end|>
<|im_start|>user
什么是有机农业？<|im_end|>
<|im_start|>assistant
有机农业是一种不使用化学肥料的农业方式。<|im_end|>
<|im_start|>user
它有什么优点？<|im_end|>
<|im_start|>assistant
```

历史对话全部编进 prompt，chosen 和 rejected 只是最后一轮的回答纯文本。

#### 最终输出

```python
return {
    "prompt":   prompts,                       # 格式化后的完整 prompt 字符串
    "chosen":   examples["response_chosen"],   # 直接透传，纯文本
    "rejected": examples["response_rejected"], # 直接透传，纯文本
}
```

chosen 和 rejected **不做格式化**。DPOTrainer 内部会自行拼接 prompt + chosen / prompt + rejected 并 tokenize。

### 后续处理：map + filter（330-353 行）

```python
tokenized_dataset = train_dataset.shuffle().map(
    return_prompt_and_responses,
    batched=True,
    num_proc=args.preprocessing_num_workers,
    remove_columns=train_dataset.column_names,  # 删掉原始5列
)
train_dataset = tokenized_dataset.filter(
    lambda x: 0 < len(x['prompt'] + x['chosen']) <= full_max_length
              and 0 < len(x['prompt'] + x['rejected']) <= full_max_length
)
```

- `batched=True`：批量处理（默认 1000 条）
- `remove_columns`：删掉原始字段，只保留 prompt/chosen/rejected
- filter：按字符数粗略过滤超长样本（真正的 token 级截断由 DPOTrainer 的 `max_length` 控制）

### 与 Reward Modeling 预处理的对比

| | Reward Modeling | DPO |
|---|---|---|
| 预处理输出 | `input_ids_chosen/rejected` + `attention_mask`（已 tokenize） | `prompt` + `chosen` + `rejected`（纯字符串） |
| tokenize 时机 | 预处理阶段自己做 | 交给 DPOTrainer 内部做 |
| prompt + response 拼接 | 预处理时就拼好 | DPOTrainer 内部拼接 |
| DataCollator | 自定义 `RewardDataCollatorWithPadding` | DPOTrainer 内置处理 |

DPO 预处理明显更简单——只给三个字符串，其余全交给 TRL。

---

## 3. 参考模型的巧妙处理（第 503-511 行）

### 为什么 DPO 需要参考模型

DPO 的 loss 公式：

```
loss = -log(sigmoid(β * ((log π_θ(chosen) - log π_ref(chosen))
                        - (log π_θ(rejected) - log π_ref(rejected)))))
```

两个模型的概率：
- `π_θ`：当前正在训练的策略模型（参数在更新）
- `π_ref`：参考模型（SFT 后、DPO 前的模型，参数冻结）

参考模型的作用是**锚定**——防止策略模型跑太远。没有 `π_ref`，模型会极端化：把 chosen 概率推到 1、rejected 推到 0，退化成套话。有了 `π_ref`，loss 衡量的是"**相对于原始模型的偏好偏移量**"，只鼓励适度调整。

### 两种模式

```python
trainer = DPOTrainer(
    model,
    ref_model=None if args.use_peft else deepcopy(model),
    peft_config=peft_config if args.use_peft else None,
)
```

| 场景 | ref_model | 原因 |
|---|---|---|
| LoRA（use_peft=True） | `None` | 关闭 adapter 就是参考模型 |
| 全参数训练 | `deepcopy(model)` | 需要显式拷贝一份冻结副本 |

### LoRA 的巧妙之处

LoRA 模型的权重结构：

```
每一层 = W_base（冻结） + B @ A（可训练的 LoRA 适配器）
```

同一个模型，一个开关切换身份：

```
开启 adapter:  输出 = (W_base + B@A) @ x   → π_θ（策略模型）
关闭 adapter:  输出 = W_base @ x            → π_ref（参考模型）
```

DPOTrainer 在 `_compute_loss` 中的实际操作（第 1175-1183 行）：

```python
if is_peft_model(model) and self.ref_model is None:
    model = self.accelerator.unwrap_model(model)
    with use_adapter(model, adapter_name="ref" if "ref" in model.peft_config else None):
        ref_outputs = self.model(**model_kwargs)  # 关闭 adapter，得到 π_ref
else:
    ref_outputs = self.ref_model(**model_kwargs)   # 用独立的参考模型
```

### 显存对比

| | 全参数训练 | LoRA |
|---|---|---|
| 策略模型 | 完整模型 ~14GB | W_base + LoRA ~14GB + 0.05GB |
| 参考模型 | deepcopy ~14GB | 关闭 adapter = W_base（已在内存里） |
| **总计** | **~28GB** | **~14GB** |

这个技巧适用于所有需要"当前策略 vs 初始策略"对比的算法（DPO、PPO、RLOO）。

---

## 4. DPO 损失函数详解

### 完整计算流程

```
┌──────────────────────────────────────────────────────────────┐
│ 1. 数据集提供: prompt、chosen、rejected（不是模型生成的）      │
│                                                              │
│ 2. 策略模型（开 adapter）对 chosen/rejected 逐 token 算概率:  │
│    per_token_logps → sum → log π_θ(chosen), log π_θ(rejected)│
│                                                              │
│ 3. 参考模型（关 adapter）同样操作:                            │
│    → log π_ref(chosen), log π_ref(rejected)                  │
│                                                              │
│ 4. 算 log ratio:                                             │
│    chosen_logratio  = log π_θ(chosen)  - log π_ref(chosen)   │
│    rejected_logratio = log π_θ(rejected) - log π_ref(rejected)│
│                                                              │
│ 5. 算 loss:                                                  │
│    diff = β * (chosen_logratio - rejected_logratio)          │
│    loss = -log(sigmoid(diff))                                │
│                                                              │
│ 6. 反向传播，只更新 LoRA 参数                                 │
└──────────────────────────────────────────────────────────────┘
```

### 具体数字例子

```
策略模型:  log π_θ(chosen)=-5.0   log π_θ(rejected)=-8.0
参考模型:  log π_ref(chosen)=-6.0  log π_ref(rejected)=-7.0

chosen_logratio  = -5.0 - (-6.0) = +1.0  ← 策略比参考更喜欢 chosen
rejected_logratio = -8.0 - (-7.0) = -1.0  ← 策略比参考更不喜欢 rejected

diff = β * (1.0 - (-1.0)) = β * 2.0

β=0.1:  diff=0.2, sigmoid=0.55, loss=-log(0.55)=0.60  ← 高 loss，继续调整
β=1.0:  diff=2.0, sigmoid=0.88, loss=-log(0.88)=0.13  ← 低 loss，接近收敛
```

diff 越大（策略越偏好 chosen），sigmoid 越接近 1，loss 越接近 0。

### β 的作用

β 是温度系数，控制偏离参考模型的惩罚强度：

| β 值 | 效果 |
|---|---|
| 大（如 1.0） | 偏离代价高，策略保守，不敢离参考模型太远 |
| 小（如 0.1） | 偏离代价低，策略激进，可以大幅调整 |

### loss 的直觉理解

与 Reward Modeling 的 loss 结构一致：

| RM loss | DPO loss |
|---|---|
| `-log(sigmoid(r_chosen - r_rejected))` | `-log(sigmoid(β * (logratio_chosen - logratio_rejected)))` |
| RM 的打分差 | 策略与参考模型的 log ratio 差 |

DPO 相当于把 reward model 的打分隐式地编码在了策略模型和参考模型的概率差异里。

---

## 5. Loss 的粒度：序列级别

### 与 GRPO 的关键区别

```
GRPO:
  per_token_logps → 逐 token 算 ratio → 逐 token 算 clip loss → token 平均 → 序列平均
  loss 粒度: token 级别

DPO:
  per_token_logps → sum 成序列总 logp → 序列级别算 log ratio → sigmoid → loss
  loss 粒度: 序列级别
```

### 源码对应

**第一步：逐 token 算概率，然后 sum（第 1151-1154 行）**

```python
per_token_logps = selective_log_softmax(shift_logits, shift_labels)
per_token_logps[shift_completion_mask == 0] = 0.0  # 只保留回答部分，prompt 部分置零
logps = per_token_logps.sum(dim=1)                  # 求和 → 一条回答一个标量
```

`completion_mask` 确保只计算回答部分的 token 概率，prompt 部分不参与。

**第二步：序列级别算 loss（第 1198-1236 行）**

```python
chosen_logratios = chosen_logps - ref_chosen_logps       # 标量
rejected_logratios = rejected_logps - ref_rejected_logps  # 标量
delta_score = chosen_logratios - rejected_logratios        # 标量
per_sequence_loss = -F.logsigmoid(self.beta * delta_score) # 标量
```

从 delta_score 到 loss，全在序列级别的标量上操作。

### 为什么 DPO 不需要 token 级别 loss

**GRPO** 有逐 token 的 advantage 乘法和 clip 截断，需要 token 级别控制，所以 loss 必须逐 token 算。

**DPO** 没有 advantage，没有 clip。它的数学推导天然在序列级别——比较的是"整条回答的总概率"。

### 反向传播仍然影响每个 token

虽然 loss 是标量，但它来自 `per_token_logps.sum()`。反向传播时梯度通过 `sum` 流回每个 token 的 logp，再流回模型参数。每个 token 的概率都会被调整，只是调整信号是统一的（不像 GRPO 每个 token 有不同的 clip 行为）。

---

## 6. 对应 TRL 源码位置

loss 计算在 `trl/trainer/dpo_trainer.py` 的 `_compute_loss` 方法中：

| 步骤 | 代码位置 | 说明 |
|---|---|---|
| 策略模型前向传播 | 第 1147 行 | `outputs = model(**model_kwargs)` |
| 逐 token 算 logp | 第 1151 行 | `selective_log_softmax(shift_logits, shift_labels)` |
| completion mask | 第 1152 行 | 只保留回答部分，prompt 置零 |
| 序列级 logp | 第 1154 行 | `per_token_logps.sum(dim=1)` |
| 参考模型（LoRA 关 adapter） | 第 1175-1181 行 | `with use_adapter(model, adapter_name=...)` |
| 参考模型（全参数） | 第 1183 行 | `self.ref_model(**model_kwargs)` |
| 算 log ratio | 第 1198-1199 行 | `chosen_logps - ref_chosen_logps` |
| 算 delta score | 第 1231 行 | `chosen_scores - rejected_scores` |
| sigmoid loss | 第 1236 行 | `-F.logsigmoid(self.beta * delta_score)` |

---

## 7. DPO vs GRPO vs RLOO 对比

| | DPO | GRPO | RLOO |
|---|---|---|---|
| 需要 Reward Model | 不需要 | 不需要（规则奖励） | **需要** |
| 训练数据 | 偏好对（chosen/rejected） | 带答案的问题 | prompt |
| 生成回答 | 不生成，数据集已给 | 策略模型生成 G 个 | 策略模型生成 K 个 |
| 参考模型 | 需要（算 log ratio） | 可选（KL 惩罚） | 可选（KL 惩罚） |
| loss 粒度 | 序列级别 | token 级别 | 序列级别 |
| 核心公式 | `-log(σ(β·Δ_logratio))` | PPO-Clip（逐 token） | PPO-Clip（序列级） |
| 优势 | 简单、稳定、不需要在线生成 | 适合有明确答案的任务 | 经典 RLHF 框架 |
| 劣势 | 依赖离线数据质量 | 需要可验证的奖励函数 | 流程复杂、显存大 |
