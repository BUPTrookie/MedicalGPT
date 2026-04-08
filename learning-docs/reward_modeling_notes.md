# Reward Modeling 学习笔记

基于 `reward_modeling.py` 的逐步解析，涵盖模型加载、数据处理、损失函数、LoRA 注入等核心环节。

---

## 1. 模型加载：如何把 LLM 改造成奖励模型

### AutoConfig.from_pretrained

```python
config = AutoConfig.from_pretrained(
    model_args.model_name_or_path,
    num_labels=1,
    torch_dtype=torch_dtype,
    trust_remote_code=model_args.trust_remote_code,
    cache_dir=model_args.cache_dir
)
```

- `num_labels=1` 是关键参数。设为 1 表示模型的分类头只输出一个标量值，即奖励分数（reward score）。不是分类任务，而是回归任务。
- 需要单独加载 config 的原因：原始预训练模型的 config 里没有 `num_labels` 字段或默认值不是 1，需要先创建带 `num_labels=1` 的 config 再传给模型。

### AutoModelForSequenceClassification.from_pretrained

```python
model = AutoModelForSequenceClassification.from_pretrained(
    model_args.model_name_or_path,
    config=config,
    torch_dtype=torch_dtype,
    load_in_4bit=model_args.load_in_4bit,
    load_in_8bit=model_args.load_in_8bit,
    device_map=model_args.device_map,
    trust_remote_code=model_args.trust_remote_code,
)
```

模型结构变化：

- 原始 LLM 的输出头 `lm_head`：`hidden_size → vocab_size`（如 3584 → 151936）
- 替换后的 `score` 层：`hidden_size → 1`（如 3584 → 1），只输出一个标量

本质就是 `nn.Linear(hidden_size, num_labels)`。`num_labels=1` 就是回归输出一个分数，`num_labels=2` 就是二分类输出两维。

---

## 2. 数据预处理：preprocess_reward_function

### 输入格式

```python
{
    "system": ["你是一个医疗助手"],
    "history": [[["之前的问题", "之前的回答"]]],
    "question": ["头疼怎么办？"],
    "response_chosen": ["建议先休息，必要时就医检查。"],
    "response_rejected": ["吃点药就行。"]
}
```

### 处理流程

**第1步**：初始化空容器，包含 `input_ids_chosen`、`attention_mask_chosen`、`input_ids_rejected`、`attention_mask_rejected` 四个空列表。

**第2步**：提取 system prompt。

**第3步**：拼接对话历史 + 当前问答，chosen 和 rejected 分别拼：

```python
chosen_messages = [["之前的问题", "之前的回答"], ["头疼怎么办？", "建议先休息，必要时就医检查。"]]
rejected_messages = [["之前的问题", "之前的回答"], ["头疼怎么办？", "吃点药就行。"]]
```

两段文本只有回答部分不同，问题和历史完全一样。

**第4步**：套用对话模板（如 vicuna 模板），生成完整的 prompt 文本。

**第5步**：tokenizer 分词，得到 `input_ids`（token ID 列表）和 `attention_mask`（此时全为 1，尚未 padding）。

### 输出格式

```python
{
    "input_ids_chosen":      [[1, 319, 13563, ..., 2]],   # List[List[int]]
    "attention_mask_chosen":  [[1, 1, 1, ..., 1]],
    "input_ids_rejected":    [[1, 319, 13563, ..., 2]],
    "attention_mask_rejected": [[1, 1, 1, ..., 1]],
}
```

外层列表长度 = batch 中的样本数，内层列表长度 = 每条文本的 token 数（chosen 和 rejected 长度通常不同）。

---

## 3. attention_mask 的作用

告诉模型哪些 token 是真实内容（1），哪些是 padding 填充（0）。

```
样本1: [你, 好, 吗, PAD, PAD]    attention_mask: [1, 1, 1, 0, 0]
样本2: [头, 疼, 怎, 么,  办]    attention_mask: [1, 1, 1, 1, 1]
```

没有 mask 的话，self-attention 会把 PAD 当成有意义的内容去计算注意力，污染结果。有了 mask，PAD 位置的注意力权重会被置为负无穷（softmax 后趋近 0），等于"看不见"。

在 `preprocess_reward_function` 里 attention_mask 全是 1（还没 padding）。真正的 padding 发生在 `RewardDataCollatorWithPadding` 中。

---

## 4. Batch 组装：RewardDataCollatorWithPadding

### 为什么需要自定义 DataCollator

标准 DataCollator 只处理一组 `input_ids`，但奖励模型每条样本有两组（chosen + rejected），需要自定义。

### 处理流程（以 batch_size=3 为例）

**第1步：拆分**。把每条样本的 chosen 和 rejected 拆成两个独立列表，key 从 `input_ids_chosen` 改为标准的 `input_ids`，以便 `tokenizer.pad()` 处理。

**第2步：分别 padding**。chosen 组和 rejected 组各自 pad 到组内最长长度：

```python
# chosen 组 pad 到 5
batch_chosen["input_ids"] = tensor([
    [1, 50, 80, 90, 2],
    [1, 60, 70,  2, 0],    # PAD 位置 attention_mask 为 0
    [1, 40,  2,  0, 0],
])

# rejected 组 pad 到 6
batch_rejected["input_ids"] = tensor([
    [1, 50, 80,  2,  0,  0],
    [1, 60,  2,  0,  0,  0],
    [1, 40, 55, 66, 77,  2],
])
```

chosen 和 rejected 是各自独立 pad 的，pad 后的长度可以不同，因为它们会分别送入模型。

**第3步：组装最终 batch**。把两组的 `input_ids` 和 `attention_mask` 加上 `_chosen`/`_rejected` 后缀合并，加上 `"return_loss": True` 告诉 Trainer 需要计算 loss。

---

## 5. 损失函数：compute_loss

### 公式

单条样本：

```
loss = -log(sigmoid(r_chosen - r_rejected))
```

多条样本取平均：

```
loss = -(1/N) * sum(log(sigmoid(r_chosen_i - r_rejected_i)))
```

其中 `sigmoid(x) = 1 / (1 + e^(-x))`。

### 直觉理解

本质上是二分类交叉熵，分类的对象是"谁更好"：

| chosen - rejected 的差值 | sigmoid 值 | loss 大小 | 含义 |
|---|---|---|---|
| 很大（如 +10） | 约 1.0 | 约 0（几乎无损失） | 模型确信 chosen 更好 |
| 接近 0（如 +0.1） | 约 0.5 | 约 0.69（高损失） | 模型分不清谁好谁差 |
| 负数（如 -3） | 约 0.05 | 约 3.0（很高损失） | 模型判断反了 |

### 目标不是让差距无限大

当差距已经比较大时（如 diff=5，loss=0.007），loss 接近 0，梯度也接近 0，模型不会继续拉大差距。这是 sigmoid 的饱和特性决定的。

目标是：**让模型有信心地判断 chosen 比 rejected 好就够了**，不追求差距最大化。

### 为什么用这个 loss

来自 InstructGPT 论文（OpenAI, 2022），符合 Bradley-Terry 模型：

- `P(chosen > rejected) = sigmoid(r_c - r_r)`，分数差通过 sigmoid 转化为"chosen 优于 rejected 的概率"
- loss 就是最大化这个概率的负对数似然
- 不需要绝对分数标签，只需要"A 比 B 好"的相对偏好，人类标注成本低

### 代码中 logsigmoid 的意义

`torch.nn.functional.logsigmoid` 把 log 和 sigmoid 合成一步计算，数值上更稳定，避免先算 sigmoid 得到接近 0 的值再取 log 导致精度问题。

---

## 6. LoRA 注入

### LoRA 的核心思想

原始权重矩阵 W 冻结不动，旁边插入一个低秩分解：

```
W' = W + (alpha / r) * B @ A
```

其中 B 的 shape 是 (d, r)，A 的 shape 是 (r, d)，r 远小于 d。

以 7B 模型的 q_proj（4096 x 4096）为例，r=8 时：

- 原始参数量：4096 x 4096 = 16,777,216（冻结）
- LoRA 参数量：4096 x 8 + 8 x 4096 = 65,536（可训练）
- 压缩比：0.4%

### 前向传播

```
output = W @ x + B(A(x))
```

x 是这一层的输入向量（如上一层传来的 hidden state）。输入先过 A 得到 r 维小向量，再过 B 映射回原维度，和原始输出相加。全程不需要显式算出 BA 这个大矩阵。

### 反向传播

A 和 B 的训练没有任何特殊处理，就是标准的链式求导，同时更新：

- 对 B 求导：需要 Ax（r 维小向量）
- 对 A 求导：需要 B 传回来的梯度和原始输入 x

每一步参与运算的都是小矩阵和向量之间的乘法，不需要算出 BA 大矩阵，所以计算量和存储都很小。

### 省显存的真正原因

冻结了大矩阵 W，只训练小矩阵 A 和 B。训练时最吃显存的三样东西——梯度、优化器状态（Adam 的一阶/二阶动量）、激活值缓存——都和可训练参数量正相关：

| 项目 | 全量微调 | LoRA (r=8) |
|---|---|---|
| 需要梯度的参数量 | 1677万 | 6.5万 |
| Adam 优化器额外开销 | 1677万 x 2 = 3354万 | 6.5万 x 2 = 13万 |

底座模型的 W 冻结，不算梯度，不存优化器状态，只需要存权重本身（还能用 8bit/4bit 量化进一步压缩）。

### 代码流程

**第1步：8bit 预处理**

```python
if model_args.load_in_8bit:
    model = prepare_model_for_kbit_training(model)
```

冻结所有原始参数，把 LayerNorm 等层转回 float32 确保训练稳定。

**第2步：确定目标层**

`find_all_linear_names` 遍历模型所有模块，找出所有 Linear 层，排除 `lm_head` 和 `score`。结果如：

```python
["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
```

**第3步：配置并注入 LoRA**

```python
peft_config = LoraConfig(
    task_type=TaskType.SEQ_CLS,   # 声明是序列分类任务
    target_modules=target_modules,
    r=8, lora_alpha=32, lora_dropout=0.05,
    modules_to_save=modules_to_save,
)
model = get_peft_model(model, peft_config)
```

注入后每个目标层变为：

```
q_proj: LoraLinear(
    base: Linear(4096, 4096)   ← 冻结
    lora_A: Linear(4096, 8)    ← 可训练
    lora_B: Linear(8, 4096)    ← 可训练
)
```

**第4步：精度转换**。LoRA 参数转 float32（底座可以保持 float16/int8 省显存），保证梯度更新数值稳定。

### 与 SFT 阶段 LoRA 的差异

**TaskType 不同**：`TaskType.SEQ_CLS`（序列分类）vs `TaskType.CAUSAL_LM`（因果语言模型）。本质上是告诉 PEFT 框架模型的结构类型，让它知道哪些层该处理、哪些该跳过。

**排除 score 层**：`score` 层是 `Linear(hidden_size, 1)`，参数量极小（如 4096 个），对这么小的层用 LoRA 没有意义。通过 `modules_to_save` 参数让 `score` 层解冻全部参数直接完整训练。

| 层 | 策略 | 原因 |
|---|---|---|
| Transformer 内部的线性层 | 注入 LoRA | 参数量大，用低秩分解省显存 |
| score（输出头） | 完整训练 | 参数量极小，不需要 LoRA |
