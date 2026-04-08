# GRPO 训练学习笔记

基于 `training/grpo_training.py` 及 TRL `GRPOTrainer` 源码的逐步解析，涵盖奖励函数、advantage 计算、loss 公式与聚合流程。

---

## 1. 整体流程

```
┌───────────────────────────────────────────────────────────────┐
│ 1. 对每个 prompt，策略模型生成 G 个回答                        │
│ 2. 两个奖励函数分别打分：accuracy_reward + format_reward       │
│ 3. 加权求和合并为一个标量 reward                               │
│ 4. 算 advantage（归一化后的"比平均水平好多少"）                │
│                                                               │
│ 5. 策略更新：                                                  │
│    a. 用当前参数重新算每个 token 的 logp_new                   │
│    b. ratio = exp(logp_new - logp_old)，逐 token              │
│    c. PPO-Clip loss：min(ratio*adv, clamp(ratio)*adv)         │
│    d. 先序列内 token 平均，再跨序列平均                        │
│    e. 反向传播，更新参数                                       │
│                                                               │
│ 6. 这批数据用完，回到第 1 步，用新策略重新生成                 │
└───────────────────────────────────────────────────────────────┘
```

---

## 2. 奖励函数 1：准确度奖励（accuracy_reward，第 71-116 行）

### 函数签名

```python
def accuracy_reward(completions, answer, **kwargs):
```

- `completions`：模型生成的回答列表，`completion[0]["content"]` 取出纯文本
- `answer`：标准答案列表，与 completions 一一对应
- 返回：reward 列表，每个值为 `1.0`（对）或 `0.0`（错）

### 两条处理路径

**路径 1：GSM8K 格式**（标准答案含 `####`）

GSM8K 数据集答案格式：`"计算过程...\n#### 72"`，`####` 后面是最终数字答案。

```python
gold_parsed = parse(sol.split("####", 1)[-1].strip())  # "72" → 数字 72
answer_parsed = parse(extract_answer(content))          # 从 <answer>72</answer> 提取
```

`extract_answer`（第 61-68 行）用正则 `<answer>(.*?)</answer>` 提取模型回答中的答案标签内容。

**路径 2：LaTeX 数学格式**（标准答案不含 `####`）

```python
gold_parsed = parse(sol, extraction_mode="first_match",
                    extraction_config=[LatexExtractionConfig()])
```

对模型回答的解析更严格，配置了 `NormalizationConfig`：
- `boxed="all"`：优先从 `\boxed{...}` 提取答案
- `boxed_match_priority=0`：boxed 匹配优先级最高
- `malformed_operators=False`：不接受格式错误的算符

### 验证

```python
reward = float(verify(answer_parsed, gold_parsed))  # True→1.0, False→0.0
```

`verify` 来自 `math_verify` 库，做数学等价性验证（如 `72` 和 `72.0` 相等，`\frac{1}{2}` 和 `0.5` 相等），不是简单字符串比较。

### 具体例子

```
completions:  ["<answer>72</answer>",  "<answer>15</answer>",  "$\\boxed{3x+1}$"]
answer:       ["...\n#### 72",         "...\n#### 20",         "3x+1"]

样本1: gold=72, parsed=72  → verify=True  → 1.0
样本2: gold=20, parsed=15  → verify=False → 0.0
样本3: gold=3x+1, parsed=3x+1 → verify=True → 1.0

返回 [1.0, 0.0, 1.0]
```

这是**二值奖励**：全对给 1，不对给 0，没有中间分。

---

## 3. 奖励函数 2：格式奖励（format_reward，第 119-127 行）

### 正则匹配

```python
def format_reward(completions, **kwargs):
    pattern = r"<think>.*?</think><answer>.*?</answer>$"
    completion_contents = [completion[0]["content"] for completion in completions]
    matches = [re.match(pattern, content) for content in completion_contents]
    rewards = [1.0 if match else 0.0 for match in matches]
    return rewards
```

要求模型输出严格为：`<think>推理过程</think><answer>最终答案</answer>`

- `re.match`：从字符串**开头**匹配，前面不能有多余内容
- `$`：必须以 `</answer>` 结尾，后面不能有多余内容
- 默认不开启 `re.DOTALL`，`.*?` 不匹配换行符

### 匹配示例

| 模型输出 | 匹配？ | reward | 原因 |
|---|---|---|---|
| `<think>先算加法</think><answer>72</answer>` | Yes | **1.0** | 完美格式 |
| `答案是72` | No | **0.0** | 没有标签 |
| `<think>推理</think>多余<answer>72</answer>` | No | **0.0** | 标签之间有多余文本 |
| `<answer>72</answer>` | No | **0.0** | 缺少 `<think>` |
| `<think>推理</think><answer>72</answer>多余` | No | **0.0** | `$` 要求不能有尾部内容 |

### 设计目的

配合 SYSTEM_PROMPT（第 130-135 行）要求模型用 `<think>` 和 `<answer>` 标签组织输出，DeepSeek-R1 风格。accuracy_reward 管"答对"，format_reward 管"格式对"，两者配合让模型既学会推理又学会规范输出。

---

## 4. 奖励合并与 advantage 计算

### 第一步：两个奖励函数各自打分

对同一个 prompt 生成 G=4 个回答：

```
              accuracy_reward    format_reward
回答A:            1.0               1.0
回答B:            0.0               1.0
回答C:            1.0               0.0
回答D:            0.0               0.0
```

### 第二步：加权求和

代码未指定 `reward_weights`，默认都是 `1.0`：

```
回答A: 1.0 + 1.0 = 2.0
回答B: 0.0 + 1.0 = 1.0
回答C: 1.0 + 0.0 = 1.0
回答D: 0.0 + 0.0 = 0.0
```

两个奖励函数**不产生两个 loss**，而是先合并为一个标量 reward，再走统一的优化流程。

### 第三步：算 advantage

GRPO 使用整组均值作为 baseline（区别于 RLOO 的 leave-one-out）：

```
mean = (2.0 + 1.0 + 1.0 + 0.0) / 4 = 1.0
std  = 0.816

advantage_A = (2.0 - 1.0) / 0.816 = +1.22   ← 最好
advantage_B = (1.0 - 1.0) / 0.816 =  0.00   ← 平均
advantage_C = (1.0 - 1.0) / 0.816 =  0.00   ← 平均
advantage_D = (0.0 - 1.0) / 0.816 = -1.22   ← 最差
```

advantage 是**序列级别**的，同一条回答的所有 token 共享同一个 advantage 值。

---

## 5. 从 advantage 到 loss 的完整计算过程

### 设定

以回答 A（"多喝水"，advantage = +1.22）和回答 B（"吃点药"，advantage = -1.22）为例。

### 5.1 取出旧策略的 logp_old（生成时已存好）

```
回答A "多喝水" → [多, 喝, 水]
  logp_old_0 = -2.0  (P_old(多) ≈ 0.135)
  logp_old_1 = -1.5  (P_old(喝) ≈ 0.223)
  logp_old_2 = -0.8  (P_old(水) ≈ 0.449)

回答B "吃点药" → [吃, 点, 药]
  logp_old_0 = -1.8  (P_old(吃) ≈ 0.165)
  logp_old_1 = -2.1  (P_old(点) ≈ 0.122)
  logp_old_2 = -1.0  (P_old(药) ≈ 0.368)
```

### 5.2 用当前策略重新算 logp_new

**不是重新生成**，而是把同样的 token 序列喂给当前模型重新算概率：

```
回答A：
  logp_new_0 = -1.7  (P_new(多) ≈ 0.183)
  logp_new_1 = -1.2  (P_new(喝) ≈ 0.301)
  logp_new_2 = -0.9  (P_new(水) ≈ 0.407)

回答B：
  logp_new_0 = -2.1  (P_new(吃) ≈ 0.122)
  logp_new_1 = -2.5  (P_new(点) ≈ 0.082)
  logp_new_2 = -1.3  (P_new(药) ≈ 0.273)
```

### 5.3 逐 token 算 ratio

`ratio = exp(logp_new - logp_old)` 是概率空间的 P_new / P_old：

```
回答A（好回答）：
  ratio_0 = exp(-1.7 - (-2.0)) = exp(+0.3) = 1.35   ← 概率提高了
  ratio_1 = exp(-1.2 - (-1.5)) = exp(+0.3) = 1.35
  ratio_2 = exp(-0.9 - (-0.8)) = exp(-0.1) = 0.905  ← 概率略降

回答B（差回答）：
  ratio_0 = exp(-2.1 - (-1.8)) = exp(-0.3) = 0.741  ← 概率降低了
  ratio_1 = exp(-2.5 - (-2.1)) = exp(-0.4) = 0.670
  ratio_2 = exp(-1.3 - (-1.0)) = exp(-0.3) = 0.741
```

### 5.4 逐 token 算 PPO-Clip loss

ε = 0.2，clamp 范围 [0.8, 1.2]。

**注意：ratio 本身没有被截断。** 是分别算无截断项和截断项，取 min 再取负。

#### 回答 A（advantage = +1.22）

**Token 0：ratio = 1.35 > 1.2，触发截断**

```
clipped_ratio = clamp(1.35, 0.8, 1.2) = 1.2
coef_1 = 1.35 × 1.22 = 1.647    ← 无截断
coef_2 = 1.2  × 1.22 = 1.464    ← 截断后
L_0 = -min(1.647, 1.464) = -1.464    ← 封顶收益
```

**Token 1：ratio = 1.35 > 1.2，同上**

```
L_1 = -1.464
```

**Token 2：ratio = 0.905，在范围内，不截断**

```
coef_1 = coef_2 = 0.905 × 1.22 = 1.104
L_2 = -1.104    ← 梯度正常流过，继续调整
```

#### 回答 B（advantage = -1.22）

**Token 0：ratio = 0.741 < 0.8，触发截断**

```
clipped_ratio = clamp(0.741, 0.8, 1.2) = 0.8
coef_1 = 0.741 × (-1.22) = -0.904
coef_2 = 0.8   × (-1.22) = -0.976
L_0 = -min(-0.904, -0.976) = -(-0.976) = +0.976    ← 封底惩罚
```

**Token 1：ratio = 0.670 < 0.8，触发截断**

```
coef_1 = 0.670 × (-1.22) = -0.817
coef_2 = 0.8   × (-1.22) = -0.976
L_1 = +0.976
```

**Token 2：ratio = 0.741 < 0.8，同 Token 0**

```
L_2 = +0.976
```

### 5.5 聚合：先序列内平均，再跨序列平均

```
回答A 的 loss = (-1.464 + -1.464 + -1.104) / 3 = -1.344   ← 负值，奖励项
回答B 的 loss = (+0.976 + 0.976 + 0.976)   / 3 = +0.976   ← 正值，惩罚项

最终 loss = (-1.344 + 0.976) / 2 = -0.184
```

优化器做梯度下降让 loss 变小：
- 好回答部分（loss 为负）→ 让它更负 → 提高这些 token 的概率
- 差回答部分（loss 为正）→ 让它趋近 0 → 降低这些 token 的概率

---

## 6. advantage 的粒度问题

### 为什么每个 token 用同一个 advantage？

advantage 是对**整条回答**的评价（序列级别），但 loss 是逐 token 算的，每个 token 乘以同一个 advantage。

这是 GRPO（REINFORCE 系列）的核心假设：**不知道哪个 token 贡献最大，对所有 token 施加相同信号。**

### 模型如何学到哪些 token 更关键？

靠**跨回答的统计平均**。同一个 prompt 的多个回答：

```
回答A: "<think>先算加法得72</think><answer>72</answer>"   advantage = +1.22
回答B: "<think>先算加法得72</think><answer>15</answer>"   advantage = -1.22
```

A 和 B 大部分 token 相同，只有 `72` 和 `15` 不同。经过大量训练：
- token `72`（出现在正确答案位置）被反复鼓励
- token `15`（出现在错误答案位置）被反复抑制
- 共有的 token `<think>先算加法得` 正负信号抵消，不受明显影响

**单条回答内一视同仁，但统计上关键 token 和噪音 token 自然分化。**

### 与标准 PPO 的对比

```
             advantage 粒度     需要 Value model     采样量要求
PPO+GAE       逐 token            需要                低
GRPO          整条序列            不需要              高（每 prompt 生成 G 个）
```

GRPO 的 `num_generations`（G）越大，统计信号越清晰。

---

## 7. 对应 TRL 源码位置

loss 计算在 `trl/trainer/grpo_trainer.py` 的 `_compute_loss` 方法中：

| 步骤 | 代码位置 | 说明 |
|---|---|---|
| 算 logp_new | 第 2117-2131 行 | `_get_per_token_logps_and_entropies(model, ...)` |
| 取 logp_old | 第 2149-2150 行 | `inputs.get("old_per_token_logps")` |
| advantage 广播 | 第 2139-2143 行 | `advantages.unsqueeze(1)` 从 `(B,)` 变 `(B,1)` 广播到所有 token |
| 算 log_ratio | 第 2167 行 | `per_token_logps - old_per_token_logps` |
| 算 ratio | 第 2179 行 | `coef_1 = torch.exp(log_importance_weights)` |
| PPO-Clip | 第 2196-2204 行 | `coef_2 = clamp(coef_1, ...)`, `per_token_loss = -min(coef_1*adv, coef_2*adv)` |
| 聚合 loss | 第 2225-2228 行 | 序列内 token 平均，再跨序列平均 |
