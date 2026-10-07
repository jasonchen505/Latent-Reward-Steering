# Latent-Reward-Steering 深入学习文档

> 上游：`jiakanglee/Latent-Reward-Steering`（作者 Jiakang Li，仓库描述 "EMNLP2026-Latent Reward Steering"，15 stars）
> 本文基于 fork `jasonchen505/Latent-Reward-Steering` 的 `main` 分支代码（2026-10-07），通读全部 LRS 主线代码后整理。
> 实现基于 `cvenhoff/thinking-llms-interp` 的代码库（README 致谢）。

---

## 1. 一句话核心思想

强推理不仅取决于模型"知道什么"，还取决于推理过程中"何时、如何"部署认知行为。已有的认知行为控制方法（prompt 诱导 / 表示层 steering）都**预先指定了要注入什么行为、沿什么固定方向干预**，可能与局部推理状态不匹配。

LRS 换了一条路：**不指定注入什么行为**，而是

1. 从成功 / 失败的推理轨迹学一个 **latent reward model**，估计中间 SAE latent state 的质量；
2. 推理时用 **reward 梯度**给出 state-specific 的修正方向；
3. 用 **reward–confidence 门控**只在被判为"脆弱"的步上干预，健康的推理步原样放过。

不改模型权重，纯推理时干预。

---

## 2. 三阶段管线

```
Stage 1: Latent Trace Construction   Stage 2: Latent Reward Learning   Stage 3: Online Selective Latent Repair
frozen LLM + SAE 采 10 维 latent 轨迹   轻量 Transformer 学轨迹质量        逐 token 门控 + 梯度上升修 latent
（按最终答案对错打标）                （无行为标注）                    （残差写回 hidden 激活）
```

### Stage 1 — 采 latent 轨迹（`LRS/collect_data/generate_data_7B.py`）

- 在 `model.model.layers[20]` 挂 `SequenceCollector` hook，frozen 的 Open-Reasoner-Zero-7B 在 AIME 2024 + 2025（共 60 题）上 greedy 生成。
- 每步激活的处理链：`x → 减 activation_mean 中心化 → L2 归一化 → SAE encode → scatter 成 10 维稠密 latent`。
- 只保留 `<think>` 标记之后的推理段（`think_idx` 定位），label = 整条轨迹最终答案对错（1.0 / 0.0）。
- 输出 `.pt`，每条含 `latent_seq [T,10]`、`label`、`length`、`think_idx`、`source`。

### Stage 2 — 训 latent reward model（`LRS/train_reward_model/train_latent_classifier_7B.py`）

- 轻量 Transformer：`input_dim=10, d_model=128, nhead=4, num_layers=2, dropout=0.1`，head 为 `Linear(128→64)→ReLU→Dropout→Linear(64→1)→Sigmoid`。
- **关键修复（代码注释原话）**：输入先过 `LayerNorm`。SAE 激活值通常极小或忽大忽小，不做归一化会被位置编码淹没——作者称这是解决"loss 卡在 0.9960 不动"的核心。
- 监督信号很粗：**整条轨迹的每个 step 共享同一个 0/1 label**（最终答案对错），没有任何行为标注。
- `WeightedRandomSampler` 处理正负样本不平衡；BCE loss；AdamW(lr=5e-4, wd=1e-4)；30 epoch；grad clip 1.0；每个 epoch 记"Trend Acc"（只看末步预测与 label 是否一致），保存 best checkpoint。

### Stage 3 — 在线选择性 latent 修复（`LRS/Steering/run_basic_overwrite.py`，核心）

`IterativeTransformerSteeringHook` 挂在 layer 20，`lm_head` 上另挂一个置信度 hook：

- **Prompt 阶段跳过**：`seq_len > 1` 时直接返回（生成时每步 seq_len=1，天然逐 token 触发）。
- 每个生成步：
  1. 取当前 token 的 10 维 latent（`no_grad`）；
  2. RM 打分得 `init_prob`（`return_logits=True`，外部手动 sigmoid）；
  3. **门控**：`init_prob < τ_r` → 修；或 `init_prob ≥ τ_r` 但**上一 token** 的 `max(softmax(logits)) < τ_c` → 修；否则跳过。默认 `τ_r=0.9, τ_c=0.72`。
  4. 修 = 在 latent 上做 K 步**归一化梯度上升**：`z ← z + α · ∇r(z)/‖∇r(z)‖`，目标是 RM logit；
  5. `delta_latent` 经 SAE decoder 投回：`delta_act = delta_latent @ W_dec`，**残差加到原 hidden 激活上**。
- 对照实验 `--steer_every_token` 跳过门控（即论文里的 LRS BASIC，ungated）。
- RM 参数 `requires_grad_(False)` + eval 模式，但用 `torch.enable_grad()` 对**输入 latent**求梯度——梯度只流向 latent，不动 RM。

---

## 3. 主要结果（README 表格，greedy decoding）

**Open-Reasoner-7B**：MATH-500 79.4→**83.8**（+4.4），AIME 2024 16.6→**26.6**（+10.0），AIME 2025 16.6→**26.6**（+10.0），GPQA-Diamond 32.3→**39.4**（+7.1），AMC23 50.0→60.0（+10.0），IneqMath 46.0→**60.0**（+14.0）。
**Open-Reasoner-1.5B**：六项全部提升，AIME 2024 3.3→13.3（+10.0）最显著。

- Ungated 的 LRS BASIC 不稳定（7B 上 AMC23 −5.0、GPQA −1.5）→ **门控是关键**，印证"只修脆弱步"的设计。
- 效率：只干预 27.9% 的 token（门控跳过 72.1%），平均每题触发约 725 次；wall-clock 115.3s→156.2s（**1.35×**）；生成长度几乎不变（2595→2596 tokens）——修的是方向，不是长度。

---

## 4. 超参（每 dataset × model 单独 sweep）

| Dataset | 7B (K, α) | 1.5B (K, α) |
|---|---|---|
| MATH-500 | 1, 1.400 | 1, 0.100 |
| AIME24 | 2, 0.295 | 4, 0.900 |
| AIME25 | 3, 1.320 | 2, 0.400 |
| GPQA-Diamond | 4, 1.150 | 3, 1.000 |
| AMC23 | 1, 1.400 | 4, 0.300 |
| IneqMath | 2, 0.700 | 2, 1.100 |

`τ_r` 基本 0.9、`τ_c` 基本 0.72（仅 7B MATH-500 用 0.8/0.69）。`Steering/` 下有大量 `*_steer_sweep_*.slurm` 脚本——K/α 是扫出来的，换新任务大概率要重扫。

---

## 5. 关键代码细节与工程观察

1. **"SAE"实际是球面 k-means**。`train-saes/train_clustering.py` 的方法是 `spherical_kmeans` / `pca_kmeans`，10 个聚类中心构成字典（checkpoint 名 `sae_{model_id}_layer{layer}_clusters{n_clusters}.pt`）。且 collect/steering 里都执行 `sae.k = n_clusters`，而 `encode` 是 `topk(self.k)`——**k 等于字典大小，top-k 取全部，scatter 是稠密的**。"稀疏"只剩名字，10 维 latent 本质是 10 个 cluster 的 affinity 向量。
2. **Steering 时 RM 只看到单步**（`[B,1,10]`）。训练喂的是完整轨迹，但推理打分和求梯度都是单 token 窗口。单 token 上 2 层 Transformer 的自注意力退化（只 attend 自己），位置编码恒为 position 0 → 实际起作用的是浅层映射（LayerNorm→Linear→Transformer层→MLP head）。"Transformer RM"的时序建模能力在 steering 时用不上——这是代码层面最值得指出的 gap（自然的改进：喂滑动窗口）。
3. **权重加载的坑**：steering 脚本的 `LatentTransformer` 在 head 里多塞了一个 `nn.Dropout`，注释写明"只为让第二层 Linear 的权重名对上 `head.3.weight`"，同时把 Sigmoid 挪到 forward 外以支持 `return_logits=True`。训练脚本和推理脚本的模型定义是**故意不对齐又对齐**的，改任一边都会炸。
4. **SAE checkpoint 校验很严**：`load_sae` 要求 checkpoint 内嵌 `activation_mean`，并校验其 shape / 有限性 / 与 model_id、layer 的一致性，甚至与落盘的 mean pkl 做逐元素比对。checkpoint 需自备（仓库 `.gitignore` 排除了 artifacts）。
5. **nnsight 的坑与 workaround**：模型经 nnsight 包装；`_unwrap_hf_pretrained_for_generate` 层层剥到 `PreTrainedModel` 再调 `.generate`，注释写明原因是 nnsight 的 generate 在深层栈上会因 `inspect.getsourcelines` 抛 `OSError` 导致整 shard 挂掉。
6. **判题全走规则，零 API 花费**：`evaluate_answer` 对 math 用 `math_verify`，GPQA 用规则抽字母，MBPP 用可执行测试。`utils/llm_judge.py` 里留的 DeepSeek judge 只服务于 `classification` 类型，主流程不用——但注意该文件**硬编码了上游作者的 DeepSeek API key**（已公开提交，fork 里同样存在；不要使用，必要时删掉或转环境变量）。
7. **分析工具链齐全**：`--save_token_records`（每 token 的 reward/conf/should_steer）、`--save_rm_path_scores`（steer 轨迹全长 RM 打分，`p_last`/`p_mean`）、`--save_sae_steer_trace`（只记实际发生 steer 的步的 pre/post latent）；`--filter_from_judge_dir` + `--steer_only` 组合可跳过已双对的题、只重跑 steer 并从旧 judge 回填 base，省算力。
8. **可解释性脚本**：`LRS/Interpretability/Sae_interpt.py`，`sae.k=1` 模式下为每维收集 top 激活的上下文（max activating examples），用来给 10 个维度做定性解释。
9. **遗留目录**：`train-saes/`（除 clustering 外的可视化/评估脚本）、`train-vectors/`（steering vector 实验）、`hybrid/`、`visualize-saes/`、`messages/` 继承自 thinking-llms-interp，非 LRS 论文主线；LRS 主线只有 `LRS/`（collect_data / train_reward_model / Steering / Interpretability）+ `utils/`。

---

## 6. 批判性思考

1. **README "zero-shot" 与代码矛盾**。现代码里数学题硬拼接了 5-shot 示例（`_FEWSHOT_MATH` 前置于题干）；GPQA 把示例拼在题干**后面**；AMC23 拼的居然是 `_FEWSHOT_INEQMATH`（疑似复制粘贴错误）；`_COT_SUFFIX` 定义了但从未被使用——表格里 CoT 那列在现有代码中没有对应实现。表格数字出自哪个代码版本，无法从仓库内确认；复现/引用数字时需留意。
2. **监督信号粗糙**：整条轨迹共享一个 0/1 label，最终答对的轨迹其中间的弯路、幻觉步也被标为正例——label noise；且训练数据只有 60 条轨迹（AIME24+25 各 30）。RM 学到的可能是"答对轨迹的平均 latent 风格"而非真正的逐步质量。
3. **单步 RM 输入**（见 §5.2）：训练-推理的输入分布不一致（完整轨迹 vs 单 token），梯度方向的有效性依赖"单步 latent 足以反映局部质量"这一隐含假设。
4. **解码残差的启发式**：encode 时做了中心化+L2 归一化，但 `delta_latent @ W_dec` 直接加回**未归一化**的激活。delta 层面 centering 相消没问题，L2 归一化的逆变换被跳过了——残差修正的幅度语义是近似的。
5. **超参泛化成本**：K/α 每任务单独 sweep，τ 相对稳定但也调过（MATH-500 7B 例外）。把 LRS 搬到新模型/新任务上，sweep 是必经之路。
6. **可解释性弱于显式方法**：与 LeJudge（judge 只读词不读数、算术全在代码里、约束文本与事实分字段防注入）相比，LRS 的修正方向是黑盒梯度——"修了什么"只能靠事后看 steer trace 和维度解释，无法给出符号化理由。这是隐式方法的固有代价。
7. **与 ARCHER 的对偶关系**：ARCHER 是"失败后做离散恢复动作路由"（reflect/replan/escalate 三选一 + CRC 控成本）；LRS 是"失败前在 latent 层面做连续预防性修正"。一个管动作选择，一个管表示修正，思路互补。

---

## 7. 可复现 / 可动手点

- **复现链路完整**：`collect → train → steer` 都有脚本和 slurm；需要自备 SAE checkpoint（`train-saes/results/vars/saes/`）、A4500 级别 GPU、`uv` 环境。
- **自然的改进点**：把 steering 时的 RM 输入从单步改成滑动窗口，让 Transformer 的时序建模真正用上；或把 `τ_r/τ_c` 做成按轨迹自适应的。
- **先验证再引用**：README 表格的数字与现代码的 few-shot/CoT 实现对不上，写论文或做对比实验前建议先用现代码重跑一组小规模对照，确认基线定义。

---

*文档位置：`notes/learning` 分支，`main` 未动。学习日期：2026-10-07。*
