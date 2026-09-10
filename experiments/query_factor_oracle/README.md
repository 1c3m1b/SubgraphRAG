# SubgraphRAG Phase 0–2：Gold Query Factors / Oracle Headroom

本目录是在不修改 `retrieve/` 与 `reason/` 核心逻辑的前提下，为 WebQSP、CWQ 增加的一套独立实验层。它解决三个问题：固定一个可审计 baseline；把 gold logical form 转成宽松但显式的 query factors；只在原 retrieval 保存的候选三元组中做 oracle reranking，量化 query prior 的真实上限。

论文与代码的对应关系如下：SubgraphRAG 的 retriever 以 question/entity/relation embedding、topic positional encoding 和双向 DDE 为每条候选边打分；`retrieve/inference.py` 最终只保存最多 500 条带 sigmoid score 的三元组。Reasoning 阶段的 `reason/preprocess/prepare_prompts.py` 对三元组按 `(h,r,t)` 去重后取前 100，再交给固定 LLM。因此本实验把“保存的 Top-500”作为唯一候选池、保持 Top-100 prompt 和最终 LLM 不变，只改变 query-side prior 导致的排列。它是相对完整候选图的保守 headroom，不是无约束 oracle。

参考资料：

- [SubgraphRAG 论文](https://arxiv.org/abs/2410.20724)
- [SubgraphRAG 官方代码](https://github.com/Graph-COM/SubgraphRAG)
- [WebQSP 格式说明](https://ad-research.cs.uni-freiburg.de/benchmarks/webqsp/doc/WebQSP.pdf)
- [ComplexWebQuestions 官方页面](https://www.tau-nlp.sites.tau.ac.il/compwebq)
- [ChatKBQA（可复用 SPARQL/S-expression 数据与转换器）](https://github.com/LHRLAB/ChatKBQA)

## 实验边界

- 主 cohort 永远是 frozen retrieval PTH 的有序 key；不按 HF 数据行号连接。
- 缺 gold LF、解析失败、question/数据版本不一致的样本全部保留，oracle 排序严格回退 baseline。
- 不使用 `a_entity`、shortest-path label、GPT-labelled triple 或最终答案参与排序。
- 不新增候选边，不重新训练 retriever，不改变 prompt、LLM 或答案匹配协议。
- reranked PTH 保留每条边的原 retriever score；列表位置表示新 rank。必须保持 `threshold=0`。
- upstream evaluator 的 `Score_h` 硬编码了作者 baseline PTH。变体实验不报告这个错误口径；Hit、Hit@1、Macro/Micro F1 继续复用原 evaluator 的答案解析与匹配函数。

## 目录与产物

```text
experiments/query_factor_oracle/
├── configs/{webqsp,cwq}.json
├── phase0_freeze.py
├── phase1_build_factors.py
├── phase2_rerank.py
├── run_reasoning.py
├── evaluate_reasoning.py
├── build_report.py
├── build_combined_report.py
├── preflight.py
├── run_remote.sh
├── qforacle/                  # 实验实现，不侵入原代码
└── tests/                     # CPU 合成测试
```

一次完整远程运行会生成：

```text
$EXP_ROOT/{webqsp,cwq}/
├── phase0/
│   ├── baseline.lock.json
│   ├── manifest.json
│   ├── validation.json
│   ├── environment.json
│   ├── retrieval/baseline_retrieval.pth
│   ├── retrieval/ranked_samples.jsonl.gz
│   └── reasoning/replicate_{0,1}/
├── phase1/
│   ├── query_factors.jsonl.gz
│   └── summary.json
├── phase2/
│   ├── retrieval_summary.json
│   └── <variant>/
│       ├── retrieval_result.pth
│       ├── rerank_diagnostics.jsonl.gz
│       ├── sample_metrics_at_<K>.jsonl.gz
│       └── reasoning/replicate_0/
└── report/
    ├── final_report.json
    └── final_report.md
```

## 远程服务器准备

本地仓库没有数据、retrieval PTH 和 Llama 权重，所以这里只能做代码与合成验证；WebQSP/CWQ 的真实数字必须在服务器生成。

1. 在服务器 checkout 与本次实验相同的 SubgraphRAG commit，准备原 baseline PTH、产生它的 retriever checkpoint、RoG prediction JSONL 和 Llama 权重。默认路径与原仓库一致；路径不同就复制一份 config 到服务器并修改，不要修改核心代码。

2. 使用两个环境。retriever 复跑严格沿用上游 `retrieve/README.md` 的 Python 3.10、Torch 2.1/CUDA 12.1、PyG 组合；Phase 0–2 与 reasoning 沿用上游 `reason/README.md` 的 Python 3.10.14、Torch 2.4.0、vLLM 0.5.5、OpenAI 1.50.2，并补齐 `datasets transformers huggingface_hub numpy tqdm networkx`。不要为了让两个 GPU 栈共存而随意升级核心包；`preflight.json` 与 `phase0/environment.json` 会记录真实版本。

3. 准备并固定 ChatKBQA checkout。不要只依赖会移动的 `main`：

```bash
git clone https://github.com/LHRLAB/ChatKBQA.git /data/ChatKBQA
export CHATKBQA_ROOT=/data/ChatKBQA
export CHATKBQA_COMMIT=请替换为固定的40位commit
git -C "$CHATKBQA_ROOT" checkout "$CHATKBQA_COMMIT"
git -C "$CHATKBQA_ROOT" rev-parse HEAD
```

WebQSP 使用 `QuestionId` 对齐；CWQ v1.1 使用唯一 `ID` 对齐，不能用会重复的 `webqsp_ID`。建议同时保存 ChatKBQA commit；脚本会记录输入文件 SHA256。

4. 使用独立 HF cache。配置已经 pin 数据集 revision：

- `ml1996/webqsp@54a9388...`；
- `rmanluo/RoG-webqsp@c063253...`；
- `rmanluo/RoG-cwq@b0f6275...`。

WebQSP 的 retrieval source（1639 条）与最终 reasoning cohort（RoG，1628 条）不同。脚本以 ID 而不是行号连接，并在 retrieval 表中同时输出完整 retrieval cohort 和固定 QA cohort；最终 QA 只在 1628 条 cohort 上比较。它会分别校验 ID、question、topic/answer entity、answer-in-graph 与 graph hash；严格 candidate membership/entity 检查只对真正产生 score 的 `ml1996/webqsp` 启用。`rmanluo/RoG-webqsp` 只作为 pinned reasoning graph/metadata 来源。

```bash
export HF_HOME=/data/hf-cache/subgraphrag-phase02
export EXP_ROOT=/data/experiments/subgraphrag-query-factors
cd /path/to/SubgraphRAG
```

5. 把服务器 config 中 `llm.resolved_revision` 改成实际 40 位模型 snapshot commit，并在运行前把该 revision 完整缓存；也可以把 `llm.model_name` 指向 `snapshots/<commit>` 目录。preflight 只做 `local_files_only` 探测，不会在正式实验中偷偷换 revision；本地 snapshot 路径与声明 revision 不一致会失败。wrapper 再把已解析目录注入原 `llm_init`。manifest 记录 snapshot、metadata hash 和权重 shard 清单。原本地 vLLM 分支没有消费 CLI seed，因此用两次完整 baseline 实测稳定性，默认要求 raw-output agreement ≥99%、parsed-answer agreement =100%。

6. 在 retriever 环境对同一 checkpoint 至少独立复跑一次。`inference.py` 会覆盖 checkpoint 目录中的 `retrieval_result.pth`，所以每次结束立即复制到不同路径：

```bash
conda activate retriever
cd /path/to/SubgraphRAG/retrieve
python inference.py -p /data/checkpoints/webqsp/cpt.pth
cp /data/checkpoints/webqsp/retrieval_result.pth /data/reruns/webqsp/retrieval_result_1.pth
python inference.py -p /data/checkpoints/webqsp/cpt.pth
cp /data/checkpoints/webqsp/retrieval_result.pth /data/reruns/webqsp/retrieval_result_2.pth
```

切回 reasoning 环境。Phase 0 默认是严格门禁；设置产生原 baseline 的 checkpoint 和独立复跑结果（原 PTH + 复跑 PTH 构成重复运行比较）：

```bash
export RETRIEVER_CHECKPOINT=/data/checkpoints/webqsp/cpt.pth
export RETRIEVAL_REPLICATE_1=/data/reruns/webqsp/retrieval_result_1.pth
# 可选但推荐：export RETRIEVAL_REPLICATE_2=/data/reruns/webqsp/retrieval_result_2.pth
```

checkpoint 内嵌 config 会与声明的 encoder/topic PE/DDE rounds/seed 比对并保存 SHA。只有 dataset audit、checkpoint/budget、paper-metric sanity、retrieval 复跑、两次 baseline LLM 与 QA sanity 全部通过，`phase0_status` 才是 `confirmed`。没有 checkpoint 时可显式设置 `ALLOW_UNCONFIRMED_BASELINE=1` 生成诊断产物；最终报告会保持 blocked，不能解释为有效 headroom。

## 一键运行

```bash
# 第三个参数是服务器专用 config；也可用 QFORACLE_CONFIG 环境变量。
# checkpoint/replicate 环境变量按“当前这一次调用”读取，切换数据集时必须一起切换。
export RETRIEVER_CHECKPOINT=/data/checkpoints/webqsp/cpt.pth
export RETRIEVAL_REPLICATE_1=/data/reruns/webqsp/retrieval_result_1.pth
unset RETRIEVAL_REPLICATE_2  # 若有第二份 WebQSP 复跑结果，则改为对应路径
bash experiments/query_factor_oracle/run_remote.sh webqsp "$EXP_ROOT" /data/configs/webqsp.phase02.json

export RETRIEVER_CHECKPOINT=/data/checkpoints/cwq/cpt.pth
export RETRIEVAL_REPLICATE_1=/data/reruns/cwq/retrieval_result_1.pth
unset RETRIEVAL_REPLICATE_2  # 不要沿用 WebQSP 的路径
bash experiments/query_factor_oracle/run_remote.sh cwq "$EXP_ROOT" /data/configs/cwq.phase02.json
```

脚本先做不加载 LLM 的 path/revision/preflight，再执行：初始冻结 → 两次 baseline LLM/evaluator → 严格 Phase-0 confirmation → Phase 1 → 六种 Phase-2 retrieval → 五个新增 LLM 变体 → evaluator → final report。Phase 0 后所有阶段只读冻结副本 `phase0/retrieval/baseline_retrieval.pth`。checkpoint 会核验 ID prefix、dataset/model snapshot、core/prompt/protocol hash；完整结果重启时不会再次初始化 vLLM。若输出目录已有 Phase-0 lock，脚本先在独立的 `phase0_restart_audit/` 中对当前输入做 lock 验证，不会先覆盖原 lock；严格确认失败只写失败诊断（常规门禁写入 `last_confirmation_attempt.json`），保留原有 canonical lock/validation。reasoning resume 同样先校验 prediction fingerprint，再改写 prompt/manifest sidecar。

显存不足时只修改 config 中 `tensor_parallel_size`，并在 baseline lock 前固定；不要在不同变体之间改。首次 audit/cache 完成后，可在同一个 cache 下设置：

```bash
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1
```

## 分阶段命令

### Phase 0：冻结 baseline

```bash
python -m experiments.query_factor_oracle.phase0_freeze \
  --config experiments/query_factor_oracle/configs/webqsp.json \
  --output-dir "$EXP_ROOT/webqsp/phase0" \
  --retriever-checkpoint "$RETRIEVER_CHECKPOINT" \
  --retrieval-replicate "rerun_1=$RETRIEVAL_REPLICATE_1" \
  --audit-datasets
```

这是“先冻结、再跑两次 LLM”的初始快照，此时 `phase0_status` 必然还不是 confirmed。checkpoint、config 与 retrieval 必须从第一次冻结起就保持不变；最省心的做法仍是使用上面的一键脚本。

Phase 0 保存所有样本的 full-precision score、完整 stored pool、prompt-effective Top-100、topic/answer entity、target triples、输入/core-file hashes 和环境信息。`validation.json` 中每个 retrieval metric 都同时保存 numerator、denominator 和 eligible sample 数；`repo_raw_topk_metrics` 复刻 `retrieve/eval.py` 的原始前 K 口径用于对论文，`topk_metrics` 则复刻 reasoning 先去重再截断的真实 prompt 口径用于 Phase 2。

配置中还锁定了论文同设置的 sanity target（容差也写入 config）：Top-100 retrieval 的 WebQSP/CWQ `(shortest-path, GPT-labelled, answer)` 分别为 `(0.883, 0.865, 0.944)` 与 `(0.811, 0.840, 0.914)`；Llama3.1-8B full-test `(Macro-F1, Hit)` 分别为 WebQSP `(70.57, 86.61)`、CWQ `(47.16, 56.98)`。脚本输出 actual、expected、difference 和 within-tolerance。这些数值只用于发现与论文设置的漂移；严格模式要求它们通过，但报告里的 baseline 仍始终来自本次远程实测。

按前面的步骤重复执行原 `retrieve/inference.py`（其实现固定保存至多 500 条），保留两个 PTH，再加入：

```bash
--retrieval-replicate rerun_1=/absolute/path/retrieval_rerun_1.pth \
--retrieval-replicate rerun_2=/absolute/path/retrieval_rerun_2.pth
```

判定标准是：样本/三元组身份与顺序完全一致、metadata 一致、score 最大绝对误差不超过 `1e-6`。没有 checkpoint 或独立 rerun 时会明确标记 `not_measured`；除非显式使用诊断模式，否则不能进入 Phase 1–2。

reasoning 的两次运行示例：

```bash
python -m experiments.query_factor_oracle.run_reasoning \
  --config experiments/query_factor_oracle/configs/webqsp.json \
  --retrieval "$EXP_ROOT/webqsp/phase0/retrieval/baseline_retrieval.pth" \
  --variant baseline --replicate 0 \
  --output-dir "$EXP_ROOT/webqsp/phase0/reasoning/replicate_0"
```

每次 reasoning 后先用 `evaluate_reasoning` 生成 `evaluation_phase0/qa_summary.json`。最后用原锁做严格确认：

```bash
python -m experiments.query_factor_oracle.phase0_freeze \
  --config experiments/query_factor_oracle/configs/webqsp.json \
  --output-dir "$EXP_ROOT/webqsp/phase0" \
  --retriever-checkpoint "$RETRIEVER_CHECKPOINT" \
  --retrieval-replicate "rerun_1=$RETRIEVAL_REPLICATE_1" \
  --prediction "replicate_0=$EXP_ROOT/webqsp/phase0/reasoning/replicate_0/predictions.jsonl" \
  --prediction "replicate_1=$EXP_ROOT/webqsp/phase0/reasoning/replicate_1/predictions.jsonl" \
  --qa-summary "replicate_0=$EXP_ROOT/webqsp/phase0/reasoning/replicate_0/evaluation_phase0/qa_summary.json" \
  --qa-summary "replicate_1=$EXP_ROOT/webqsp/phase0/reasoning/replicate_1/evaluation_phase0/qa_summary.json" \
  --audit-datasets --require-confirmed-baseline \
  --verify-lock "$EXP_ROOT/webqsp/phase0/baseline.lock.json"
```

只有 `validation.json.phase0_status == "confirmed"` 才能解释后续 oracle 数字。严格确认失败只更新 `last_confirmation_attempt.json`，不会破坏原 frozen PTH 和 lock。

### Phase 1：logical form → query factors

```bash
python -m experiments.query_factor_oracle.phase1_build_factors \
  --config experiments/query_factor_oracle/configs/webqsp.json \
  --baseline "$EXP_ROOT/webqsp/phase0/retrieval/baseline_retrieval.pth" \
  --output-dir "$EXP_ROOT/webqsp/phase1"
```

解析器支持 official WebQSP/CWQ JSON、ChatKBQA 风格 JSON/JSONL、gold SPARQL 和 S-expression。它输出：

- `query_family` 与 operator tags（single/composition/conjunction/count/comparative/superlative 等）；
- relation set；
- 每个 relation slot 的 KG subject/object dependency、answer/topic direction、anchor-rooted hop depth、branch ID；
- answer/topic/中间变量的共享 dependency，以及只从 LF 显式类型约束得到的 answer type；
- raw LF、同格式 canonical form、跨 SPARQL/S-expression 的 semantic-factor signature、全部 parse error 和其他 gold alternatives。

等价处理包括：AND/OR flatten 后分支排序、双重 inverse 消去、Freebase URI/`ns:`/slash relation 归一、SPARQL triple 顺序无关、变量重命名无关。branch 是 answer/topic/常量/度数不为 2 的 junction 之间的最大路径；不等长、多 anchor 查询的深度从各 anchor 在 LF 图上计算，等距时方向保留为 `unknown`，不会强猜。多个 WebQSP parse 全部保留；主结果使用不依赖候选 score/答案的确定性 primary parse（execute-right 优先，其次 S-expression、长度和字典序），不会为了提高 oracle 指标挑 parse。

`canonical_signature` 用于同一种 LF 语法内的等价化；`semantic_factor_signature` 用规范化 dependency graph 检查跨格式的一致性。UNION、OPTIONAL、EXISTS、MINUS、property path 或未知 operator 等不能可靠表达完整语义的样本仍写入产物并统计，但 `oracle_eligible=false`，六个 reranker 都对它们保持 baseline 顺序。

`summary.json` 给出 ID/question 对齐、parse rate、family/operator、slot/branch/hop、direction、answer type、relation frequency、重复 relation slots、gold relation 在 frozen pool 中的覆盖及 OOV 关系。

### Phase 2：Gold Prior / Oracle Headroom

```bash
python -m experiments.query_factor_oracle.phase2_rerank \
  --config experiments/query_factor_oracle/configs/webqsp.json \
  --baseline "$EXP_ROOT/webqsp/phase0/retrieval/baseline_retrieval.pth" \
  --factors "$EXP_ROOT/webqsp/phase1/query_factors.jsonl.gz" \
  --output-dir "$EXP_ROOT/webqsp/phase2"
```

变体固定为：

| Variant | 使用的 gold 信息 |
|---|---|
| `baseline` | 原 question-only retrieval，顺序完全不动 |
| `family_structure` | 匿名 anchored query skeleton / family；relation 与方向 masked |
| `relation_set` | unordered gold relation identity |
| `relation_slot_branch` | relation + slot/branch + shared-variable dependency；KG 方向 masked（SR） |
| `direction` | 在 SR 上加入真实 KG subject/object 方向（SRD） |
| `all_factors` | SRD + 可靠的显式 answer type/operator（Joint） |

排序是 deterministic lexicographic priority，以原 baseline rank 作最终 tie-break；没有在 test 上搜索 lambda。结构臂只在 frozen candidate triples 上做 anchored partial query-graph homomorphism：同一 dependency 必须绑定同一 KG 节点，一个候选边最多满足一个重复 slot，SR 允许边两端交换，SRD 固定 KG 方向。它只是对已有候选池的匹配与重排，不会扩图或注入 gold 边。

每个 K 输出逐样本与聚合：shortest-path triple recall、原仓库 GPT-labelled triple recall、answer recall/hit、gold relation coverage、relation-slot coverage、带 direction 的 slot coverage、共享变量一致的 slot/branch coverage 与 full-structure hit，并按 family/parse status 分组。K=500（或实际 pool 大小）给出当前 stored-pool ceiling。回溯上限由 `phase2.match_state_limit` 固定；逐样本 diagnostics 与汇总都记录 truncation。只要任一结构匹配被截断，单数据集和 combined 报告都会阻断结构结论，先增大上限重跑。

### 固定 LLM 与 evaluator

每个变体使用自己的输出目录：

```bash
python -m experiments.query_factor_oracle.run_reasoning \
  --config experiments/query_factor_oracle/configs/webqsp.json \
  --retrieval "$EXP_ROOT/webqsp/phase2/all_factors/retrieval_result.pth" \
  --variant all_factors --replicate 0 \
  --output-dir "$EXP_ROOT/webqsp/phase2/all_factors/reasoning/replicate_0"

python -m experiments.query_factor_oracle.evaluate_reasoning \
  --config experiments/query_factor_oracle/configs/webqsp.json \
  --predictions "$EXP_ROOT/webqsp/phase2/all_factors/reasoning/replicate_0/predictions.jsonl" \
  --retrieval "$EXP_ROOT/webqsp/phase2/all_factors/retrieval_result.pth" \
  --factors "$EXP_ROOT/webqsp/phase1/query_factors.jsonl.gz" \
  --output-dir "$EXP_ROOT/webqsp/phase2/all_factors/reasoning/replicate_0/evaluation"
```

Reasoning 直接复用原 `get_data`、`get_prompts_for_data`、`llm_init` 和 `llm_inf_all`。Evaluator 直接复用原 corrected/original answer matcher，但显式传入本变体 retrieval；输出 full test、answer-in-graph subset、family、parse status、hop 分层的 Hit、Hit@1、Macro/Micro F1。

### 最终报告与判据

`build_report.py` 不会填造缺失 LLM 数字：若 QA 尚未在远程完成，报告会明确显示 pending。完整报告比较：

- `relation_set - baseline`：relation grounding 的价值；
- `relation_slot_branch - relation_set`：slot/branch 在 relation identity 之上的增量；
- `direction - relation_slot_branch`：direction 的净增量；
- `all_factors - direction`：可靠 type/operator 的净增量；
- retrieval delta 与 final Hit/F1 delta；
- 各 query family 的 paired gain 和样本数。

默认决策阈值写入 `final_report.json`，可审计而不是事后解释：joint answer-recall 增益小于 0.5pp 时暂停 predictor；slot/branch 增量至少 0.5pp 时支持结构表示；relation 占 joint gain 至少 80% 时优先 relation grounding；retrieval 至少提升 1pp 但 Macro-F1 小于 0.5 分时，优先研究 retrieval/reasoning 共享 query state。最终结论必须同时查看 WebQSP、CWQ、完整 cohort 与 parseable subset。

两个数据集都完成后生成唯一的跨数据集结论（命令会验证 dataset/split、Phase-0 状态、cohort 与输入 hash，路径写反、重复输入、缺 QA 或 matcher truncation 都会 blocked）：

```bash
python -m experiments.query_factor_oracle.build_combined_report \
  --report "webqsp=$EXP_ROOT/webqsp/report/final_report.json" \
  --report "cwq=$EXP_ROOT/cwq/report/final_report.json" \
  --output-dir "$EXP_ROOT/combined_report"
```

## 本地验证

无需下载数据或模型：

```bash
python -m unittest discover -s experiments/query_factor_oracle/tests -v
```

测试覆盖 canonical/跨格式语义等价、inverse、不等长多 anchor 深度、中间 junction 的正反例、数据源合并/错配、不可解析保留、候选集合不变、原 score 保留、无 answer/target-label 泄漏、重复 slot、prompt 去重、Phase-0 失败不覆盖、reasoning drift resume 不覆盖、report provenance，以及 Phase 0→2 的 CPU 端到端 smoke test。
