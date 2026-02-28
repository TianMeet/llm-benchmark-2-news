# 系统架构（当前实现审阅版）

本文档基于当前代码实现（`eval/` + `llm_core/`）整理，不是目标态设计图。

## 1. 架构总览图

```mermaid
flowchart TB
    %% ─────────── 样式定义 ───────────
    classDef input     fill:#EFF6FF,stroke:#3B82F6,stroke-width:2px,color:#1E3A5F,font-weight:bold
    classDef orch      fill:#F0FDF4,stroke:#22C55E,stroke-width:2px,color:#14532D,font-weight:bold
    classDef model     fill:#FFF7ED,stroke:#F97316,stroke-width:2px,color:#7C2D12,font-weight:bold
    classDef store     fill:#F5F3FF,stroke:#8B5CF6,stroke-width:2px,color:#3B0764,font-weight:bold
    classDef output    fill:#FFF1F2,stroke:#F43F5E,stroke-width:2px,color:#881337,font-weight:bold
    classDef task      fill:#ECFEFF,stroke:#06B6D4,stroke-width:2px,color:#164E63,font-weight:bold
    classDef api       fill:#FEFCE8,stroke:#EAB308,stroke-width:2px,color:#713F12,font-weight:bold

    %% ─────────── 输入层 ───────────
    subgraph INPUT["  📥 INPUT — 输入层  "]
        direction LR
        CLI["🖥️ CLI\npython -m eval.cli.runner"]
        DS["📄 Dataset\n*.jsonl"]
        WF_YAML["📋 Workflow Spec\nnews_pipeline.yaml"]
        REG_YAML["⚙️ Model Registry\nllm_providers.yaml"]
    end

    %% ─────────── 编排层 ───────────
    subgraph ORCH["  🎛️ ORCHESTRATOR — 编排层 (eval/)  "]
        direction TB
        RUNNER["🏃 Runner\nrunner.py\n单任务 / workflow 入口"]
        WFL["🔀 WorkflowLoader\nworkflow.py\nstep 依赖 & input_from"]
        REG["📦 ModelRegistry\nregistry.py\n模型注册 + 成本估算"]
        GATE["🚪 LLMGateway\ngateway.py\n统一调用门面"]
        CACHE["💾 EvalCache\ncache.py\nmodel+params+messages 键"]
        MET["📊 Metrics\nmetrics.py\naggregate_records()"]
        REP["📝 Reporter\nreporter.py\nMarkdownReporter"]

        subgraph TASKS["  任务插件层 (eval/tasks/)  "]
            direction LR
            T_BASE["🔧 EvalTask\nbase.py\n抽象基类"]
            T_IE["📌 IEJsonTask\nie_json.py"]
            T_STOCK["📈 StockScoreTask\nstock_score.py"]
            T_DEDUP["🗞️ NewsDedupTask\nnews_dedup.py"]
            T_GEN["⚡ GenericTask\ngeneric.py"]
        end
    end

    %% ─────────── 模型层 ───────────
    subgraph MODEL["  🤖 MODEL LAYER — 模型层 (llm_core/)  "]
        direction TB
        CLIENT_F["🏭 LLM Client Factory\nbase_client.py"]
        OAI["🔌 OpenAICompatibleClient\nopenai_client.py"]
        PROMPT["✏️ PromptRenderer\nprompt_renderer.py\nJinja2 渲染"]
        PARSER["🔍 ResponseParser\nresponse_parser.py"]
        BATCH["⚙️ BatchHelper\nbatch.py"]
    end

    %% ─────────── 存储层 ───────────
    subgraph STORE_L["  🗄️ STORE — 存储层 (eval/)  "]
        direction LR
        RSTORE["📁 RunStore\nstore.py\n产物读写抽象"]
        DATA_DIR["📂 runs/{timestamp}/"]
    end

    %% ─────────── 外部 API ───────────
    subgraph APIS["  🌐 外部 LLM API  "]
        direction LR
        DEEPSEEK["DeepSeek\nAPI"]
        KIMI["Kimi\n月之暗面"]
        GPT["OpenAI\ngpt-4o etc."]
        OTHERS["其他\nOpenAI-Compatible"]
    end

    %% ─────────── 输出产物 ───────────
    subgraph OUTPUT["  📤 OUTPUT — 输出产物  "]
        direction LR
        O_CFG["📄 config.json\n运行配置快照"]
        O_RES["📊 results.jsonl\n逐条结果 v1"]
        O_SUM["📋 summary.csv\n聚合统计 v1"]
        O_RPT["📝 report.md\n可读报告"]
    end

    %% ─────────── 配置文件 ───────────
    subgraph CONFIGS["  📁 CONFIGS  "]
        direction LR
        C_PROV["llm_providers.yaml\n/json"]
        C_PROMPT["prompts/\n*.yaml"]
        C_ENV[".env\nAPI Keys"]
    end

    %% ─────────── 数据集 ───────────
    subgraph DATASETS["  📂 DATASETS  "]
        direction LR
        D_BENCH["benchmark_news\n.jsonl"]
        D_DEMO["demo_news\n.jsonl"]
        D_EVAL["news_summary_eval\n.jsonl"]
    end

    %% ─────────── 连接关系 ───────────
    CLI --> RUNNER
    DS --> RUNNER
    WF_YAML --> WFL --> RUNNER
    REG_YAML --> REG

    RUNNER --> TASKS
    RUNNER --> GATE
    RUNNER --> MET
    RUNNER --> REP
    RUNNER --> RSTORE

    T_BASE --> T_IE & T_STOCK & T_DEDUP & T_GEN

    REG --> GATE
    GATE <-->|"cache hit / miss"| CACHE
    GATE --> CLIENT_F

    CLIENT_F --> OAI
    OAI --> PROMPT
    OAI --> PARSER
    OAI --> BATCH

    OAI -->|"REST / SSE"| DEEPSEEK & KIMI & GPT & OTHERS

    MET --> RSTORE
    REP --> RSTORE
    RSTORE --> DATA_DIR

    DATA_DIR --> O_CFG & O_RES & O_SUM & O_RPT

    CONFIGS -.->|"加载"| REG & RUNNER & OAI
    DATASETS -.->|"读取"| RUNNER

    %% ─────────── 样式应用 ───────────
    class CLI,DS,WF_YAML,REG_YAML input
    class RUNNER,WFL,REG,GATE,CACHE,MET,REP orch
    class T_BASE,T_IE,T_STOCK,T_DEDUP,T_GEN task
    class CLIENT_F,OAI,PROMPT,PARSER,BATCH model
    class RSTORE,DATA_DIR store
    class DEEPSEEK,KIMI,GPT,OTHERS api
    class O_CFG,O_RES,O_SUM,O_RPT output
```

## 2. 执行时序（workflow 模式）

```mermaid
sequenceDiagram
  participant U as User/CLI
  participant R as Runner
  participant W as WorkflowSpec
  participant T as EvalTask step
  participant G as LLMGateway
  participant C as EvalCache
  participant M as ModelRegistry and llm_core
  participant S as RunStore
  participant A as API

  U->>R: python -m eval.cli.runner --workflow ...
  R->>W: load_workflow()
  R->>S: write_config(config.json)

  loop each sample
    loop each workflow step
      R->>T: build_prompt(sample, context)
      R->>G: call(model_id, task, sample_cache_id, messages)
      G->>C: get(cache_key)
      alt cache hit
        C-->>G: cached UnifiedCallResult
      else cache miss
        G->>M: create_client(model_id, params)
        M->>A: chat.completions.create(...)
        A-->>M: response
        M-->>G: LLMResponse
        G->>C: set(cache_key, result)
      end
      G-->>R: UnifiedCallResult
      R->>T: parse()/metrics()
      R->>S: append_result(step row)
    end
    R->>S: append_result(workflow_e2e row)
  end

  R->>R: aggregate_records()
  R->>S: write_summary(summary.csv)
  R->>S: generate_report(report.md)
```

## 3. 当前实现要点（审阅结论）

1. 架构分层清晰：`runner` 负责编排，`gateway` 负责调用与缓存，`registry` 负责模型配置解析，`store/report` 负责产物落盘与展示。
2. 数据闭环完整：每次运行都会产出 `config/results/summary/report`，满足最小可复现要求。
3. workflow 依赖关系通过 `input_from` + 上游 `parse_success` 控制，失败会写入 `skipped` 记录并继续执行后续样本。
4. 当前并发能力：
   - task 模式支持按模型并发（`--concurrency`，模型级 semaphore）。
   - workflow 模式支持样本级并发（`--workflow-concurrency`），单样本内 step 仍保持顺序依赖。
5. 缓存命中粒度合理：键由 `model + params + messages + sample_cache_id` 组成，能覆盖 task/workflow 的重复调用复用。
6. `LLMGateway` 已对相同 `model_id + params_override` 复用 client，降低重复建连开销。
7. 运行产物已带版本契约：`results.schema_version=result_row.v1`、`summary.schema_version=summary_row.v1`，`config.json` 含 `scorer_version`。

## 4. 建议的下一步演进

1. 将 workflow 并发从“样本级”扩展到“DAG 级”（可并行的 step 分支调度）。
2. 为 `results.jsonl`/`summary.csv` 提供正式 JSON Schema 文件，并在 CI 中做契约校验。
3. 增加跨 run 基线对比报告（按 `task + model + prompt_version + scorer_version`）。
