# Config Design: Hydra + Typer + DVC Integration

## Design Background

이 프로젝트는 체계적인 ablation 실험 (Layer×Gating×Fusion×Encoder×Backbone)을 수행한다.
Config 관리 시스템이 다음 요구사항을 충족해야 한다:

1. **Config Group 분리**: baseline, split, evaluation 등 독립적 Config Group
2. **CLI 호환**: Typer 기반 CLI에서 config override 가능
3. **DVC 연동**: `dvc exp run -S` 로 Grid Search 자동화
4. **확장성**: 후속 Phase에서 model/, gating/, fusion/ 등 Config Group 추가

## Architecture

```
[사용자 CLI]
  typer.Option(--override, -o)
         │
         ▼
[Hydra Compose API]
  initialize_config_dir() + compose()
         │
         ▼
[OmegaConf DictConfig]
  cfg.split.train_end, cfg.baseline.factors
         │
         ▼
[src/ 모듈]
  NamedTuple로 변환하여 사용
```

## Config Group Structure

```
configs/
├── config.yaml              # defaults list
├── data/
│   └── hm.yaml              # H&M dataset paths
├── split/
│   └── temporal.yaml         # Train/val/test boundaries
├── filter/
│   └── active5.yaml          # Customer filtering
├── baseline/
│   ├── als.yaml              # ALS hyperparameters
│   ├── bpr.yaml              # BPR hyperparameters
│   └── popularity.yaml       # Popularity window
├── evaluation/
│   └── default.yaml          # k=12, metrics list
└── (future phases)
    ├── extract/              # L1/L2/L3 extraction settings
    ├── model/                # DeepFM, SASRec, etc.
    ├── gating/               # G1-G4 variants
    ├── fusion/               # F1-F4 variants
    └── encoder/              # BGE, E5, TF-IDF
```

## Usage

### Direct Script Execution

```bash
# Default config
python scripts/train.py --data-dir data/processed --model-dir results/models --backbone userknn

# With Hydra overrides (when scripts support it)
python scripts/train.py \
    --config-dir configs \
    --override baseline=bpr \
    --override baseline.factors=256
```

### DVC Experiment Grid Search

```bash
# Switch config group
dvc exp run -S 'baseline=bpr'

# Override individual parameter
dvc exp run -S 'baseline.factors=256'

# Grid search (queue multiple experiments)
dvc exp run -S 'baseline.factors=range(64,256,64)' --queue
dvc exp run -S 'baseline=als,bpr' --queue

# Compare results
dvc exp show --sort-by results/metrics/eval_metrics.json:map_at_12
```

## Adding New Config Groups

후속 Phase에서 새 Config Group 추가 방법:

### Step 1: Config 파일 생성

```yaml
# configs/model/deepfm.yaml
embedding_dim: 64
hidden_dims: [256, 128]
dropout: 0.1
learning_rate: 0.001
batch_size: 4096
epochs: 50
```

### Step 2: defaults list 업데이트

```yaml
# configs/config.yaml
defaults:
  - data: hm
  - split: temporal
  - filter: active5
  - baseline: als
  - evaluation: default
  - model: deepfm       # ← 추가
```

### Step 3: Script에서 접근

```python
with initialize_config_dir(config_dir=str(config_dir.resolve()), version_base=None):
    cfg = compose(config_name="config", overrides=overrides)

# cfg.model.embedding_dim, cfg.model.hidden_dims 등으로 접근
```

## Design Principles

1. **Config Group별 파일 분리** — ablation에서 `baseline=bpr` 한 줄로 전체 config 전환
2. **Compose API** — `@hydra.main()` 없이 Typer와 완벽 호환
3. **DVC 네이티브 지원** — Hydra params를 DVC가 직접 추적
4. **NamedTuple 변환** — OmegaConf DictConfig → NamedTuple로 타입 안전성 확보
