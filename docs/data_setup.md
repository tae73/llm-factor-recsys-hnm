# 데이터 셋업 가이드 (Data Setup)

이 프로젝트는 H&M Personalized Fashion Recommendations 데이터셋과 그로부터 생성되는 파생
산출물을 모두 `data/` 아래에 둔다. `data/`는 **git-ignored**(`.gitignore`의 `data/`, 예외로
`data/.gitkeep`만 추적)이므로 저장소에는 데이터가 들어 있지 않다. 이 문서는 **어떤 데이터가 어떤
디렉토리에 어떤 형태로 필요하고, 어떻게 다시 받는지**를 설명한다.

데이터는 두 부류다.

- **RAW** — Kaggle 원본(CSV + 이미지). 약 33GB. 재취득은 Kaggle 재다운로드 또는 외장 보관본 복사.
- **파생(Derived)** — 파이프라인(`scripts/`)이 RAW로부터 생성. 재실행으로 재생성 가능. 대부분 원격
  A100 서버에 백업되어 있다(`segmentation` 제외).

> **현재 상태(2026-06)**: 로컬 디스크 절약을 위해 **RAW는 외장 드라이브
> `/Volumes/samsungj2/data/`로 이동**했고, **파생 데이터는 로컬에서 삭제**했다. 아래 "되받기"
> 절차로 언제든 복원할 수 있다.

---

## 디렉토리 레이아웃 · 형태 · 출처

모든 경로는 CLI(`--raw-dir`, `--data-dir` 등) 또는 `configs/data/hm.yaml`로 변경 가능하다.
아래는 기본(default) 경로 기준이다.

### RAW

| 디렉토리 / 파일 | 형태 | 용량(약) | 출처 |
|---|---|---|---|
| `data/h-and-m-personalized-fashion-recommendations/articles.csv` | CSV (~105K 상품 메타) | 36M | Kaggle |
| `…/customers.csv` | CSV (~1.37M 고객) | 207M | Kaggle |
| `…/transactions_train.csv` | CSV (~31M 거래) | 3.5G | Kaggle |
| `…/sample_submission.csv` | CSV | 270M | Kaggle |
| `…/images/{3자리 prefix}/{article_id}.jpg` | JPG (~105K 흰배경 상품 사진) | ~29G | Kaggle |

> **합계 ~33GB.** `images/`는 `010`, `011`, … 식 3자리 prefix 하위 폴더로 분할되어 있다.

### 파생 (Derived)

| 디렉토리 | 주요 파일 / 형태 | 용량(약) | 생성 스크립트 | 서버 백업 |
|---|---|---|---|---|
| `data/processed/` | `*.parquet`(articles/customers/transactions, train·val·test split) + `*.json`(active/sparse customer ids, ground truth) | 1.8G | `scripts/preprocess.py` | ✓ |
| `data/knowledge/factual/` | `factual_knowledge.parquet`(LLM L1+L2+L3) + `quality_report.json` | 14M | `scripts/extract_factual_knowledge.py` (**LLM ~$10**) | ✓ |
| `data/knowledge/reasoning/` | `user_profiles.parquet`, `reasoning_texts.parquet` + `quality_report.json` | 1.6G | `scripts/extract_reasoning_knowledge.py` | ✓ |
| `data/features/` | `*.npz`(train_pairs/user_features/item_features) + `*.json`(id_maps, cat_vocab, feature_meta) | 930M | `scripts/build_features.py` | ✓ |
| `data/embeddings/` | `item_bge_embeddings.npz`, `user_bge_embeddings.npz` + `ablation/{l1,l2,l3,l1_l2,l1_l3,l2_l3,l1_l2_l3}.npz` | 2.8G | `scripts/segment.py` | ✓ |
| `data/segmentation/` | `customer_l{1,2,3}_vectors.npz`, `customer_segments.parquet`, `product_clusters.parquet`, `segment_*` 등 | 2.3G | `scripts/segment.py` | **✗ (백업 없음)** |

> `data/prestore/`(Augmented Vector, `scripts/prestore.py` 산출)는 학습된 KAR 모델이 있을 때
> 생성되며, 위 표에는 없을 수 있다. 모델 학습 후 `scripts/prestore.py`로 재생성한다.

---

## 되받기 (Restore)

### RAW 복원

1. **외장 보관본 복사** (가장 빠름):
   ```bash
   rsync -a /Volumes/samsungj2/data/h-and-m-personalized-fashion-recommendations/ \
            data/h-and-m-personalized-fashion-recommendations/
   ```
2. **외장에 둔 채 사용** (복사 없이 경로만 지정):
   ```bash
   python scripts/preprocess.py \
     --raw-dir /Volumes/samsungj2/data/h-and-m-personalized-fashion-recommendations \
     --output-dir data/processed
   # 또는 configs/data/hm.yaml 의 raw_dir 를 외장 경로로 수정
   ```
3. **처음부터 (Kaggle)**:
   ```bash
   kaggle competitions download -c h-and-m-personalized-fashion-recommendations
   unzip h-and-m-personalized-fashion-recommendations.zip \
     -d data/h-and-m-personalized-fashion-recommendations
   ```

> macOS 기본 `rsync`는 `openrsync`(2.6.9 호환)라 `--info=progress2` 같은 rsync 3.x 옵션을
> 지원하지 않는다. 위처럼 `-a`만 쓰거나 `-av`/`-P`를 사용한다.

### 파생 데이터 복원

**옵션 A — 원격 서버에서 pull (대부분 백업되어 있어 빠름)**

`scripts/sync.sh`에는 파생 데이터를 서버에서 되받는 명령이 없으므로(현재 `push-data`/`pull`만 존재)
수동 rsync를 사용한다. 서버 접속 정보는 메모리/`scripts/sync.sh` 참고
(`ssh -p 5040 mail-agent@3.38.195.121`, 프로젝트 `~/project/llm-factor-recsys-hnm`):

```bash
rsync -avz -e 'ssh -p 5040' \
  --exclude 'h-and-m-personalized-fashion-recommendations' \
  mail-agent@3.38.195.121:'project/llm-factor-recsys-hnm/data/' \
  ./data/
```

> 서버에는 `processed`/`knowledge`/`features`/`embeddings`가 있고, **`segmentation`과 RAW
> 이미지는 없다**(sync 제외). `segmentation`은 옵션 B로만 복원 가능.

**옵션 B — 파이프라인으로 재생성** (`CLAUDE.md` Pipeline Usage 0→4 참고)

```bash
python scripts/preprocess.py            --raw-dir <raw> --output-dir data/processed
python scripts/extract_factual_knowledge.py   ...   # LLM ~$10
python scripts/extract_reasoning_knowledge.py  ...
python scripts/build_features.py        ...
python scripts/segment.py               ...          # embeddings + segmentation 생성
```

`data/segmentation/`은 서버 백업이 없으므로 `scripts/segment.py` 재실행으로만 복원된다.

---

## 디스크 사용 요약

| 부류 | 로컬 보관 위치 | 용량(약) |
|---|---|---|
| RAW | 외장 `/Volumes/samsungj2/data/` (로컬엔 없음) | ~33G |
| 파생 | 삭제됨 (서버 백업 또는 재생성) | ~10G |

RAW를 외장으로 옮기고 파생을 삭제하면 로컬에서 약 **42GB**를 확보한다.
