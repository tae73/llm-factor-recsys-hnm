#!/usr/bin/env bash
# sync.sh — rsync 기반 로컬 ↔ 원격 서버 동기화 (PyCharm SFTP auto-upload 대체)
#
# 접속 설정은 프로젝트 루트의 .sync.env 에서 읽음.
#
# 사용법:
#   ./scripts/sync.sh push            # 코드만 업로드 (가장 빠름, <1초)
#   ./scripts/sync.sh push-knowledge  # data/knowledge/ 업로드 (LLM 산출물 ~1.6G)
#   ./scripts/sync.sh push-data       # 파생 데이터 전체 업로드 (raw 32G 제외)
#   ./scripts/sync.sh pull            # 서버 results/ → 로컬 (모델·예측 회수)
#   ./scripts/sync.sh remote "<cmd>"  # 서버에서 명령 실행 (예: remote "whoami && pwd")
#
# 플래그:
#   -n, --dry-run   실제 전송 없이 미리보기 (--delete 영향 사전확인)
#   --delete        대상에서 원본에 없는 파일 삭제 (push 미러링; 기본 off)
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# --- config 로드 ---
ENV_FILE="$PROJECT_ROOT/.sync.env"
if [[ ! -f "$ENV_FILE" ]]; then
  echo "ERROR: $ENV_FILE 없음. .sync.env 를 먼저 생성하세요." >&2
  exit 1
fi
# shellcheck disable=SC1090
set -a; source "$ENV_FILE"; set +a

: "${REMOTE_USER:?REMOTE_USER 미설정 (.sync.env)}"
: "${REMOTE_HOST:?REMOTE_HOST 미설정 (.sync.env)}"
: "${REMOTE_PORT:?REMOTE_PORT 미설정 (.sync.env)}"
: "${REMOTE_DIR:?REMOTE_DIR 미설정 (.sync.env)}"

# --- ssh 명령 구성 (SSH_KEY 설정 시에만 -i 추가) ---
SSH_OPTS=(-p "$REMOTE_PORT")
if [[ -n "${SSH_KEY:-}" ]]; then
  SSH_OPTS+=(-i "$SSH_KEY")
fi
SSH_CMD="ssh ${SSH_OPTS[*]}"
REMOTE="$REMOTE_USER@$REMOTE_HOST"

# --- 플래그 파싱 (bash 3.2 호환: 빈 배열 전개 회피) ---
DRY=""
DELETE=""
ARGS=()
for a in "$@"; do
  case "$a" in
    -n|--dry-run) DRY="-n" ;;
    --delete)     DELETE="--delete" ;;
    *)            ARGS+=("$a") ;;
  esac
done
if [[ ${#ARGS[@]} -gt 0 ]]; then set -- "${ARGS[@]}"; else set --; fi
CMD="${1:-}"
[[ $# -gt 0 ]] && shift || true

# rsync 기본 옵션 (DRY는 설정됐을 때만 추가 → 빈 인자 주입 방지)
RSYNC=(rsync -az --partial --progress -e "$SSH_CMD"
       --exclude-from="$PROJECT_ROOT/.rsync-exclude")
[[ -n "$DRY" ]] && RSYNC+=("$DRY")
# 서버 rsync 경로 지정 (userland 설치 시; .sync.env REMOTE_RSYNC)
[[ -n "${REMOTE_RSYNC:-}" ]] && RSYNC+=(--rsync-path="$REMOTE_RSYNC")

# 코드 push 대상 (data/ 미포함 → 빠름). 존재하는 항목만 추림(미생성 디렉터리 skip).
CODE_CANDIDATES=(src scripts configs tests mlops notebooks docs pyproject.toml README.md CLAUDE.md PLAN.md)
CODE_PATHS=()
for p in "${CODE_CANDIDATES[@]}"; do [[ -e "$PROJECT_ROOT/$p" ]] && CODE_PATHS+=("$p"); done

cd "$PROJECT_ROOT"

case "$CMD" in
  push)
    echo "▶ 코드 push → $REMOTE:$REMOTE_DIR"
    R=("${RSYNC[@]}")
    [[ -n "$DELETE" ]] && R+=("$DELETE")
    R+=("${CODE_PATHS[@]}" "$REMOTE:$REMOTE_DIR/")
    "${R[@]}"
    ;;
  push-knowledge)
    echo "▶ data/knowledge/ push → $REMOTE:$REMOTE_DIR  (LLM 산출물)"
    R=("${RSYNC[@]}")
    [[ -n "$DELETE" ]] && R+=("$DELETE")
    R+=(-R data/knowledge "$REMOTE:$REMOTE_DIR/")
    "${R[@]}"
    ;;
  push-data)
    echo "▶ 파생 데이터 push (raw 제외) → $REMOTE:$REMOTE_DIR"
    R=("${RSYNC[@]}")
    [[ -n "$DELETE" ]] && R+=("$DELETE")
    R+=(-R data/processed data/features data/embeddings data/segmentation data/knowledge \
        "$REMOTE:$REMOTE_DIR/")
    "${R[@]}"
    ;;
  pull)
    echo "▶ results/ pull ← $REMOTE:$REMOTE_DIR  (--delete 미사용, 로컬 보호)"
    mkdir -p results
    R=("${RSYNC[@]}")
    R+=("$REMOTE:$REMOTE_DIR/results/" results/)
    "${R[@]}"
    ;;
  remote)
    [[ $# -ge 1 ]] || { echo "usage: $0 remote \"<command>\"" >&2; exit 1; }
    echo "▶ remote @ $REMOTE: $*"
    # shellcheck disable=SC2086
    exec $SSH_CMD "$REMOTE" "$*"
    ;;
  ""|-h|--help|help)
    sed -n '2,21p' "$0" | sed 's/^# \{0,1\}//'
    ;;
  *)
    echo "ERROR: 알 수 없는 명령 '$CMD'  (push | push-knowledge | push-data | pull | remote)" >&2
    exit 1
    ;;
esac
