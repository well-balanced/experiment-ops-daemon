# experiment-ops-daemon

GPU 슬롯을 모니터링하면서 실험 스크립트를 자동으로 큐잉·실행하는 데몬.

논문 제출 직전, 주말에 서버 접속이 어려울 때, GPU가 노는 게 싫을 때.

---

## 설치

```bash
pip install pyyaml
# 카카오 알림 없이 쓴다면 이게 전부
```

## 빠른 시작

```bash
# 1. 실험 스크립트를 queue/ 에 넣는다
cp my_train.sh queue/

# 2. 데몬 시작
python expd.py start -d

# 3. 상태 확인
python expd.py status

# 4. 데몬 종료
python expd.py stop
```

---

## 기능

### GPU 자동 감지

`nvidia-smi`로 compute 프로세스가 없는 GPU만 감지. 메모리 기준이 아니라 프로세스 유무로 판단.

### 실험 큐

`queue/` 디렉토리에 `.sh` 파일을 드롭하면 자동 감지. 파일명 알파벳 순으로 실행.

```
queue/
  001_baseline.sh
  002_ablation.sh
```

### 실험 히스토리

실험마다 `runs/{name}/` 디렉토리를 생성. 데몬을 재시작해도 큐와 히스토리가 유지된다.

```
runs/
  20260509143022_baseline/
    run.sh          # 실행된 스크립트 사본 (# MEMO 등 주석 포함)
    status.json     # 실행 이력 배열 (재시도 포함)
    run.log         # stdout + stderr 합친 로그 (폴링마다 최근 100줄로 트림)
    analysis.md     # # ANALYZE 있을 때만 생성
```

`status.json`은 배열로, 같은 실험을 여러 번 시도하면 이력이 쌓인다.

```json
[
  {"status": "crashed", "gpu_id": 2, "start_time": "...", "end_time": "..."},
  {"status": "completed", "gpu_id": 7, "start_time": "...", "end_time": "..."}
]
```

### GPU 번호 참조

`expd`는 실행 전에 `CUDA_VISIBLE_DEVICES`를 자동으로 세팅한다. 스크립트 안에서 이 값을 그대로 쓰면 된다.

```bash
#!/bin/bash
# expd가 CUDA_VISIBLE_DEVICES=7 로 세팅한 뒤 실행
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} \
MUJOCO_EGL_DEVICE_ID=${CUDA_VISIBLE_DEVICES} \  # EGL 등 물리 GPU ID가 필요한 경우
python train.py ...
```

스크립트에 GPU 번호를 하드코딩하지 말 것. `expd`가 빈 슬롯을 골라서 넘겨준다.

### 메모 (MEMO)

스크립트 헤더에 실험 의도를 기록. `run.sh`에 그대로 저장되어 나중에 맥락 파악에 활용.

```bash
#!/bin/bash
# MEMO: lr 1e-4 → 1e-3 수렴 속도 비교
# MEMO: baseline 대비 critic layer norm 추가
```

### 종속성 (DEPENDS_ON / WAIT_FOR)

둘 중 하나만 써도 되고, 같이 써도 된다. 조건이 모두 만족될 때까지 대기.

**DEPENDS_ON** — 선행 run이 `completed` 상태일 때 실행.

```bash
# DEPENDS_ON: tw-base
# DEPENDS_ON: tw-base tw-e2e  ← 여러 개는 스페이스로
```

**WAIT_FOR** — 특정 파일/디렉토리가 존재할 때 실행. 체크포인트 대기에 유용.

```bash
# WAIT_FOR: /path/to/checkpoints/step_1000000
```

**같이 쓰기** — 둘 다 만족해야 실행.

```bash
# DEPENDS_ON: tw-base
# WAIT_FOR: /path/to/tw-base/checkpoints/step_1000000
```

### 자동 분석 (ANALYZE)

스크립트에 `# ANALYZE` 주석이 있으면 실험 완료 후 Claude API로 `analysis.md` 자동 생성.
`agent.md`에 프로젝트 컨텍스트를 작성해두면 분석 품질이 올라간다.

```bash
#!/bin/bash
# ANALYZE

python train.py ...
```

분석 기능을 쓰려면 `anthropic` 패키지 필요: `pip install anthropic`

### 카카오톡 알림

실험 시작 / 완료 / 실패·크래시 시 즉시 알림, 12시간마다 현황 리포트.

**설정:**

1. `kakao_auth.json` 작성:
```json
{
  "client_id": "REST_API_키",
  "access_token": "...",
  "refresh_token": "..."
}
```

2. 액세스 토큰 만료 시 리프레시 토큰으로 자동 갱신.

---

## CLI

```bash
python expd.py start          # 포그라운드 실행
python expd.py start -d       # 백그라운드 데몬
python expd.py stop           # 데몬 종료
python expd.py status         # 큐·실행 현황 출력
python expd.py add train.sh   # 스크립트를 queue/ 에 추가
```

---

## 설정 (config.yaml)

```yaml
poll_interval_seconds: 300    # GPU 체크 주기 (기본 5분)
max_concurrent_per_gpu: 1     # GPU당 최대 동시 실행 수
log_snapshot_lines: 100       # run.log 보존 줄 수
kakao_report_interval_hours: 12
```

---

## 디렉토리 구조

```
experiment-ops-daemon/
├── expd.py             # 메인 데몬 + CLI
├── config.yaml         # 설정
├── agent.md            # 프로젝트 컨텍스트 (ANALYZE 분석에 사용)
├── kakao_auth.json     # 카카오 인증 정보 (gitignore)
├── queue/              # 실험 스크립트 드롭존
└── runs/               # 실험 히스토리
```
