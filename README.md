# Battery Strategy Agent

## Subject

배터리 시장 전략 분석 Agent 개발

## Overview

이 프로젝트는 전기차 캐즘 환경에서 LG에너지솔루션과 CATL의 포트폴리오 다각화 전략을 비교 분석하는 멀티 에이전트 시스템의 초기 스캐폴딩입니다.

현재 저장소는 LangGraph/LangChain 기반 멀티에이전트 구조 위에 LLM supervisor routing, local-first retrieval expansion decision, retrieval sufficiency 분기, skeptic 재검증, 실제 한국어 HTML/PDF 보고서 생성까지 연결된 상태입니다.

## Features

- Supervisor 단일 패턴 기반 멀티 에이전트 구조
- `StateGraph`, `START`, `END`, `TypedDict` 기반 상태 설계
- Planner, Supervisor, Market, LGES, CATL, Skeptic, Compare/SWOT, Writer, Validator 노드 분리
- `local RAG -> LLM retrieval decision -> balanced web search / query refinement -> skeptic re-check` 정책
- `parallel_retrieval` 이후 retrieval sufficiency 기반 supervisor 동적 분기
- retryable / non-retryable validation issue와 안전 종료 사유 기록
- 실제 한국어 보고서 본문 및 클래식 스타일 HTML/PDF 파일 저장
- Prompt 파일 분리 및 역할별 계약 정리
- `MemorySaver` 기반 그래프 컴파일 지점 제공
- app startup 시 local RAG prewarm 및 noisy third-party log 축소 지원
- `.env.example`, `requirements.txt`, 기본 디렉터리 구조 포함

## Tech Stack

- Python 3.11+
- LangGraph
- LangChain
- Chroma
- Qwen/Qwen3-Embedding-0.6B
- python-dotenv

## Agents

- Planner Node: 사용자 목표를 보고 섹션 계획을 정리
- Supervisor Agent: 현재 state를 보고 다음 broad step queue를 LLM으로 결정
- Market Agent: EV 캐즘 및 산업 배경 조사
- LGES Agent: LG에너지솔루션 전략 조사
- CATL Agent: CATL 전략 조사
- Skeptic Agent: 반대 근거와 리스크 재검증
- Compare / SWOT Agent: 비교 프레임 정렬 및 5장 생성
- Writer Agent: 한국어 보고서 본문 생성
- Validator Agent: 초안 검증 및 수정 루프 제어

## Architecture

기본 흐름은 아래와 같습니다.

`Plan -> Retrieve -> (LLM Supervisor Branch) -> Skeptic? -> Compare -> Write -> Validate -> Finalize`

retrieval 정책은 아래 순서를 따릅니다.

`1차 local RAG -> 2차 LLM retrieval decision -> 3차 balanced web search / refinement -> 4차 skeptic re-check`

현재는 local RAG가 먼저 실행되고, web search나 refinement를 더 할지는 LLM decider가 판단합니다. retrieval sufficiency / skeptic 분기 / 실제 보고서 생성 / safe termination 로직은 상태 전이로 연결되어 있습니다.

## LLM Policy

- `LLM_MODEL`: planner 등 일반 LLM 단계의 기본 모델
- `REPORT_LLM_MODEL`: `Compare / SWOT` 단계의 기본 보고서 모델
- `WRITER_LLM_MODEL`: `Writer` 단계 전용 모델
- 기본값은 `LLM_MODEL=gpt-4o-mini`, `REPORT_LLM_MODEL=gpt-4o`, `WRITER_LLM_MODEL=gpt-4o`

## Prompt Policy

- 실제 workflow에서 사용하는 prompt는 `planner.md`, `supervisor.md`, `retrieval_decider.md`, `query_refiner.md`, `compare_swot.md`, `writer.md`입니다.
- `prompts/vision/common.md`, `prompts/vision/other.md`는 ingestion vision 분석에서 사용됩니다.
- 모든 prompt는 코드의 실제 동작과 맞아야 하며, placeholder나 TODO 문구를 남기지 않는 것을 원칙으로 합니다.

## Local Run Notes

- 기본 실행: `uv run app.py`
- 기본값으로 `LOCAL_RAG_PREWARM_ENABLED=true`가 적용되어 app 시작 시 Chroma collection과 embedding backend를 미리 데웁니다.
- 기본값으로 `QUIET_THIRD_PARTY_LOGS=true`가 적용되어 `httpx`, `sentence_transformers`, `huggingface_hub`의 과한 INFO 로그를 줄입니다.
- 초기 모델 다운로드가 잦거나 rate limit이 걸리면 `HF_TOKEN`을 설정하면 안정성과 속도에 도움이 됩니다.

## Graph Visualization

LangGraph compiled object는 Mermaid PNG 렌더링을 먼저 시도하고, 실패하면 ASCII 그래프로 fallback 하도록 바로 시각화할 수 있습니다.

```python
from graph.builder import build_graph
from graph.visualization import display_graph

supervisor = build_graph()
display_graph(supervisor)
```

위 헬퍼는 내부적으로 아래 방식으로 동작합니다.

```python
from IPython.display import Image, display

try:
    display(Image(supervisor.get_graph().draw_mermaid_png()))
except Exception:
    print(supervisor.get_graph().draw_ascii())
```

같은 헬퍼는 `custom_supervisor` 같은 다른 compiled graph 객체에도 그대로 사용할 수 있습니다.

```python
custom_supervisor = build_graph()
display_graph(custom_supervisor)
```

## Directory Structure

```text
.
├── app.py
├── README.md
├── requirements.txt
├── .env.example
├── data/
│   └── .gitkeep
├── outputs/
│   └── .gitkeep
├── prompts/
│   ├── planner.md
│   ├── supervisor.md
│   ├── retrieval_decider.md
│   ├── query_refiner.md
│   ├── compare_swot.md
│   ├── writer.md
│   └── vision/
├── agents/
│   ├── __init__.py
│   ├── base.py
│   ├── planner.py
│   ├── supervisor.py
│   ├── market.py
│   ├── lges.py
│   ├── catl.py
│   ├── skeptic.py
│   ├── compare_swot.py
│   ├── writer.py
│   └── validator.py
├── graph/
│   ├── __init__.py
│   ├── builder.py
│   └── router.py
├── retrieval/
│   ├── __init__.py
│   ├── local_rag.py
│   ├── balanced_web_search.py
│   └── query_policy.py
├── schemas/
│   ├── __init__.py
│   └── state.py
├── config/
│   ├── __init__.py
│   └── settings.py
└── utils/
    ├── __init__.py
    ├── logging.py
    └── prompt_loader.py
```

## TODO

- local corpus 확장 및 metadata 품질 개선
- retrieval reranking 및 decision context 품질 개선
- 본문 인라인 citation 전략 보강

## Contributors

- TODO: Contributor 1
- TODO: Contributor 2
