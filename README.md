# Battery Strategy Agent

## Subject

배터리 시장 전략 분석 Agent 개발

## Overview

이 프로젝트는 전기차 캐즘 환경에서 LG에너지솔루션과 CATL의 포트폴리오 다각화 전략을 비교 분석하는 멀티 에이전트 시스템의 초기 스캐폴딩입니다.

현재 저장소는 LangGraph/LangChain 기반 멀티에이전트 구조 위에 retrieval sufficiency 분기, skeptic 재검증, 실제 한국어 HTML/PDF 보고서 생성까지 연결된 상태입니다.

## Features

- Supervisor 단일 패턴 기반 멀티 에이전트 구조
- `StateGraph`, `START`, `END`, `TypedDict` 기반 상태 설계
- Planner, Supervisor, Market, LGES, CATL, Skeptic, Compare/SWOT, Writer, Validator 노드 분리
- `local RAG -> balanced web search -> skeptic re-check` 정책
- `parallel_retrieval` 이후 retrieval sufficiency 기반 supervisor 동적 분기
- retryable / non-retryable validation issue와 안전 종료 사유 기록
- 실제 한국어 보고서 본문 및 클래식 스타일 HTML/PDF 파일 저장
- Prompt 파일 분리 및 역할별 계약 정리
- `MemorySaver` 기반 그래프 컴파일 지점 제공
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
- Supervisor Agent: 현재 phase 기준으로 다음 specialist를 라우팅
- Market Agent: EV 캐즘 및 산업 배경 조사
- LGES Agent: LG에너지솔루션 전략 조사
- CATL Agent: CATL 전략 조사
- Skeptic Agent: 반대 근거와 리스크 재검증
- Compare / SWOT Agent: 비교 프레임 정렬 및 5장 생성
- Writer Agent: 한국어 보고서 본문 생성
- Validator Agent: 초안 검증 및 수정 루프 제어

## Architecture

기본 흐름은 아래와 같습니다.

`Plan -> Retrieve -> (Sufficiency Branch) -> Skeptic? -> Compare -> Write -> Validate -> Finalize`

retrieval 정책은 아래 순서를 따릅니다.

`1차 local RAG -> 2차 balanced web search -> 3차 skeptic re-check`

현재는 local RAG가 placeholder이므로 대부분 web search fallback을 타지만, retrieval sufficiency / skeptic 분기 / 실제 보고서 생성 / safe termination 로직은 상태 전이로 연결되어 있습니다.

## LLM Policy

- `LLM_MODEL`: planner 등 일반 LLM 단계의 기본 모델
- `REPORT_LLM_MODEL`: 보고서 품질이 직접 걸리는 `Compare / SWOT`, `Writer` 단계의 모델
- 기본값은 `LLM_MODEL=gpt-4o-mini`, `REPORT_LLM_MODEL=gpt-4o`

## Prompt Policy

- `planner.md`, `compare_swot.md`, `writer.md`는 실제 system prompt로 사용됩니다.
- 나머지 prompt 파일은 현재 deterministic node의 운영 계약서 역할을 겸합니다.
- 모든 prompt는 코드의 실제 동작과 맞아야 하며, placeholder나 TODO 문구를 남기지 않는 것을 원칙으로 합니다.

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
│   ├── market.md
│   ├── lges.md
│   ├── catl.md
│   ├── skeptic.md
│   ├── compare_swot.md
│   ├── writer.md
│   └── validator.md
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

- Supervisor/validator 등 deterministic node를 실제 LLM handoff 구조로 확장할지 검토
- Chroma 컬렉션 생성 및 로컬 문서 ingestion 구현
- evidence 추출 및 citation formatting 구현
- 본문 인라인 citation 전략 보강

## Contributors

- TODO: Contributor 1
- TODO: Contributor 2
