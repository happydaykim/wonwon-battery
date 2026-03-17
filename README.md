# Battery Strategy Agent

## Subject

배터리 시장 전략 분석 Agent 개발

## Overview

이 프로젝트는 전기차 캐즘 환경에서 LG에너지솔루션과 CATL의 포트폴리오 다각화 전략을 비교 분석하는 멀티 에이전트 시스템의 초기 스캐폴딩입니다.

현재 저장소는 LangGraph/LangChain 기반 협업용 skeleton 상태이며, 실제 비즈니스 로직, 실제 검색, 실제 임베딩, 실제 보고서 생성은 구현하지 않았습니다.

## Features

- Supervisor 단일 패턴 기반 멀티 에이전트 구조
- `StateGraph`, `START`, `END`, `TypedDict` 기반 상태 설계
- Planner, Supervisor, Market, LGES, CATL, Skeptic, Compare/SWOT, Writer, Validator 노드 분리
- `local RAG -> balanced web search -> skeptic re-check` 정책용 인터페이스 skeleton
- Prompt 파일 분리
- `MemorySaver` 기반 그래프 컴파일 지점 제공
- `.env.example`, `requirements.txt`, 기본 디렉터리 구조 포함

## Tech Stack

- Python 3.11+
- LangGraph
- LangChain
- Chroma
- Qwen3-Embedding-0.6B
- python-dotenv

## Agents

- Planner Node: 사용자 목표를 보고 섹션 계획을 정리
- Supervisor Agent: 현재 phase 기준으로 다음 specialist를 라우팅
- Market Agent: EV 캐즘 및 산업 배경 조사 skeleton
- LGES Agent: LG에너지솔루션 전략 조사 skeleton
- CATL Agent: CATL 전략 조사 skeleton
- Skeptic Agent: 반대 근거와 리스크 재검증 skeleton
- Compare / SWOT Agent: 비교 프레임 정렬 및 SWOT 구조화 skeleton
- Writer Agent: 보고서 초안 생성 skeleton
- Validator Agent: 초안 검증 및 수정 루프 제어

## Architecture

기본 흐름은 아래와 같습니다.

`Plan -> Retrieve -> Analyze -> Compare -> Write -> Validate -> Reflect -> Finalize`

retrieval 정책은 아래 순서를 따릅니다.

`1차 local RAG -> 2차 balanced web search -> 3차 skeptic re-check`

현재는 skeleton 상태이므로 retrieval, 분석, 작성 로직은 모두 TODO/stub 입니다.

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

- LangChain `create_agent` 또는 동등한 factory에 실제 모델 연결
- Chroma 컬렉션 생성 및 로컬 문서 ingestion 구현
- balanced web search provider 연결
- evidence 추출 및 citation formatting 구현
- Writer/Validator 기준을 실제 보고서 출력 포맷과 연결

## Contributors

- TODO: Contributor 1
- TODO: Contributor 2
