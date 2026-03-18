# AGENTS.md

## Purpose

이 저장소는 배터리 시장 전략 분석 멀티에이전트 시스템의 협업용 코드베이스다.
현재 구현은 LangGraph/LangChain 기반 scaffold에서 한 단계 더 나아가, planner queue, LLM supervisor routing, local-first retrieval expansion decision, gap 기반 query refinement, Skeptic 재검증, Supervisor 단일 종료, 내부 provenance trace를 유지하는 한국어 보고서 본문 생성과 HTML/PDF export까지 연결된 상태다.

이 문서는 이 저장소에서 작업하는 사람이나 코딩 에이전트가 현재 구조와 작업 원칙을 빠르게 이해하도록 돕기 위한 운영 가이드다.

## Current Architecture

- 패턴: Supervisor pattern
- 실행 구조:
  - `Planner`가 broad step queue를 만든다.
  - `Supervisor`가 `state["plan"]`의 현재 step을 해석해 다음 node를 선택한다.
  - `Planner`를 제외한 모든 specialist agent는 실행 후 다시 `Supervisor`로 복귀한다.
  - `Supervisor`만 specialist 실행 순서를 결정하고, `Supervisor`만 최종 `END`로 나간다.
  - `parallel_retrieval` 단계에서 `Market`, `LGES`, `CATL`이 병렬 실행된다.
  - retrieval bundle이 끝나면 `Supervisor`가 sufficiency를 보고 남은 queue를 동적으로 재작성한다.
  - 회사 정보가 충분하면 바로 `Compare -> Write -> Validate`로 간다.
  - 회사 정보가 부족하면 필요한 회사에 대해서만 `Skeptic -> Compare -> Write -> Validate`로 이어진다.
  - `Validator`는 retryable / non-retryable issue와 `termination_reason`을 계산하지만, 그래프 종료 자체는 하지 않는다.
  - 최종 `END`는 queue가 비었을 때 `Supervisor`가 처리한다.
- 현재 broad step queue:
  - `parallel_retrieval`
  - `skeptic_lges`
  - `skeptic_catl`
  - `compare`
  - `write`
  - `validate`
  - planner는 위 broad step ID를 유지하지만, 실제 실행 시 skeptic step은 supervisor가 제거/유지할 수 있다.

## Retrieval Policy

- 검색 원칙:
  - `1차 local RAG`
  - `2차 LLM retrieval decision`
  - `3차 web search`
  - `4차 gap 기반 query refinement`
  - `5차 Skeptic Agent 재검증`
- 현재 구현 상태:
  - `retrieval/local_rag.py`는 Chroma 기반 실제 local retrieval을 수행한다.
  - local 결과는 공통 retrieval contract로 normalize되어 sufficiency 판단과 artifact 생성에 직접 사용된다.
  - local 결과를 본 뒤 web search로 확장할지, 아니면 현재 수집본에서 멈출지를 `retrieval/retrieval_decider.py`가 먼저 결정한다.
  - merged 결과를 본 뒤 한 번 더 refined query round를 돌릴지도 같은 decider가 결정한다.
  - query refinement는 decider가 추가 round를 허용했을 때만 실행된다.
  - query refinement는 기본적으로 `LLM_MODEL`을 사용해 시도하고, LLM/API key가 없거나 실패하면 deterministic fallback query로 내려간다.
  - `app.py`는 시작 시 local RAG 자원을 prewarm하여 병렬 retrieval 첫 진입 시 초기화 race를 줄인다.
  - `retrieval/vector_store.py`, `retrieval/embeddings.py`는 cache + lock으로 Chroma collection과 embedding backend 초기화를 공유한다.
  - web search는 `langchain_teddynote.tools.GoogleNews` adapter를 통해 수행한다.
  - query는 한국어/영어 변형으로 이중 실행된다.
  - 결과는 `positive_results`, `risk_results`로 normalize 된다.
  - retrieval state에는 `query_history`, `refinement_rounds`, `decision_notes`, `retrieval_failures`가 남아 이후 summary / validator / debugging에 사용된다.
  - `used_web_search`는 web search 결과가 남았는지가 아니라, web search를 실제로 한 번이라도 시도했는지를 뜻한다.
  - local/web helper는 예외를 바로 터뜨리기보다 warning log + failure note + empty result로 내려와 supervisor 흐름을 유지한다.
  - validator는 순수 정보 부족 gap과 retrieval execution failure를 구분해서 다룬다.
  - Skeptic도 risk-only web retrieval + refinement loop를 사용한다.
  - retrieval 단계는 콘솔 로그로 현재 진행 상황을 출력한다.
    - `Retrieval round N executing queries: ...`
    - `Local RAG hits=...`
    - `Local RAG returned 0 hits. Falling back to web search.`
    - `Local RAG was insufficient. Falling back to web search.`
    - `Local RAG was sufficient. Web search skipped.`
    - `Web search hits: positive=..., risk=...`
    - `documents=..., evidence=..., preview_titles=[...]`
- Sufficiency rule:
  - 최소 evidence 개수
  - distinct coverage unit 개수
  - source 다양성
  - positive/risk 균형
  - scope별 필수 topic tag 커버리지
  - 위 조건 중 하나라도 부족하면 insufficient로 판단한다.
  - web/news 결과는 같은 story title/date 반복이 coverage를 부풀리지 않도록 보수적으로 묶는다.
  - local RAG는 문서 전체 1건으로 뭉개지지 않도록 chunk/page 단위 identity를 유지한다.
  - local 결과 sufficiency는 LLM decider의 입력으로 사용되며, web 확장 여부를 직접 강제하지는 않는다.
  - merged 결과 sufficiency는 refinement 지속 여부, supervisor 분기, validator 종료 판단의 주요 입력으로 사용된다.

## Report Generation Policy

- 현재 보고서는 실제 사람이 읽는 한국어 전략 분석 보고서로 생성된다.
- 최종 산출물은 HTML/PDF 파일로 저장된다.
  - 저장 경로: `outputs/{thread_id}_{YYYYMMDD_HHMMSS}.html`
  - 저장 경로: `outputs/{thread_id}_{YYYYMMDD_HHMMSS}.pdf`
  - `outputs/`는 `.gitignore`에 포함되어 있으며 저장소에 커밋하지 않는다.
- `final_report`는 내부적으로 writer/validator가 다루는 보고서 본문이며, export 단계에서 reader-facing HTML/PDF로 렌더된다.
- 필수 목차는 아래 구조를 따른다.
  - `1. SUMMARY`
  - `2. 시장 배경`
    - writer는 `### 2.x` 형식의 소제목을 자율적으로 2개 이상 설계할 수 있다.
  - `3. LG에너지솔루션의 포트폴리오 다각화 전략과 핵심 경쟁력`
  - `4. CATL의 포트폴리오 다각화 전략과 핵심 경쟁력`
  - `5. 핵심 전략 비교 분석`
    - `5.1 전략 방향 차이`
    - `5.2 데이터 기반 비교표`
    - `5.3 SWOT 분석`
  - `6. 종합 시사점`
  - `7. REFERENCE`
- PDF export는 기본적으로 연속 문서 레이아웃을 사용한다.
  - 보고서는 페이지별 대제목 고정 배치보다, 본문이 자연스럽게 이어지는 dense layout을 우선한다.
  - 다만 제목 orphan, 표/행렬 단절, summary box 분리처럼 가독성이 크게 나빠지는 경우에는 제한적으로 page break를 허용한다.
  - HTML과 PDF는 같은 section 순서를 유지하되, PDF는 print layout과 page-break-inside 제어만 추가한다.
- `SUMMARY`는 executive summary이며 900자 이내로 관리한다.
- `REFERENCE`는 실제로 사용한 자료만 남기며 일반 참고문헌 라인 형식으로 출력한다.
- 최종 Markdown 본문에는 inline citation label을 노출하지 않는다.
- 대신 `section_drafts[*].citations`에 문장/불릿/표 단위 evidence trace를 내부적으로 남긴다.
- `REFERENCE`는 state에 있는 모든 문서를 나열하지 않고, 내부 citation trace에서 실제 사용된 verifiable document만 남긴다.
- `2장`은 writer가 `### 2.x` 형식의 소제목을 자율적으로 정하고, 필요하면 `3장`, `4장`, `6장`에도 numbered subsection을 추가할 수 있다.
- 보고서 본문은 얇은 브리핑 메모보다 실제 보고서 문단 밀도를 우선한다.
- 보고서 본문은 가능하면 근거 자료에 실제로 있는 정량 수치를 포함하되, evidence/정량 근거 block에 없는 숫자는 새로 만들지 않는다.
- 근거가 부족한 축은 hallucination 대신 `정보 부족/추가 검증 필요` 또는 동등한 caveat로 드러낸다.

## LLM Policy

- 일반 기본 모델:
  - `LLM_PROVIDER`
  - `LLM_MODEL`
- 보고서 품질이 직접 걸리는 단계용 모델:
  - `REPORT_LLM_PROVIDER`
  - `REPORT_LLM_MODEL`
  - `WRITER_LLM_PROVIDER`
  - `WRITER_LLM_MODEL`
- 현재 기본값:
  - `LLM_MODEL=gpt-4o-mini`
  - `REPORT_LLM_MODEL=gpt-4o`
  - `WRITER_LLM_MODEL=gpt-4o`
- `Compare / SWOT`는 기본적으로 `REPORT_LLM_MODEL`을 사용한다.
- `Writer`는 기본적으로 `WRITER_LLM_MODEL`을 사용한다.
- 현재 기본 `Writer` 모델은 `gpt-4o`이며, writer 품질 저하를 피하려면 가능하면 유지한다.
- `Planner`는 기본 `LLM_MODEL`을 사용하고, 실패하면 deterministic fallback plan으로 내려간다.

## Prompt Policy

- `prompts/planner.md`, `prompts/supervisor.md`, `prompts/retrieval_decider.md`, `prompts/query_refiner.md`, `prompts/compare_swot.md`, `prompts/writer.md`는 실제 LLM system prompt로 사용된다.
- `prompts/vision/common.md`, `prompts/vision/other.md`는 ingestion 시 vision 분석 prompt로 사용된다.
- 모든 prompt 파일은 실제 코드 동작과 맞아야 한다.
- placeholder 문구나 `TODO 수준의 skeleton` 같은 설명은 남기지 않는다.
- prompt를 수정할 때는 해당 단계가 실제로 모델 입력에 쓰는 파일인지 먼저 확인한다.

## Key Files

- `app.py`
  - 초기 상태를 만들고 그래프를 실행하는 entrypoint.
  - `PLAN_STEP_LABELS`와 `build_initial_state()`가 workflow queue 구조를 설명한다.
  - app startup 시 local RAG prewarm을 수행할 수 있다.
  - 실행 종료 후 `final_report`가 있으면 HTML/PDF report artifact를 `outputs/`에 저장한다.
- `graph/builder.py`
  - LangGraph 노드와 edge를 연결하는 핵심 파일.
  - `parallel_retrieval_dispatch` fan-out과 join, supervisor route, 모든 specialist의 supervisor 복귀 edge가 여기서 조립된다.
- `graph/router.py`
  - `state["plan"]`의 현재 step을 실제 node name으로 매핑한다.
  - `Supervisor`만 `done -> END`로 나가며, `Validator`는 더 이상 직접 branch하지 않는다.
  - step 이름을 바꾸면 반드시 여기와 `app.py` label, 관련 agent를 같이 수정해야 한다.
- `agents/__init__.py`, `graph/__init__.py`
  - package import 시 하위 모듈을 eager import하지 않고 lazy export를 사용한다.
  - direct module import와 test loader에서 circular import를 피하기 위한 안전장치다.
- `schemas/state.py`
  - 전체 상태 스키마의 source of truth.
  - `documents`, `evidence`, `companies`는 병렬 retrieval merge를 위해 reducer를 사용한다.
  - retrieval debug용 `query_history`, `refinement_rounds`, `decision_notes`, `retrieval_failures`와 내부 citation trace용 `SectionDraft.citations`도 여기 정의된다.
- `retrieval/pipeline.py`
  - retrieval orchestration의 중심이다.
  - sufficiency 판단, local/web merge, LLM decider 호출, refinement loop, retry-safe helper, artifacts 변환, retrieval summary 생성이 여기 있다.
  - raw evidence count와 distinct coverage count를 함께 다루며, same-story repetition이 web sufficiency를 과대평가하지 않도록 보수적으로 계산한다.
  - retry-safe helper는 failure note를 남기고 empty result로 내려와 graph 흐름을 유지한다.
  - local sufficiency 판단, retrieval decision, refinement round, web search hit 수를 콘솔에 기록한다.
- `retrieval/retrieval_decider.py`
  - local-first 정책을 유지한 채 다음 retrieval action을 LLM structured output으로 결정한다.
  - web search 확장 여부와 refinement continuation 여부를 여기서 판단한다.
  - 판단 입력에는 evidence/source count뿐 아니라 coverage count와 web search attempt 여부도 포함된다.
- `retrieval/query_refiner.py`
  - retrieval gap을 바탕으로 추가 positive/risk query를 생성한다.
  - LLM 기반 refinement와 deterministic fallback을 함께 제공한다.
- `retrieval/vector_store.py`
  - persistent Chroma collection 접근 지점이다.
  - path/collection 기준 cache와 lock으로 병렬 초기화를 안정화한다.
- `retrieval/embeddings.py`
  - shared embedding backend 로딩과 model id normalization을 담당한다.
  - legacy alias를 canonical Hugging Face model id로 정규화한다.
- `retrieval/balanced_web_search.py`
  - `langchain_teddynote.tools.GoogleNews` 기반 adapter.
  - positive/risk 검색, 한/영 query expansion, normalized item 생성이 여기 있다.
- `retrieval/query_policy.py`
  - Market/LGES/CATL별 positive/risk query policy 정의.
- `utils/logging.py`
  - 공통 로깅 설정과 LangSmith 연동 지점이다.
  - 기본값으로 noisy third-party log와 progress bar를 줄인다.
- `agents/market.py`, `agents/lges.py`, `agents/catl.py`
  - `local RAG -> LLM decider -> web expansion/refinement -> artifacts -> summary` 흐름을 동일한 패턴으로 수행한다.
  - 최종 documents/evidence 개수와 preview title을 콘솔에 기록한다.
- `agents/skeptic.py`
  - retrieval layer를 통해 risk 중심 counter-evidence를 보강한다.
  - `counter_evidence_ids`, skeptic 완료 여부, 재평가된 retrieval gap을 갱신한다.
- `agents/compare_swot.py`
  - 보고서 5장에 들어갈 비교 서술, Markdown 비교표, SWOT 구조를 만든다.
  - 정량 근거 packet을 함께 받아 실제 근거에 있는 수치를 비교표에 우선 반영한다.
  - `REPORT_LLM_MODEL`을 사용한다.
- `agents/writer.py`
  - 사람이 읽는 한국어 보고서 본문과 reference section을 조립한다.
  - `2장`의 numbered subsection을 writer가 자율적으로 정하고, 필요하면 `3장`, `4장`, `6장`에도 subsection을 추가할 수 있다.
  - evidence packet과 별도의 정량 근거 packet을 함께 받아 수치를 근거 기반으로 풀어 쓴다.
  - 내부 citation trace를 남기되, 최종 Markdown에는 visible citation label을 넣지 않는다.
  - 실제로 cited된 document만 REFERENCE에 남긴다.
  - `WRITER_LLM_MODEL`을 사용한다.
- `agents/validator.py`
  - 필수 헤딩, writer-chosen subsection 구조, SUMMARY 길이, reference 형식, placeholder 문구, retrieval gap 노출 여부를 검증한다.
  - retrieval failure와 pure information gap을 분리해 warning issue로 올린다.
  - 내부 citation trace 존재 여부와 citation trace가 REFERENCE로 resolve되는지도 검증한다.
- `utils/evidence_context.py`
  - 일반 evidence packet과 정량 근거 packet을 함께 만든다.
  - writer/compare가 실제 자료에 있는 숫자를 더 쉽게 가져가도록 numeric snippet을 추출한다.
- `utils/citation_linker.py`
  - 문장/불릿/표 단위 evidence trace를 section draft에 붙인다.
  - visible inline markup 없이 internal provenance만 남기는 정책이 여기 구현된다.
- `utils/report_export.py`
  - `final_report` 및 `section_drafts`를 reader-facing HTML/PDF artifact로 렌더한다.
  - PDF에서는 주요 섹션이 새 페이지에서 시작되도록 page grouping과 print layout을 조정한다.
  - 비교 섹션과 SWOT을 같은 PDF page group으로 묶어 기본 보고서 구조의 pagination을 안정화한다.
- `config/settings.py`
  - 기본 LLM과 보고서용 LLM 설정을 함께 관리한다.
  - retrieval refinement budget도 여기서 관리한다.
- `prompts/*.md`
  - 실제 runtime에서 호출되는 system prompt만 유지한다.

## Working Rules

- queue semantics를 유지한다.
  - 현재 워크플로는 `state["plan"]`을 실행 큐로 사용한다.
  - step을 추가/삭제/개명하면 `planner`, `router`, `app.py`, 관련 agent를 함께 맞춘다.
  - 다만 `parallel_retrieval` 이후 `Supervisor`가 남은 queue를 재작성하는 현재 semantics는 유지한다.
- retrieval 정책을 임의로 바꾸지 않는다.
  - 현재 기본 원칙은 `local RAG -> LLM retrieval decision -> GoogleNews web search -> query refinement -> Skeptic`이다.
  - local RAG는 반드시 먼저 실행한다.
  - web search와 refinement continuation 여부는 retrieval layer의 LLM decider가 결정한다.
  - refinement는 retrieval layer 안에서만 수행하고, agent가 직접 ad-hoc query를 하드코딩하지 않는다.
  - `used_web_search`는 “web 결과가 있었다”가 아니라 “web을 실제로 시도했다”는 의미로 유지한다.
  - provider/runtime failure는 가능한 한 retrieval layer에서 note + warning + empty result로 흡수하고, validator가 최종적으로 gap과 failure를 함께 노출하도록 유지한다.
  - same-story repetition이 sufficiency를 부풀리지 않도록 coverage 계산을 약화시키는 방향으로 되돌리지 않는다.
- local provider 초기화는 공유한다.
  - Chroma collection과 embedding backend는 shared cache를 통해 재사용한다.
  - agent 단에서 provider를 매번 독자적으로 초기화하는 방향으로 되돌리지 않는다.
- web search provider는 retrieval layer에만 캡슐화한다.
  - agent는 provider 세부 구현을 몰라야 한다.
  - provider를 바꿀 때는 우선 `retrieval/balanced_web_search.py`만 수정한다.
- state 구조를 함부로 바꾸지 않는다.
  - `ReportState`는 graph, agents, retrieval helper가 함께 사용한다.
  - 병렬 단계에서 merge 충돌이 없는지 항상 확인한다.
- 보고서 provenance 정책을 임의로 바꾸지 않는다.
  - visible citation label은 최종 Markdown에 넣지 않는다.
  - reader-facing Markdown formatting과 내부 provenance trace를 분리해서 유지한다.
  - 대신 `SectionDraft.citations`와 `references`의 일관성을 유지한다.
- prompt는 코드에 하드코딩하지 않는다.
  - 시스템 프롬프트는 `prompts/*.md`에 두고 agent는 loader를 통해 읽는다.
- 출력 산출물은 저장소에 고정하지 않는다.
  - `outputs/*.html`, `outputs/*.pdf`는 런타임 산출물이며 커밋하지 않는다.
- 실제 기업 분석 결과를 저장소에 고정하지 않는다.
  - scaffold 단계에서는 결론보다 구조, state 흐름, retrieval 연결이 우선이다.

## Editing Guidance

- 새로운 broad step을 추가할 때:
  - `agents/planner.py`
  - `graph/router.py`
  - `agents/supervisor.py`
  - `app.py`
  - 필요한 agent 파일
  - 위 다섯 곳을 함께 본다.
- retrieval 정책을 확장할 때:
  - `retrieval/local_rag.py`
  - `retrieval/embeddings.py`
  - `retrieval/vector_store.py`
  - `retrieval/balanced_web_search.py`
  - `retrieval/query_policy.py`
  - `retrieval/query_refiner.py`
  - `retrieval/pipeline.py`
  - 필요하면 `app.py` prewarm 경로와 `config/settings.py`도 함께 수정한다.
- 새로운 state 필드를 추가할 때:
  - `schemas/state.py`
  - `app.py`의 `build_initial_state()`
  - 해당 필드를 읽거나 쓰는 agent/retrieval helper
  - 병렬 merge가 필요한 필드면 reducer도 함께 검토한다.
- prompt 파일 이름을 바꿀 때:
  - 해당 agent의 blueprint `prompt_name`과 반드시 같이 맞춘다.
- 보고서 목차나 validator 규칙을 바꿀 때:
  - `agents/writer.py`
  - `agents/compare_swot.py`
  - `agents/validator.py`
  - `utils/citation_linker.py`
  - `app.py`
  - 관련 prompt 파일
  - 위 여섯 곳을 함께 본다.
- LLM 모델 정책을 바꿀 때:
  - `config/settings.py`
  - `.env.example`
  - 실제 모델을 생성하는 agent 파일
  - 관련 테스트
  - 위 네 곳을 함께 본다.

## Local Run Notes

- 권장 Python 버전: 3.11.11
- 기본 설치:
  - `uv sync`
- 환경 변수:
  - `.env.example`을 기준으로 `.env`를 채운다.
  - 보고서 품질을 유지하려면 `REPORT_LLM_MODEL`이 기대한 값인지 먼저 확인한다.
  - 기본값으로 `LOCAL_RAG_PREWARM_ENABLED=true`, `QUIET_THIRD_PARTY_LOGS=true`가 적용된다.
  - retrieval refinement는 `RETRIEVAL_REFINEMENT_MAX_ROUNDS`, `RETRIEVAL_REFINEMENT_MAX_QUERIES_PER_BUCKET`로 조절한다.
  - 초기 embedding 다운로드나 rate limit 문제가 있으면 `HF_TOKEN` 설정을 검토한다.
- 실행:
  - `uv run app.py`
  - 또는 `./.venv/bin/python app.py`
- 디버깅 팁:
  - retrieval 콘솔 로그만 조용히 보고 싶으면 `LANGSMITH_ENABLED=false ./.venv/bin/python app.py`를 우선 사용한다.
  - third-party 상세 로그가 필요하면 `QUIET_THIRD_PARTY_LOGS=false`로 일시 해제한다.
  - startup에서 `Local RAG resources prewarmed: ...`가 찍히면 prewarm이 정상 동작한 것이다.
  - refinement가 돌면 `Retrieval round N executing queries: ...` 로그가 추가로 보인다.
  - 보고서 생성 모델은 실행 로그의 아래 문구로 확인할 수 있다.
    - `Compare/SWOT report generation model configured: ...`
    - `Writer report generation model configured: ...`

## Known Runtime Behavior

- 외부 네트워크가 없으면:
  - planner LLM은 fallback plan으로 내려간다.
  - GoogleNews는 빈 결과 또는 provider 경고를 낼 수 있다.
  - LangSmith는 연결 경고를 남길 수 있다.
  - query refiner는 deterministic fallback query로 내려가거나, 추가 query 없이 종료할 수 있다.
  - 이 경우에도 retrieval 로그는 `local RAG 실행 -> LLM retrieval decision -> web search 시도(실패 가능) -> refinement 여부 판단` 형태로 남는다.
  - web search를 실제로 시도했지만 결과가 비거나 provider가 실패한 경우에도 `used_web_search=true`와 `retrieval_failures`가 state에 남는다.
  - validator는 이를 단순 정보 부족과 별도의 retrieval failure warning으로 노출할 수 있다.
- 첫 실행이나 캐시 미스 상태에서는:
  - embedding model download/load 때문에 startup이 느릴 수 있다.
  - prewarm이 활성화되어 있으면 이 비용은 retrieval 시작 전에 먼저 지불된다.
- 현재 `langchain_teddynote.tools.GoogleNews`는 간단한 RSS keyword search wrapper다.
  - 설정에는 `GOOGLE_NEWS_PERIOD`가 남아 있지만, 현재 provider 구현에서는 강제 기간 필터를 직접 적용하지 않는다.
  - 이 값은 향후 provider 교체 또는 확장을 위한 placeholder 성격이 있다.
- Writer/Compare는 현재 실제 보고서 본문을 생성한다.
  - local corpus와 web source가 함께 쓰일 수 있지만, source diversity gap이나 topic coverage gap은 여전히 남을 수 있다.
  - visible inline citation은 제거되었고, provenance는 state 내부 trace로만 유지된다.
- Validator는 retryable / non-retryable issue를 구분하고 `termination_reason`을 계산한다.
  - retryable issue가 있고 revision budget이 남아 있으면 `write -> validate`를 다시 queue에 넣는다.
  - retryable issue가 없고 gap만 남으면 `done_with_gaps`를 남긴다.
  - 실제 `END`는 이후 supervisor가 빈 queue를 보고 처리한다.
- local RAG 품질은 Chroma corpus와 metadata 정합성, embedding 설정에 크게 의존한다.

## Guardrails

- 이 저장소는 아직 “구조 안정화 + 검색 흐름 연결 + 보고서 품질 개선” 단계다.
- retrieval path와 state 흐름이 깨지지 않는 것이 개별 문장 품질보다 우선이다.
- 큰 리팩터링 전에는 최소한 아래 다섯 축이 계속 맞는지 확인한다:
  - 상태 스키마
  - planner queue
  - supervisor/router 규칙
  - parallel retrieval merge
  - prompt/agent 파일 대응 관계

## Tests

- 현재 테스트는 `unittest` 기반이다.
- 기본 실행:
  - `./.venv/bin/python -m unittest discover -s tests`
- 중요한 테스트 축:
  - retrieval sufficiency 및 종료 조건
  - direct agent/graph import safety와 circular import 방지
  - local-first retrieval과 LLM 기반 web expansion decision
  - web search attempt/failure tracking과 validator failure surfacing
  - gap 기반 query refinement loop
  - same-story repetition을 보수적으로 처리하는 coverage 판단
  - vector store cache 및 병렬 초기화 안정성
  - supervisor post-retrieval branching 및 supervisor 단일 종료 구조
  - skeptic counter-evidence 보강
  - app startup prewarm 및 logging 설정
  - writer reference filtering 및 internal citation trace
  - validator 구조/형식/citation trace 검증
  - report model selection
  - prompt contract 품질 검사

## If a Task Includes a Design Artifact

- 설계 문서, PDF, DOCX 같은 별도 artifact가 함께 주어지면 그것을 source of truth로 우선한다.
- 현재 저장소 구현과 충돌하면 충돌 지점을 명시하고 아래 항목을 함께 정렬한다:
  - `schemas/state.py`
  - `agents/planner.py`
  - `graph/router.py`
  - `graph/builder.py`
  - 관련 retrieval/agent 파일
