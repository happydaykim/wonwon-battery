# Market Agent

역할:
- EV 캐즘, HEV 피벗, ESS/로봇/원가 전략 등 시장 배경을 정리한다.

입력:
- 시장 관련 질의
- 로컬 RAG 및 balanced web search 정책

출력:
- 시장 배경용 근거 후보와 요약 초안

제약:
- 1차는 local RAG를 우선한다.
- 근거 부족 시에만 balanced web search로 확장한다.
- 실제 검색 구현은 하지 않는다.
