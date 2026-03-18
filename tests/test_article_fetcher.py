from __future__ import annotations

import unittest

from retrieval.article_fetcher import (
    ArticleFetchResult,
    GoogleNewsDecodeParams,
    _extract_google_news_decode_params,
    _parse_google_news_batch_response,
    extract_article_content,
)
from retrieval.pipeline import enrich_results_with_article_content


class _FakeArticleFetcher:
    def __init__(self, result: ArticleFetchResult | None) -> None:
        self._result = result

    def fetch(self, url: str | None) -> ArticleFetchResult | None:
        _ = url
        return self._result


class ArticleFetcherTests(unittest.TestCase):
    def test_extract_google_news_decode_params_reads_timestamp_and_signature(self) -> None:
        html = """
        <html>
          <body>
            <c-wiz>
              <div
                jslog="x"
                data-n-a-ts="1773794478"
                data-n-a-sg="AZ5r3eRWtbhkLmgwo-YJ20d2_Hyd">
              </div>
            </c-wiz>
          </body>
        </html>
        """

        params = _extract_google_news_decode_params(
            html,
            encoded_id="CBMibEFV-test",
        )

        self.assertEqual(
            GoogleNewsDecodeParams(
                article_id="CBMibEFV-test",
                timestamp="1773794478",
                signature="AZ5r3eRWtbhkLmgwo-YJ20d2_Hyd",
            ),
            params,
        )

    def test_parse_google_news_batch_response_returns_decoded_url(self) -> None:
        response_text = """\
)]}'

[["wrb.fr","Fbv4je","[\\"garturlres\\",\\"https://www.electimes.com/news/articleView.html?idxno\\\\u003d365645\\",1,\\"https://www.electimes.com/news/articleViewAmp.html?idxno\\\\u003d365645\\"]",null,null,null,""],["di",12],["af.httprm",12,"7828634099389876302",36]]
"""

        decoded_url = _parse_google_news_batch_response(response_text)

        self.assertEqual(
            "https://www.electimes.com/news/articleView.html?idxno=365645",
            decoded_url,
        )

    def test_extract_article_content_prefers_article_body_over_meta(self) -> None:
        html = """
        <html>
          <head>
            <title>Example Title</title>
            <meta property="og:description" content="short description" />
            <script type="application/ld+json">
              {
                "@context": "https://schema.org",
                "@type": "NewsArticle",
                "headline": "Example Title",
                "datePublished": "2026-01-30T11:32:45+09:00",
                "articleBody": "첫 문단입니다. 실제 기사 본문이 충분히 길게 이어집니다. 두 번째 문장도 포함됩니다."
              }
            </script>
          </head>
          <body>
            <article>
              <p>이 문장은 JSON-LD보다 짧아서 우선되지 않아야 합니다.</p>
            </article>
          </body>
        </html>
        """

        parsed = extract_article_content(html, char_limit=4000)

        self.assertIsNotNone(parsed)
        assert parsed is not None
        self.assertEqual("Example Title", parsed["title"])
        self.assertEqual("2026-01-30", parsed["published_at"])
        self.assertIn("실제 기사 본문", parsed["full_text"])
        self.assertIn("첫 문단입니다", parsed["excerpt"])

    def test_extract_article_content_reads_published_date_from_meta_tag(self) -> None:
        html = """
        <html>
          <head>
            <meta property="og:title" content="배터리 시장 기사" />
            <meta property="article:published_time" content="2025-12-09T08:15:00Z" />
          </head>
          <body>
            <article>
              <p>배터리 관련 실제 기사 본문이 충분한 길이로 이어지며 날짜 추출 테스트를 지원한다.</p>
              <p>두 번째 문단도 포함되어 기사 본문 판별 기준을 충족한다.</p>
            </article>
          </body>
        </html>
        """

        parsed = extract_article_content(html, char_limit=4000)

        self.assertIsNotNone(parsed)
        assert parsed is not None
        self.assertEqual("2025-12-09", parsed["published_at"])

    def test_extract_article_content_strips_daum_example_prefix(self) -> None:
        html = """
        <html>
          <head>
            <meta property="og:site_name" content="Daum | 이투데이" />
            <meta property="og:title" content="중국 추격 속 기술 승부" />
          </head>
          <body>
            <article>
              <p>(예시) 가장 빠른 뉴스가 있고 다양한 정보, 쌍방향 소통이 숨쉬는 다음뉴스를 만나보세요. 다음뉴스는 국내외 주요이슈와 실시간 속보, 문화생활 및 다양한 분야의 뉴스를 입체적으로 전달하고 있습니다.</p>
              <p>국내 배터리 업체들이 전기차 수요 둔화가 길어지면서 생산 확대 중심 투자에서 기술 중심 전략으로 무게추를 옮기고 있다.</p>
              <p>설비투자는 줄이는 대신 연구개발에 집중하며 차세대 기술 확보에 속도를 내는 모습이다.</p>
            </article>
          </body>
        </html>
        """

        parsed = extract_article_content(html, char_limit=4000)

        self.assertIsNotNone(parsed)
        assert parsed is not None
        self.assertEqual("이투데이", parsed["publisher_name"])
        self.assertNotIn("(예시)", parsed["full_text"])
        self.assertNotIn("가장 빠른 뉴스가 있고 다양한 정보", parsed["full_text"])
        self.assertTrue(parsed["full_text"].startswith("국내 배터리 업체들이"))
        self.assertTrue(parsed["excerpt"].startswith("국내 배터리 업체들이"))

    def test_enrich_results_with_article_content_injects_full_text(self) -> None:
        merged_results = {
            "positive_results": [
                {
                    "title": "Search title",
                    "link": "https://example.com/article",
                    "source": "SourceA",
                    "published_at": "2026-03-18",
                    "snippet": "search snippet",
                    "query": "LGES 포트폴리오 다각화",
                    "stance": "positive",
                    "topic_tags": ["strategy", "expansion"],
                }
            ],
            "risk_results": [],
        }

        enriched = enrich_results_with_article_content(
            merged_results,
            article_fetcher=_FakeArticleFetcher(
                ArticleFetchResult(
                    resolved_url="https://publisher.example.com/final",
                    publisher_name="Publisher Example",
                    title="Fetched article title",
                    published_at="2025-12-09",
                    excerpt="Fetched article excerpt",
                    full_text="Fetched full article text with meaningful details.",
                )
            ),
            max_documents=2,
            company_scope="LGES",
        )

        enriched_item = enriched["positive_results"][0]
        self.assertEqual("https://publisher.example.com/final", enriched_item["link"])
        self.assertEqual("Publisher Example", enriched_item["source"])
        self.assertEqual("Fetched article title", enriched_item["title"])
        self.assertEqual("2025-12-09", enriched_item["published_at"])
        self.assertEqual("Fetched article excerpt", enriched_item["snippet"])
        self.assertEqual(
            "Fetched full article text with meaningful details.",
            enriched_item["article_text"],
        )


if __name__ == "__main__":
    unittest.main()
