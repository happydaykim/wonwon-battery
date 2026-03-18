from __future__ import annotations

from dataclasses import dataclass
import base64
import html
import json
import re
from typing import Any
from urllib.parse import quote, urlparse

import httpx

from config.settings import Settings, load_settings
from utils.logging import get_logger


logger = get_logger(__name__)

USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/124.0.0.0 Safari/537.36"
)

LEADING_BOILERPLATE_PATTERNS = (
    re.compile(
        r"^\(예시\)\s*가장 빠른 뉴스가 있고 다양한 정보, 쌍방향 소통이 숨쉬는 다음뉴스를 만나보세요\.\s*"
        r"다음뉴스는 국내외 주요이슈와 실시간 속보, 문화생활 및 다양한 분야의 뉴스를 입체적으로 전달하고 있습니다\.\s*"
    ),
    re.compile(
        r"^가장 빠른 뉴스가 있고 다양한 정보, 쌍방향 소통이 숨쉬는 다음뉴스를 만나보세요\.\s*"
        r"다음뉴스는 국내외 주요이슈와 실시간 속보, 문화생활 및 다양한 분야의 뉴스를 입체적으로 전달하고 있습니다\.\s*"
    ),
)


@dataclass(frozen=True, slots=True)
class ArticleFetchResult:
    resolved_url: str | None
    publisher_name: str | None
    title: str | None
    excerpt: str | None
    full_text: str | None


@dataclass(frozen=True, slots=True)
class GoogleNewsDecodeParams:
    article_id: str
    timestamp: str
    signature: str


@dataclass(slots=True)
class ArticleContentFetcher:
    timeout_seconds: int
    max_retries: int
    char_limit: int

    @classmethod
    def from_settings(
        cls,
        settings: Settings | None = None,
    ) -> "ArticleContentFetcher":
        resolved = settings or load_settings()
        return cls(
            timeout_seconds=resolved.article_fetch_timeout_seconds,
            max_retries=resolved.article_fetch_max_retries,
            char_limit=resolved.article_fetch_char_limit,
        )

    def fetch(self, url: str | None) -> ArticleFetchResult | None:
        if not url:
            return None

        attempts = 0
        while True:
            try:
                with httpx.Client(
                    follow_redirects=True,
                    timeout=self.timeout_seconds,
                    headers={
                        "User-Agent": USER_AGENT,
                        "Accept": "text/html,application/xhtml+xml",
                    },
                ) as client:
                    resolved_target = decode_google_news_url(url, client=client) or url
                    response = client.get(resolved_target)
                response.raise_for_status()
                content_type = response.headers.get("content-type", "")
                if "html" not in content_type and "xml" not in content_type:
                    return None

                parsed = extract_article_content(
                    response.text,
                    char_limit=self.char_limit,
                )
                if parsed is None:
                    return None

                return ArticleFetchResult(
                    resolved_url=str(response.url),
                    publisher_name=parsed.get("publisher_name")
                    or _infer_publisher_name_from_url(str(response.url)),
                    title=parsed.get("title"),
                    excerpt=parsed.get("excerpt"),
                    full_text=parsed.get("full_text"),
                )
            except Exception as exc:  # pragma: no cover - depends on live network/pages
                if attempts >= self.max_retries:
                    logger.warning(
                        "Article fetch failed for %s after %d attempt(s): %s",
                        resolved_target,
                        attempts + 1,
                        exc,
                    )
                    return None
                attempts += 1
                logger.warning(
                    "Article fetch failed for %s. Retrying (%d/%d): %s",
                    resolved_target,
                    attempts,
                    self.max_retries,
                    exc,
                )


def extract_article_content(html_text: str, *, char_limit: int) -> dict[str, str] | None:
    normalized_html = html_text or ""
    if not normalized_html.strip():
        return None

    title = (
        _extract_meta_content(normalized_html, "property", "og:title")
        or _extract_meta_content(normalized_html, "name", "twitter:title")
        or _extract_title_tag(normalized_html)
    )
    description = (
        _extract_meta_content(normalized_html, "property", "og:description")
        or _extract_meta_content(normalized_html, "name", "description")
    )
    publisher_name = _normalize_publisher_name(_extract_publisher_name(normalized_html))

    article_body = _extract_article_body_from_json_ld(normalized_html)
    paragraph_text = _extract_paragraph_text(normalized_html)
    full_text = _choose_best_article_text(article_body, paragraph_text, char_limit=char_limit)
    full_text = _sanitize_article_text(full_text)
    excerpt = _build_excerpt(full_text, fallback=description, char_limit=min(700, char_limit))

    if not any((title, excerpt, full_text)):
        return None

    return {
        "publisher_name": publisher_name or "",
        "title": title or "",
        "excerpt": excerpt or "",
        "full_text": full_text or "",
    }


def _extract_article_body_from_json_ld(html_text: str) -> str | None:
    pattern = re.compile(
        r"<script[^>]+type=[\"']application/ld\+json[\"'][^>]*>(.*?)</script>",
        re.IGNORECASE | re.DOTALL,
    )
    candidates: list[str] = []
    for raw_json in pattern.findall(html_text):
        cleaned = html.unescape(raw_json).strip()
        if not cleaned:
            continue
        try:
            payload = json.loads(cleaned)
        except json.JSONDecodeError:
            continue
        candidates.extend(_collect_article_bodies(payload))

    candidates = [candidate for candidate in candidates if candidate]
    if not candidates:
        return None
    return max(candidates, key=len)


def _extract_publisher_name(html_text: str) -> str | None:
    return (
        _extract_meta_content(html_text, "property", "og:site_name")
        or _extract_meta_content(html_text, "name", "application-name")
        or _extract_publisher_from_json_ld(html_text)
    )


def _extract_publisher_from_json_ld(html_text: str) -> str | None:
    pattern = re.compile(
        r"<script[^>]+type=[\"']application/ld\+json[\"'][^>]*>(.*?)</script>",
        re.IGNORECASE | re.DOTALL,
    )
    candidates: list[str] = []
    for raw_json in pattern.findall(html_text):
        cleaned = html.unescape(raw_json).strip()
        if not cleaned:
            continue
        try:
            payload = json.loads(cleaned)
        except json.JSONDecodeError:
            continue
        candidates.extend(_collect_publisher_names(payload))

    cleaned_candidates = [candidate.strip() for candidate in candidates if candidate and candidate.strip()]
    if not cleaned_candidates:
        return None
    return max(cleaned_candidates, key=len)


def _collect_publisher_names(payload: Any) -> list[str]:
    if isinstance(payload, list):
        names: list[str] = []
        for item in payload:
            names.extend(_collect_publisher_names(item))
        return names

    if isinstance(payload, dict):
        names: list[str] = []
        graph = payload.get("@graph")
        if isinstance(graph, list):
            for item in graph:
                names.extend(_collect_publisher_names(item))

        publisher = payload.get("publisher")
        if isinstance(publisher, dict):
            name = publisher.get("name")
            if isinstance(name, str):
                names.append(_clean_text(name))
        elif isinstance(publisher, list):
            for item in publisher:
                if isinstance(item, dict):
                    name = item.get("name")
                    if isinstance(name, str):
                        names.append(_clean_text(name))

        return names

    return []


def _collect_article_bodies(payload: Any) -> list[str]:
    if isinstance(payload, list):
        bodies: list[str] = []
        for item in payload:
            bodies.extend(_collect_article_bodies(item))
        return bodies

    if isinstance(payload, dict):
        bodies: list[str] = []
        graph = payload.get("@graph")
        if isinstance(graph, list):
            for item in graph:
                bodies.extend(_collect_article_bodies(item))

        body = payload.get("articleBody")
        if isinstance(body, str):
            bodies.append(_clean_text(body))

        description = payload.get("description")
        if isinstance(description, str) and len(description) > 120:
            bodies.append(_clean_text(description))

        return bodies

    return []


def _extract_paragraph_text(html_text: str) -> str | None:
    working = re.sub(
        r"<(script|style|noscript|svg|iframe)[^>]*>.*?</\1>",
        " ",
        html_text,
        flags=re.IGNORECASE | re.DOTALL,
    )

    article_match = re.search(
        r"<article\b[^>]*>(.*?)</article>",
        working,
        flags=re.IGNORECASE | re.DOTALL,
    )
    if article_match:
        article_text = _paragraphs_from_fragment(article_match.group(1))
        if article_text:
            return article_text

    return _paragraphs_from_fragment(working)


def _paragraphs_from_fragment(fragment: str) -> str | None:
    paragraphs = re.findall(
        r"<p\b[^>]*>(.*?)</p>",
        fragment,
        flags=re.IGNORECASE | re.DOTALL,
    )
    cleaned_paragraphs: list[str] = []
    seen: set[str] = set()
    for paragraph in paragraphs:
        text = _clean_text(paragraph)
        if len(text) < 60:
            continue
        if text in seen:
            continue
        seen.add(text)
        cleaned_paragraphs.append(text)

    if not cleaned_paragraphs:
        return None
    return "\n".join(cleaned_paragraphs[:12])


def _choose_best_article_text(
    article_body: str | None,
    paragraph_text: str | None,
    *,
    char_limit: int,
) -> str | None:
    candidates = [candidate for candidate in (article_body, paragraph_text) if candidate]
    if not candidates:
        return None
    best = max(candidates, key=len)
    return _truncate_text(best, char_limit)


def _build_excerpt(
    full_text: str | None,
    *,
    fallback: str | None,
    char_limit: int,
) -> str | None:
    if full_text:
        excerpt_source = full_text.split("\n", maxsplit=1)[0]
        if len(excerpt_source) < 120:
            excerpt_source = full_text
        return _truncate_text(excerpt_source, char_limit)

    if fallback:
        return _truncate_text(_clean_text(fallback), char_limit)
    return None


def _extract_meta_content(html_text: str, attr_name: str, attr_value: str) -> str | None:
    pattern = re.compile(
        rf"<meta[^>]+{attr_name}=[\"']{re.escape(attr_value)}[\"'][^>]+content=[\"'](.*?)[\"'][^>]*>",
        re.IGNORECASE | re.DOTALL,
    )
    match = pattern.search(html_text)
    if not match:
        return None
    return _clean_text(match.group(1))


def _extract_title_tag(html_text: str) -> str | None:
    match = re.search(r"<title[^>]*>(.*?)</title>", html_text, flags=re.IGNORECASE | re.DOTALL)
    if not match:
        return None
    return _clean_text(match.group(1))


def _clean_text(value: str) -> str:
    without_tags = re.sub(r"<[^>]+>", " ", value)
    unescaped = html.unescape(without_tags)
    return " ".join(unescaped.split())


def _truncate_text(value: str, char_limit: int) -> str:
    normalized = " ".join(value.split())
    if len(normalized) <= char_limit:
        return normalized
    return normalized[: char_limit - 3].rstrip() + "..."


def _sanitize_article_text(value: str | None) -> str | None:
    if not value:
        return value

    cleaned = value.strip()
    for pattern in LEADING_BOILERPLATE_PATTERNS:
        cleaned = pattern.sub("", cleaned)
    cleaned = cleaned.replace("(예시) ", "").replace("(예시)", "")
    cleaned = " ".join(cleaned.split())
    return cleaned or None


def decode_google_news_url(
    source_url: str,
    *,
    client: httpx.Client | None = None,
) -> str | None:
    parsed = urlparse(source_url)
    path_parts = parsed.path.split("/")
    if parsed.hostname != "news.google.com" or len(path_parts) < 3:
        return source_url
    if path_parts[-2] not in {"articles", "read"}:
        return source_url

    encoded_id = path_parts[-1]
    try:
        decoded_bytes = base64.urlsafe_b64decode(encoded_id + "==")
        decoded_str = decoded_bytes.decode("latin1")
    except Exception:
        return source_url

    prefix = b"\x08\x13\x22".decode("latin1")
    if decoded_str.startswith(prefix):
        decoded_str = decoded_str[len(prefix) :]

    suffix = b"\xd2\x01\x00".decode("latin1")
    if decoded_str.endswith(suffix):
        decoded_str = decoded_str[: -len(suffix)]

    bytes_array = bytearray(decoded_str, "latin1")
    if not bytes_array:
        return source_url

    length = bytes_array[0]
    if length >= 0x80 and len(decoded_str) > 2:
        decoded_candidate = decoded_str[2 : length + 1]
    else:
        decoded_candidate = decoded_str[1 : length + 1]

    if decoded_candidate.startswith("AU_yqL"):
        resolved_client = client or _build_http_client(timeout_seconds=10)
        should_close_client = client is None
        try:
            decode_params = _fetch_google_news_decode_params(
                source_url,
                encoded_id=encoded_id,
                client=resolved_client,
            )
            if decode_params is None:
                return source_url
            return _decode_google_news_batch_url(
                decode_params,
                client=resolved_client,
            ) or source_url
        finally:
            if should_close_client:
                resolved_client.close()
    if decoded_candidate.startswith("http://") or decoded_candidate.startswith("https://"):
        return decoded_candidate
    return source_url


def _build_http_client(*, timeout_seconds: int) -> httpx.Client:
    return httpx.Client(
        follow_redirects=True,
        timeout=timeout_seconds,
        headers={
            "User-Agent": USER_AGENT,
            "Accept": "text/html,application/xhtml+xml",
        },
    )


def _fetch_google_news_decode_params(
    source_url: str,
    *,
    encoded_id: str,
    client: httpx.Client,
) -> GoogleNewsDecodeParams | None:
    try:
        response = client.get(source_url)
        response.raise_for_status()
    except Exception as exc:  # pragma: no cover - depends on live network/google
        logger.warning("Google News wrapper fetch failed for %s: %s", encoded_id, exc)
        return None

    return _extract_google_news_decode_params(
        response.text,
        encoded_id=encoded_id,
    )


def _extract_google_news_decode_params(
    html_text: str,
    *,
    encoded_id: str,
) -> GoogleNewsDecodeParams | None:
    timestamp_match = re.search(r'data-n-a-ts="([^"]+)"', html_text)
    signature_match = re.search(r'data-n-a-sg="([^"]+)"', html_text)
    if not timestamp_match or not signature_match:
        return None

    return GoogleNewsDecodeParams(
        article_id=encoded_id,
        timestamp=timestamp_match.group(1),
        signature=signature_match.group(1),
    )


def _decode_google_news_batch_url(
    decode_params: GoogleNewsDecodeParams,
    *,
    client: httpx.Client,
) -> str | None:
    articles_request = [
        [
            "Fbv4je",
            (
                '["garturlreq",'
                '[["X","X",["X","X"],null,null,1,1,"US:en",null,1,null,null,null,null,null,0,1],'
                '"X","X",1,[1,1,1],1,1,null,0,0,null,0],'
                f'"{decode_params.article_id}",{decode_params.timestamp},"{decode_params.signature}"]'
            ),
        ]
    ]
    payload = "f.req=" + quote(json.dumps([articles_request]))
    try:
        response = client.post(
            "https://news.google.com/_/DotsSplashUi/data/batchexecute",
            headers={
                "Content-Type": "application/x-www-form-urlencoded;charset=UTF-8",
                "Referer": "https://news.google.com/",
            },
            content=payload,
        )
        response.raise_for_status()
    except Exception as exc:  # pragma: no cover - depends on live network/google endpoint
        logger.warning(
            "Google News URL decode request failed for %s: %s",
            decode_params.article_id,
            exc,
        )
        return None

    return _parse_google_news_batch_response(response.text)


def _parse_google_news_batch_response(response_text: str) -> str | None:
    parts = response_text.split("\n\n", maxsplit=1)
    if len(parts) != 2:
        return None

    try:
        payload = json.loads(parts[1])
    except json.JSONDecodeError:
        return None

    for item in payload:
        if not isinstance(item, list) or len(item) < 3:
            continue
        if item[1] != "Fbv4je" or not isinstance(item[2], str):
            continue
        try:
            inner_payload = json.loads(item[2])
        except json.JSONDecodeError:
            continue
        if (
            isinstance(inner_payload, list)
            and len(inner_payload) >= 2
            and inner_payload[0] == "garturlres"
            and isinstance(inner_payload[1], str)
        ):
            return html.unescape(inner_payload[1])

    return None


def _infer_publisher_name_from_url(source_url: str | None) -> str | None:
    if not source_url:
        return None
    parsed = urlparse(source_url)
    hostname = (parsed.hostname or "").lower()
    if not hostname:
        return None
    if hostname.startswith("www."):
        hostname = hostname[4:]
    if hostname.startswith("m."):
        hostname = hostname[2:]
    if hostname.endswith(".co.kr"):
        hostname = hostname[: -len(".co.kr")]
    elif "." in hostname:
        hostname = hostname.rsplit(".", maxsplit=1)[0]

    label = hostname.split(".")[-1]
    if not label:
        return None
    if label.isupper():
        return label
    return label.replace("-", " ").strip().title()


def _normalize_publisher_name(value: str | None) -> str | None:
    cleaned = (value or "").strip()
    if not cleaned:
        return None

    for prefix in ("Daum | ", "다음 | "):
        if cleaned.startswith(prefix):
            cleaned = cleaned[len(prefix) :].strip()
            break

    return cleaned or None
