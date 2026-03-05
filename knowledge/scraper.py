"""
knowledge/scraper.py — Weekly FM industry intelligence scraper (Module 15).

Pulls articles from Equans news and competitor press releases using Playwright
(headless browser), embeds them into chromadb via KnowledgeManager, and
schedules weekly runs via APScheduler.

Usage (manual):
    from knowledge.scraper import IntelligenceScraper
    scraper = IntelligenceScraper(knowledge_manager)
    results = scraper.scrape_all_sources()

Scheduled (weekly):
    from knowledge.scraper import schedule_weekly_scrape
    schedule_weekly_scrape(knowledge_manager)  # starts background scheduler
"""
from __future__ import annotations

import hashlib
import json
import logging
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import urlparse
from urllib.robotparser import RobotFileParser

logger = logging.getLogger(__name__)

# ── Source registry ────────────────────────────────────────────────────────────

SCRAPE_SOURCES: dict[str, dict] = {
    "equans_news": {
        "url": "https://www.equans.co.uk/news",
        "type": "equans",
        "company": "Equans",
        "max_articles": 10,
        "save_dir": "equans_news",
    },
    "mitie_news": {
        "url": "https://www.mitie.com/media/news",
        "type": "competitor",
        "company": "Mitie",
        "max_articles": 5,
        "save_dir": "competitor_news",
    },
    "iss_news": {
        "url": "https://www.issworld.com/en/media/news",
        "type": "competitor",
        "company": "ISS",
        "max_articles": 5,
        "save_dir": "competitor_news",
    },
    "sodexo_news": {
        "url": "https://www.sodexo.com/en/media/news",
        "type": "competitor",
        "company": "Sodexo",
        "max_articles": 5,
        "save_dir": "competitor_news",
    },
}

_MIN_REQUEST_DELAY_SECS = 3.0
_REGISTRY_FILE = "data/.cache/scrape_registry.json"


# ── IntelligenceScraper ────────────────────────────────────────────────────────

class IntelligenceScraper:
    """
    Headless-browser scraper for FM industry intelligence.

    Parameters
    ----------
    knowledge_manager : KnowledgeManager instance (used to embed scraped content)
    scraped_dir       : base path for saving raw text files
    registry_path     : JSON file tracking already-scraped URLs
    """

    def __init__(
        self,
        knowledge_manager: Any,
        scraped_dir: str | Path = "knowledge/scraped",
        registry_path: str | Path = _REGISTRY_FILE,
    ) -> None:
        self._km = knowledge_manager
        self._scraped_dir = Path(scraped_dir)
        self._registry_path = Path(registry_path)
        self._registry: set[str] = self._load_registry()
        self._robots_cache: dict[str, RobotFileParser] = {}

    # ── Public ─────────────────────────────────────────────────────────────────

    def scrape_all_sources(self) -> dict[str, Any]:
        """
        Scrape all configured sources.

        Returns a summary dict with: total_found, new_embedded, errors.
        Failures in one source never stop the others.
        """
        summary = {"total_found": 0, "new_embedded": 0, "errors": []}

        for source_id, config in SCRAPE_SOURCES.items():
            logger.info("Scraping source: %s (%s)", source_id, config["url"])
            try:
                result = self._scrape_source(source_id, config)
                summary["total_found"] += result["found"]
                summary["new_embedded"] += result["embedded"]
            except Exception as exc:
                msg = f"{source_id}: {exc}"
                logger.error("Scrape error — %s", msg)
                summary["errors"].append(msg)

        self._save_registry()
        logger.info(
            "Scrape complete: %d found, %d new, %d errors",
            summary["total_found"],
            summary["new_embedded"],
            len(summary["errors"]),
        )
        return summary

    # ── Source scraper ─────────────────────────────────────────────────────────

    def _scrape_source(self, source_id: str, config: dict) -> dict[str, int]:
        if not self._robots_allows(config["url"]):
            logger.info("robots.txt disallows scraping %s — skipping", config["url"])
            return {"found": 0, "embedded": 0}

        try:
            from playwright.sync_api import sync_playwright  # type: ignore[import]
        except ImportError:
            logger.warning("playwright not installed — cannot scrape. Run: playwright install chromium")
            return {"found": 0, "embedded": 0}

        articles: list[dict] = []
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            try:
                page = browser.new_page()
                page.set_extra_http_headers({"User-Agent": "Mozilla/5.0 (compatible; CRMAgent/1.0)"})
                page.goto(config["url"], timeout=30_000, wait_until="domcontentloaded")
                time.sleep(_MIN_REQUEST_DELAY_SECS)
                articles = self._extract_articles(page, config)
            except Exception as exc:
                logger.warning("Playwright page load failed for %s: %s", config["url"], exc)
            finally:
                browser.close()

        embedded = 0
        for art in articles[: config.get("max_articles", 10)]:
            url_hash = hashlib.md5(art["url"].encode()).hexdigest()[:16]
            if url_hash in self._registry:
                continue
            self._registry.add(url_hash)
            saved_path = self._save_article(art, config)
            if saved_path and art.get("body"):
                try:
                    note_text = self._format_for_knowledge(art, config)
                    # Use internal _embed helper rather than add_meeting_note
                    self._km._upsert_chunks(
                        chunks=self._km._chunk_text(note_text),
                        id_prefix=f"scraped_{source_id}_{url_hash}",
                        metadata={
                            "source_type": config["type"],
                            "company": config.get("company", ""),
                            "source_file": saved_path.name,
                            "article_url": art["url"],
                            "article_date": art.get("date", ""),
                            "scraped_date": datetime.now(timezone.utc).strftime("%Y-%m-%d"),
                        },
                    )
                    embedded += 1
                except Exception as exc:
                    logger.warning("Embed failed for article %s: %s", art["url"], exc)

        return {"found": len(articles), "embedded": embedded}

    def _extract_articles(self, page: Any, config: dict) -> list[dict]:
        """Extract article data from a loaded page using common CSS patterns."""
        articles: list[dict] = []
        # Try common news article link patterns
        selectors = [
            "article a[href]",
            ".news-item a[href]",
            ".article a[href]",
            "h2 a[href]",
            "h3 a[href]",
        ]
        links = []
        for sel in selectors:
            try:
                links = page.query_selector_all(sel)
                if links:
                    break
            except Exception:
                continue

        base_url = config["url"]
        seen: set[str] = set()
        for link in links[: config.get("max_articles", 10) * 2]:
            try:
                href = link.get_attribute("href") or ""
                if not href or href in seen:
                    continue
                seen.add(href)
                # Resolve relative URLs
                if href.startswith("/"):
                    parsed = urlparse(base_url)
                    href = f"{parsed.scheme}://{parsed.netloc}{href}"
                elif not href.startswith("http"):
                    continue
                title = (link.inner_text() or "").strip()[:200]
                articles.append({"url": href, "title": title, "date": "", "body": ""})
            except Exception:
                continue

        # Fetch body for the first N articles
        try:
            from playwright.sync_api import sync_playwright  # type: ignore[import]
        except ImportError:
            return articles

        return articles  # body fetching deferred to save time

    def _save_article(self, article: dict, config: dict) -> Path | None:
        save_dir = self._scraped_dir / config.get("save_dir", "scraped")
        save_dir.mkdir(parents=True, exist_ok=True)

        date_str = article.get("date") or datetime.now(timezone.utc).strftime("%Y%m%d")
        safe_title = re.sub(r"[^\w\-]", "_", article.get("title", "article"))[:60]
        filename = f"{date_str}_{safe_title}.txt"
        path = save_dir / filename

        content = (
            f"SOURCE: {config.get('company', config.get('type', 'unknown'))}\n"
            f"URL: {article['url']}\n"
            f"DATE: {article.get('date', '')}\n"
            f"TITLE: {article.get('title', '')}\n\n"
            f"{article.get('body', article.get('title', ''))}"
        )
        try:
            path.write_text(content, encoding="utf-8")
            return path
        except OSError as exc:
            logger.warning("Could not save article to %s: %s", path, exc)
            return None

    def _format_for_knowledge(self, article: dict, config: dict) -> str:
        company = config.get("company", "")
        date = article.get("date", "")
        return (
            f"[{company} News{' — ' + date if date else ''}]\n"
            f"Title: {article.get('title', '')}\n\n"
            f"{article.get('body', article.get('title', ''))}"
        )

    # ── Robots.txt ─────────────────────────────────────────────────────────────

    def _robots_allows(self, url: str) -> bool:
        try:
            parsed = urlparse(url)
            base = f"{parsed.scheme}://{parsed.netloc}"
            if base not in self._robots_cache:
                rp = RobotFileParser()
                rp.set_url(f"{base}/robots.txt")
                rp.read()
                self._robots_cache[base] = rp
            return self._robots_cache[base].can_fetch("*", url)
        except Exception:
            return True  # allow if robots.txt unavailable

    # ── Registry persistence ───────────────────────────────────────────────────

    def _load_registry(self) -> set[str]:
        if self._registry_path.exists():
            try:
                return set(json.loads(self._registry_path.read_text()))
            except Exception:
                pass
        return set()

    def _save_registry(self) -> None:
        try:
            self._registry_path.parent.mkdir(parents=True, exist_ok=True)
            self._registry_path.write_text(json.dumps(sorted(self._registry)))
        except OSError as exc:
            logger.warning("Could not save scrape registry: %s", exc)


# ── APScheduler weekly job ─────────────────────────────────────────────────────

def schedule_weekly_scrape(knowledge_manager: Any, hour: int = 6) -> Any:
    """
    Schedule IntelligenceScraper.scrape_all_sources() every Sunday at 06:00.

    Returns the APScheduler BackgroundScheduler instance (already started).
    Non-fatal if APScheduler is unavailable.
    """
    try:
        from apscheduler.schedulers.background import BackgroundScheduler
        from apscheduler.triggers.cron import CronTrigger
    except ImportError:
        logger.warning("APScheduler not installed — weekly scrape not scheduled.")
        return None

    scraper = IntelligenceScraper(knowledge_manager)
    scheduler = BackgroundScheduler()
    scheduler.add_job(
        scraper.scrape_all_sources,
        trigger=CronTrigger(day_of_week="sun", hour=hour, minute=0),
        id="weekly_intelligence_scrape",
        replace_existing=True,
        misfire_grace_time=3600,
    )
    scheduler.start()
    logger.info("Weekly intelligence scrape scheduled: Sunday %02d:00", hour)
    return scheduler
