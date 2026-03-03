"""
tracker/database.py — SQLite persistence via SQLAlchemy.

Stores:
  - Query history (question, code, result summary, score, timestamp)
  - Successful code patterns for self-improvement memory
  - Session metadata

Public API
----------
TrackerDB(db_url: str = "sqlite:///tracker/crm_agent.db")
    .log_query(question, code, result_summary, score, iterations)
    .get_recent_queries(n: int) -> list[dict]
    .log_pattern(question_type, code_pattern, score)
    .get_patterns(question_type: str) -> list[dict]
    .log_session(session_id, file_names) -> None
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path

from sqlalchemy import (
    Column, DateTime, Float, Integer, String, Text,
    create_engine, text,
)
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

logger = logging.getLogger(__name__)


# ── ORM Models ────────────────────────────────────────────────────────────────

class Base(DeclarativeBase):
    pass


class QueryLog(Base):
    __tablename__ = "query_log"

    id = Column(Integer, primary_key=True, autoincrement=True)
    timestamp = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    question = Column(Text, nullable=False)
    code = Column(Text)
    result_summary = Column(Text)
    score = Column(Float)
    iterations = Column(Integer, default=1)
    error = Column(Text)


class CodePattern(Base):
    __tablename__ = "code_patterns"

    id = Column(Integer, primary_key=True, autoincrement=True)
    timestamp = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    question_type = Column(String(200))
    code_pattern = Column(Text, nullable=False)
    score = Column(Float)
    use_count = Column(Integer, default=1)


class SessionLog(Base):
    __tablename__ = "session_log"

    id = Column(Integer, primary_key=True, autoincrement=True)
    session_id = Column(String(64), unique=True)
    started_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    file_names = Column(Text)   # JSON array


# ── DB class ──────────────────────────────────────────────────────────────────

class TrackerDB:
    """Wraps SQLAlchemy session for all persistence operations."""

    DEFAULT_URL = "sqlite:///tracker/crm_agent.db"

    def __init__(self, db_url: str | None = None) -> None:
        Path("tracker").mkdir(exist_ok=True)
        url = db_url or self.DEFAULT_URL
        self.engine = create_engine(url, echo=False)
        Base.metadata.create_all(self.engine)
        self._Session = sessionmaker(bind=self.engine)
        logger.info("TrackerDB initialised: %s", url)

    # ── Query logging ─────────────────────────────────────────────────────────

    def log_query(
        self,
        question: str,
        code: str,
        result_summary: str = "",
        score: float | None = None,
        iterations: int = 1,
        error: str | None = None,
    ) -> None:
        with self._Session() as session:
            session.add(QueryLog(
                question=question,
                code=code,
                result_summary=result_summary[:500] if result_summary else "",
                score=score,
                iterations=iterations,
                error=error,
            ))
            session.commit()

    def get_recent_queries(self, n: int = 20) -> list[dict]:
        with self._Session() as session:
            rows = (
                session.query(QueryLog)
                .order_by(QueryLog.timestamp.desc())
                .limit(n)
                .all()
            )
            return [
                {
                    "id": r.id,
                    "timestamp": str(r.timestamp),
                    "question": r.question,
                    "code": r.code,
                    "score": r.score,
                    "iterations": r.iterations,
                    "error": r.error,
                }
                for r in rows
            ]

    # ── Pattern memory ────────────────────────────────────────────────────────

    def log_pattern(self, question_type: str, code_pattern: str, score: float) -> None:
        with self._Session() as session:
            existing = (
                session.query(CodePattern)
                .filter_by(question_type=question_type)
                .first()
            )
            if existing:
                existing.use_count += 1
                existing.score = max(existing.score or 0, score)
            else:
                session.add(CodePattern(
                    question_type=question_type,
                    code_pattern=code_pattern,
                    score=score,
                ))
            session.commit()

    def get_patterns(self, question_type: str | None = None) -> list[dict]:
        with self._Session() as session:
            q = session.query(CodePattern)
            if question_type:
                q = q.filter(CodePattern.question_type.ilike(f"%{question_type}%"))
            rows = q.order_by(CodePattern.score.desc()).all()
            return [
                {
                    "question_type": r.question_type,
                    "code_pattern": r.code_pattern,
                    "score": r.score,
                    "use_count": r.use_count,
                }
                for r in rows
            ]

    # ── Session logging ───────────────────────────────────────────────────────

    def log_session(self, session_id: str, file_names: list[str]) -> None:
        import json
        with self._Session() as session:
            session.add(SessionLog(
                session_id=session_id,
                file_names=json.dumps(file_names),
            ))
            session.commit()
