"""Structured database logging utility for the telemetry processing pipeline.

Format:  TAG: CODE: MESSAGE
Example: INFO: 200: Successful query of contact abc-123
"""
import logging
import psycopg2
from psycopg2.extras import Json
from typing import Optional, Dict, Any
from lib.settings import database_config, env_flag

# Module-level console logger
_console_logger = logging.getLogger("db_logger")
if not _console_logger.handlers:
    _handler = logging.StreamHandler()
    _handler.setFormatter(logging.Formatter(
        "%(asctime)s - %(levelname)s - %(message)s"
    ))
    _console_logger.addHandler(_handler)
    _console_logger.setLevel(logging.DEBUG)


def log_event(
    tag: str,
    code: int,
    message: str,
    source: str = "unknown",
    contact_id: Optional[str] = None,
    spacecraft_id: Optional[str] = None,
    extra: Optional[Dict[str, Any]] = None,
    conn: Optional[Any] = None,
) -> None:
    """
    Writes a structured log event to both the console and the database.
    
    Parameters
    ----------
    tag : str
        Log level tag (INFO, DEBUG, WARNING, ERROR).
    code : int
        Application-specific status code (e.g. 200, 404, 500).
    message : str
        Human-readable message.
    source : str
        Module or function name that emitted the log.
    contact_id : str, optional
        Contact UUID for context.
    spacecraft_id : str, optional
        Spacecraft UUID for context.
    extra : dict, optional
        Arbitrary JSON-serializable metadata.
    conn : psycopg2 connection, optional
        Existing DB connection to reuse (avoids opening a new one).
    """
    # --- Console output ---
    level = getattr(logging, tag.upper(), logging.INFO)
    console_msg = f"{tag}: {code}: {message}"
    _console_logger.log(level, f"[{source}] {console_msg}")

    if not env_flag("DB_LOGGING_ENABLED", default=True):
        return

    # --- Database output ---
    should_close = conn is None
    cursor = None
    try:
        if conn is None:
            conn = psycopg2.connect(**database_config())
        cursor = conn.cursor()
        cursor.execute(
            """
            INSERT INTO processing_logs 
                (timestamp, tag, code, message, source, contact_id, spacecraft_id, metadata_json)
            VALUES 
                (NOW(), %s, %s, %s, %s, %s, %s, %s)
            """,
            (
                tag,
                str(code),
                message,
                source,
                contact_id,
                spacecraft_id,
                Json(extra) if extra else None,
            ),
        )
        conn.commit()
    except Exception as exc:
        # Never let logging failures break the pipeline
        _console_logger.error("DB log write failed: %s", exc)
    finally:
        if cursor is not None:
            cursor.close()
        if conn is not None and should_close:
            conn.close()


class ProcessLogger:
    """
    Convenience logger that binds source/contact/spacecraft context.
    
    Usage
    -----
    logger = ProcessLogger("processor", contact_id="abc", spacecraft_id="xyz")
    logger.info(200, "Query successful", extra={"rows": 42})
    """
    def __init__(
        self,
        source: str,
        contact_id: Optional[str] = None,
        spacecraft_id: Optional[str] = None,
    ):
        self.source = source
        self.contact_id = contact_id
        self.spacecraft_id = spacecraft_id

    def log(
        self,
        tag: str,
        code: int,
        message: str,
        extra: Optional[Dict[str, Any]] = None,
    ):
        log_event(
            tag, code, message, self.source,
            self.contact_id, self.spacecraft_id, extra,
        )

    def info(self, code: int, message: str, extra: Optional[Dict[str, Any]] = None):
        self.log("INFO", code, message, extra)

    def debug(self, code: int, message: str, extra: Optional[Dict[str, Any]] = None):
        self.log("DEBUG", code, message, extra)

    def warning(self, code: int, message: str, extra: Optional[Dict[str, Any]] = None):
        self.log("WARNING", code, message, extra)

    def error(self, code: int, message: str, extra: Optional[Dict[str, Any]] = None):
        self.log("ERROR", code, message, extra)

    def bind(
        self,
        contact_id: Optional[str] = None,
        spacecraft_id: Optional[str] = None,
    ) -> "ProcessLogger":
        """Return a new logger with updated context."""
        return ProcessLogger(
            self.source,
            contact_id=contact_id or self.contact_id,
            spacecraft_id=spacecraft_id or self.spacecraft_id,
        )
