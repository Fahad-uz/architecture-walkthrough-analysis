from __future__ import annotations

import logging
import re


class SanitizingFormatter(logging.Formatter):
    _token_pattern = re.compile(r"(?i)(token|password|secret|api[_-]?key)=([^\\s]+)")

    def format(self, record: logging.LogRecord) -> str:
        rendered = super().format(record)
        return self._token_pattern.sub(r"\1=<redacted>", rendered)


def configure_logging(level: int = logging.INFO) -> None:
    handler = logging.StreamHandler()
    handler.setFormatter(SanitizingFormatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
    root = logging.getLogger()
    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(level)
