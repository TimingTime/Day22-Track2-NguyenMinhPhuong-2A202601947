"""Cấu hình output UTF-8 ổn định cho terminal Windows và Unix."""

import sys


def configure_utf8_output() -> None:
    """Tránh UnicodeEncodeError khi terminal Windows mặc định dùng CP1252."""
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            reconfigure(encoding="utf-8", errors="replace")
