"""JSON logs for every ggrpo event, to stderr (read live with hl in tmux) and appended to the configured log file."""
import logging

import structlog

from ggrpo.settings import Settings

settings = Settings()
settings.log.file.parent.mkdir(parents=True, exist_ok=True)
logger = logging.getLogger("ggrpo")   # its own handlers, so host processes such as trl keep their logging untouched
logger.handlers = [logging.StreamHandler(), logging.FileHandler(settings.log.file)]
logger.setLevel(logging.INFO)
logger.propagate = False
structlog.configure(
    processors=[
        structlog.processors.CallsiteParameterAdder([structlog.processors.CallsiteParameter.MODULE]),
        structlog.stdlib.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.EventRenamer("message"),
        structlog.processors.JSONRenderer(),
    ],
    logger_factory=structlog.stdlib.LoggerFactory(),
)
