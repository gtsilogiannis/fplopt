"""Collecting the warnings of a run for the machine-readable output (--json).

Every module logs its messages to the 'fplopt' logger as they happen; the command line prints
them on stderr. collect() additionally keeps the warnings (and every message logged with a
'code', e.g. log.info(..., extra={'code': 'time_limit'})) as a list of {'code', 'message'}."""
import logging
from contextlib import contextmanager

PREFIXES = ('[WARNING] ', '[INFO] ', '[*] ')


class MessageCollector(logging.Handler):
    """A logging handler that stores the messages instead of printing them."""

    def __init__(self):
        super().__init__(level=logging.DEBUG)
        self.items = []

    def emit(self, record):
        code = getattr(record, 'code', None)
        if logging.WARNING <= record.levelno < logging.ERROR or code:   # errors are reported separately
            msg = record.getMessage()
            for prefix in PREFIXES:                 # the JSON carries the code, not the text prefix
                if msg.startswith(prefix):
                    msg = msg[len(prefix):]
            self.items.append({'code': code or 'warning', 'message': msg})


@contextmanager
def collect():
    """with collect() as items: ...  -> items = the warnings logged inside the block."""
    handler = MessageCollector()
    package_log = logging.getLogger('fplopt')
    package_log.addHandler(handler)
    try:
        yield handler.items
    finally:
        package_log.removeHandler(handler)
