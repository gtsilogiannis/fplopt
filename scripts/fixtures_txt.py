"""Parser for openfootball's Football.TXT fixture files (public domain, CC0).

Only what the congestion study needs: the calendar date and the two team names of every match.
Format: one block per round, a date line (weekday, month, day, optional year), then its match
lines; a match line without a time has the date of the line above.
    ▪ Round 3
      Sat Jan 11 2025
        12:15  Team A  v Team B  2-1 (1-0)
               Team C  v Team D  0-0
"""
import re
from datetime import date

MONTHS = {m: i for i, m in enumerate(
    ('Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'), start=1)}
# Date line, e.g. "  Sat Jan 11 2025" or "  Sun Jan 12" (the year is given only when it changes)
DATE_RE = re.compile(r'^\s*(Mon|Tue|Wed|Thu|Fri|Sat|Sun)\s+([A-Z][a-z]{2})/?\s*(\d{1,2})(?:\s+(\d{4}))?\s*$')
# Match line: optional "HH:MM", then "home v away", then the rest (score, venue, notes)
MATCH_RE = re.compile(r'^\s*(?:\d{1,2}[:.]\d{2}\s+)?(.+?)\s+v\s+(.+?)(?:\s{2,}|\s+\d+-\d+|\s+@|\s*$)')
HEADER_YEAR_RE = re.compile(r'^#\s*Date\s+\w{3}\s+\w{3}\s+\d{1,2}\s+(\d{4})')


def parse(path):
    """List of (date, home, away) in file order. Raises ValueError on a match line before any date."""
    out = []
    year = None   # current year: from the '# Date' header, then from any date line that states it
    month_prev = None
    cur = None    # current match date
    with open(path, encoding='utf-8') as f:   # explicit UTF-8: names like 'Bodø/Glimt'
        for line in f:
            line = line.rstrip('\n')
            if not line.strip() or line.lstrip().startswith(('=', '▪', '»')):
                continue
            if line.startswith('#'):
                m = HEADER_YEAR_RE.match(line)
                if m and year is None:
                    year = int(m.group(1))
                continue
            m = DATE_RE.match(line)
            if m:
                mon, day = MONTHS[m.group(2)], int(m.group(3))
                if m.group(4):
                    year = int(m.group(4))
                elif month_prev is not None and mon < month_prev:
                    year += 1          # no year given and the month went back (December to January)
                month_prev = mon
                cur = date(year, mon, day)
                continue
            m = MATCH_RE.match(line)
            if m and ' v ' in line:
                if cur is None:
                    raise ValueError(f'{path}: match line before any date: {line!r}')
                out.append((cur, m.group(1).strip(), m.group(2).strip()))
    return out
