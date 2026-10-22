"""fplopt: Fantasy Premier League squad optimizer.

A multi-period mixed-integer linear program (PuLP 4, HiGHS) chooses transfers, starting
XI, captain, vice-captain and bench order over several gameweeks, on top of an expected-points
model that is backtested on past seasons (scripts/calibrate_xp.py).

Modules:
  xp_model     expected points (pure functions, self-contained)
  rules        FPL transfer and chip rules
  api          HTTP access to the public FPL API
  data         fixtures and per-gameweek history from the API
  squad        current squad, selling prices, free transfers, available chips
  milp         the MILP and the solver interface
  chips        chip opportunity costs ("Chip suggestions")
  engine       one optimizer run -> OptimizeResult (data only, no printing)
  predictions  prediction log and its evaluation against actual points
  report       text output
  cli          command line (python -m fplopt / fplopt)
"""
import logging

__version__ = "0.9.0"

# A library logs, the application decides where the messages go (cli.py sends them to stderr).
logging.getLogger(__name__).addHandler(logging.NullHandler())
