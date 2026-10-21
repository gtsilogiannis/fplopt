"""Exceptions of the optimizer. The library raises them; only the command line (cli.py)
turns them into messages and an exit status, so the engine can also be used from other
Python code (tests, backtests, a web server) without the process ending.

Each class carries the exit status of the command line and a short machine-readable kind
(the 'error.type' field of the JSON output):

  exit  kind         class
  0     -            (success; a plan stopped by the time limit is still a success)
  1     internal     any other exception (a bug)
  2     input        InputError        an argument is invalid
  3     network      ApiError          FPL API unreachable, timeout, HTTP 5xx/429 after retries, bad reply
  4     not_found    NotFoundError     HTTP 404: unknown team id or gameweek
  5     solver       SolverError       the solver found no plan
  6     user_action  UserActionError   the run needs the user (e.g. --ft, a missing package)
  7     data         DataError         a local file needed by the run is missing
"""


class FplError(Exception):
    """Base class: every error the optimizer reports on purpose."""
    exit_code = 1
    kind = 'internal'


class InputError(FplError):
    """Invalid command-line argument or option value."""
    exit_code = 2
    kind = 'input'


class ApiError(FplError):
    """Network failure, timeout or HTTP error from the FPL API."""
    exit_code = 3
    kind = 'network'


class NotFoundError(ApiError):
    """HTTP 404 from the FPL API: the team id or gameweek does not exist."""
    exit_code = 4
    kind = 'not_found'


class SolverError(FplError):
    """The MILP solver returned no usable plan."""
    exit_code = 5
    kind = 'solver'


class UserActionError(FplError):
    """The run needs a decision or a fix by the user (the message says which)."""
    exit_code = 6
    kind = 'user_action'


class DataError(FplError):
    """A local data file needed by the run is missing or unreadable."""
    exit_code = 7
    kind = 'data'


EXIT_INTERNAL = 1        # unexpected exception (not an FplError)
EXIT_INTERRUPTED = 130   # Ctrl+C (128 + SIGINT, the shell convention)
