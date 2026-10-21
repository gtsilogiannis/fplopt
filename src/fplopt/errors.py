"""Exceptions of the optimizer. The library raises them; only the command line (cli.py)
turns them into messages and an exit status, so the engine can also be used from other
Python code (tests, backtests, a web server) without the process ending.

FplError
 +-- InputError        an argument is invalid
 +-- ApiError          the FPL API could not be reached or answered with an error
 |    +-- NotFoundError   404: unknown team id or gameweek
 +-- SolverError       the solver found no plan
 +-- UserActionError   the run cannot continue without the user (e.g. pass --ft)
 +-- DataError         a local file needed by the run is missing
"""


class FplError(Exception):
    """Base class: every error the optimizer reports on purpose."""


class InputError(FplError):
    """Invalid command-line argument or option value."""


class ApiError(FplError):
    """Network failure, timeout or HTTP error from the FPL API."""


class NotFoundError(ApiError):
    """HTTP 404 from the FPL API: the team id or gameweek does not exist."""


class SolverError(FplError):
    """The MILP solver returned no usable plan."""


class UserActionError(FplError):
    """The run needs a decision or a fix by the user (the message says which)."""


class DataError(FplError):
    """A local data file needed by the run is missing or unreadable."""
