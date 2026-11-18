"""Entry point of python -m fplopt: the command line (cli.main), whose return value is the exit status."""
from .cli import main

raise SystemExit(main())
