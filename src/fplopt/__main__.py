"""python -m fplopt  ->  the command line (cli.main); its return value is the exit status."""
from .cli import main

raise SystemExit(main())
