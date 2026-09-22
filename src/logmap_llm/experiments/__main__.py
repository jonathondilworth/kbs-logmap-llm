"""Allow ``python -m logmap_llm.experiments`` without package installation."""

from logmap_llm.experiments.cli import main


raise SystemExit(main())
