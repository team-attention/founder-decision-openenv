"""OpenEnv local-validator wrapper for the src-layout server."""

from yc_founder_decision_env.server.app import app
from yc_founder_decision_env.server.app import main as _main

__all__ = ["app", "main"]


def main() -> None:
    _main()


if __name__ == "__main__":
    main()
