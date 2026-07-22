"""FastAPI/WebSocket OpenEnv server."""

from openenv.core.env_server.http_server import create_app

from ..models import FounderAction, FounderObservation
from .environment import FounderDecisionEnvironment

app = create_app(
    FounderDecisionEnvironment,
    FounderAction,
    FounderObservation,
    env_name="yc_founder_decision_env",
    max_concurrent_envs=8,
)


def main() -> None:
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)


if __name__ == "__main__":
    main()
