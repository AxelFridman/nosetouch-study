"""ASGI entry point: `uvicorn app:app` (see Procfile). Configuration comes from the environment."""
from nosetouch_server.main import create_app

app = create_app()
