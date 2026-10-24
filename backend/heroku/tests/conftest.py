"""Fixtures of the upload-service tests: FastAPI TestClient on in-memory SQLite.

Set NOSETOUCH_TEST_DATABASE_URL to a throwaway Postgres database whose name contains 'test' to run every
test a second time against Postgres (its tables are dropped and recreated per test).
"""
from __future__ import annotations

import os
import sys
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

import pytest

HERE = Path(__file__).resolve().parent
SERVICE_DIR = HERE.parent
REPO = SERVICE_DIR.parents[1]
for p in (str(SERVICE_DIR), str(HERE)):
    if p not in sys.path:
        sys.path.insert(0, p)

from fastapi.testclient import TestClient  # noqa: E402

from nosetouch_server.main import create_app  # noqa: E402
from nosetouch_server.settings import Settings, normalize_database_url  # noqa: E402

PAGES_ORIGIN = 'https://axelfridman.github.io'
ADMIN_TOKEN = 'test-admin-token-0123456789'
REAL_EXPORT = REPO / 'nosetouch_P-VPSDL4_20261024-0159.json'

_PG_URL = os.environ.get('NOSETOUCH_TEST_DATABASE_URL', '').strip()
_BACKENDS = ['sqlite'] + (['postgres'] if _PG_URL else [])


def _postgres_url() -> str:
    url = normalize_database_url(_PG_URL)
    dbname = url.rsplit('/', 1)[-1].split('?', 1)[0]
    if 'test' not in dbname:
        raise RuntimeError(f'refusing to drop tables in {dbname!r}: the test database name must contain "test"')
    return url


@pytest.fixture(params=_BACKENDS)
def backend(request: pytest.FixtureRequest) -> str:
    return request.param


@pytest.fixture
def make_client(backend: str) -> Iterator[Callable[..., TestClient]]:
    """make_client(**settings overrides) -> a started TestClient (lifespan run); closed after the test."""
    started: list[TestClient] = []

    def _make(base_url: str = 'http://testserver', **over: Any) -> TestClient:
        params: dict[str, Any] = {'admin_token': ADMIN_TOKEN}
        if backend == 'postgres':
            params['database_url'] = _postgres_url()
        params.update(over)
        settings = Settings(**params)
        app = create_app(settings)
        if backend == 'postgres' and not started:
            app.state.db.drop_all()  # a clean database for every test
        client = TestClient(app, base_url=base_url)
        client.__enter__()
        started.append(client)
        return client

    yield _make
    for c in started:
        c.__exit__(None, None, None)


@pytest.fixture
def client(make_client: Callable[..., TestClient]) -> TestClient:
    return make_client()


@pytest.fixture
def admin_headers() -> dict[str, str]:
    return {'Authorization': f'Bearer {ADMIN_TOKEN}'}
