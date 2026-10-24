"""NoseTouch upload service for Heroku: the app's json_post contract, stored in Postgres.

Modules:
    validation  the upload contract of server/receiver.py (stdlib only, shared with import_export.py)
    settings    environment variables and DATABASE_URL normalisation
    db          SQLAlchemy Core tables, engine, schema creation, ingest_log pruning
    store       idempotent upserts (participants, sessions, trials), deletion, admin queries
    exporter    the app's export bundle and the backend folder layout as a zip, both streamed
    admin       admin authentication and the /admin page
    main        the FastAPI application (routes, CORS, error answers, logging)
"""

SERVER_VERSION = '1.0.0'
SERVICE_NAME = 'nosetouch-heroku'
