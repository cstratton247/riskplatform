"""Database connection. DATABASE_URL comes from the environment or a .env file."""
from __future__ import annotations

import os

from dotenv import find_dotenv, load_dotenv
from sqlalchemy import Engine, create_engine


def get_engine() -> Engine:
    load_dotenv(find_dotenv(usecwd=True))
    url = os.environ.get("DATABASE_URL")
    if not url:
        raise RuntimeError("DATABASE_URL is not set (see .env.example)")
    return create_engine(url, pool_pre_ping=True)
