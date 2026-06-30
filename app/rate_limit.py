"""Rate limiting using slowapi with configurable limits."""

import os

from slowapi import Limiter
from slowapi.util import get_remote_address

limiter = Limiter(
    key_func=get_remote_address,
    default_limits=[
        os.environ.get("RATE_LIMIT_DEFAULT", "60/minute"),
    ],
    storage_uri=os.environ.get("RATE_LIMIT_STORAGE", "memory://"),
)
