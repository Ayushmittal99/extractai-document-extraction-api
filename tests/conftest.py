import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ["GOOGLE_API_KEY"] = ""  # tests must never use a real key

import json
import pytest
from fastapi.testclient import TestClient

import main
from config import Settings
from extractor import Extractor


class FakeBackend:
    """Stands in for Gemini. `reply` is a str, dict (-> JSON) or an Exception to raise."""
    def __init__(self, reply):
        self.reply, self.calls, self.last_prompt = reply, 0, None

    def generate(self, document_type, prompt):
        self.calls += 1
        self.last_prompt = prompt
        r = self.reply
        if isinstance(r, Exception):
            raise r
        if callable(r):
            return r(self.calls)
        return r if isinstance(r, str) else json.dumps(r)


TEST_SETTINGS = Settings(google_api_key="test", max_input_chars=500, max_retries=2, retry_base_delay_seconds=0.0)


@pytest.fixture
def make_client():
    def _make(reply):
        backend = FakeBackend(reply)
        ext = Extractor(TEST_SETTINGS, backend=backend, sleep=lambda s: None)
        main.app.dependency_overrides[main.get_extractor] = lambda: ext
        main.app.dependency_overrides[main.settings_dep] = lambda: TEST_SETTINGS
        return TestClient(main.app, raise_server_exceptions=False), backend
    yield _make
    main.app.dependency_overrides.clear()
