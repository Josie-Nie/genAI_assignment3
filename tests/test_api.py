from fastapi.testclient import TestClient
from app import main
from helper_lib.model import Generator

client = TestClient(main.app)


def test_home():
    assert client.get('/').status_code == 200


def test_generate_png(monkeypatch):
    # Uses an untrained model ONLY to check API mechanics, not image quality.
    monkeypatch.setattr(main, 'get_generator', lambda: Generator())
    response = client.get('/generate?num_samples=3')
    assert response.status_code == 200
    assert response.headers['content-type'] == 'image/png'
    assert response.content.startswith(b'\x89PNG')


def test_invalid_count():
    response = client.get('/generate?num_samples=101')
    assert response.status_code == 422
