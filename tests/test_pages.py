"""HTML screens (guide section 10): every page renders, the cookie is set, static files are served."""
import datetime as dt

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("sqlalchemy")
pytest.importorskip("httpx")
pytest.importorskip("jinja2")

TODAY = dt.date(2026, 10, 29)
PAGES = {"/": "home", "/add": "add", "/categories": "categories", "/recommendations": "recommendations",
         "/goals": "goals", "/subscriptions": "subscriptions", "/settings": "settings"}


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setenv("DATABASE_URL", "sqlite://")
    monkeypatch.setenv("APP_ENV", "test")
    from app import config, db
    from app.services import clock

    config.get_settings.cache_clear()
    db.reset_engine()
    monkeypatch.setattr(clock, "today", lambda tz="Europe/Moscow": TODAY)
    from fastapi.testclient import TestClient
    from app.main import app

    with TestClient(app) as c:
        yield c
    db.reset_engine()
    config.get_settings.cache_clear()


@pytest.mark.parametrize("path,page", PAGES.items())
def test_every_screen_renders(client, path, page):
    r = client.get(path)
    assert r.status_code == 200
    assert f'data-page="{page}"' in r.text
    assert 'id="app"' in r.text
    assert "не является" in r.text and "консультацией" in r.text      # the disclaimer is on every screen
    assert "/static/app.js" in r.text


def test_first_page_request_sets_the_user_cookie(client):
    r = client.get("/")
    assert "uid" in r.cookies or "uid=" in r.headers.get("set-cookie", "")
    assert "httponly" in r.headers["set-cookie"].lower()


def test_budget_redirects_to_goals(client):
    r = client.get("/budget", follow_redirects=False)
    assert r.status_code in (301, 302, 307, 308) and r.headers["location"].endswith("/goals")


def test_static_files(client):
    assert client.get("/static/app.js").status_code == 200
    css = client.get("/static/style.css")
    assert css.status_code == 200 and "prefers-color-scheme" in css.text


def test_api_and_pages_share_the_user(client):
    client.get("/")
    client.post("/api/demo/load")
    assert client.get("/api/summary").json()["total"] > 0
