"""HTML screens (guide section 10). The pages are thin: the data is fetched by app/static/app.js from /api/*.

The user (and its cookie) is created here, on the page request, so that the several parallel API calls the
browser makes right afterwards all belong to the same user.
"""
from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, Depends, Request
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates

from app.models import User
from app.routers.deps import COOKIE, current_user, set_user_cookie

TEMPLATES_DIR = Path(__file__).resolve().parent.parent / "templates"
templates = Jinja2Templates(directory=str(TEMPLATES_DIR))
router = APIRouter(include_in_schema=False)

SCREENS = {                      # path -> (template, title)
    "/": ("home.html", "Главная"),
    "/add": ("add.html", "Добавить траты"),
    "/categories": ("categories.html", "Категории"),
    "/recommendations": ("recommendations.html", "Советы"),
    "/goals": ("goals.html", "Цели"),
    "/subscriptions": ("subscriptions.html", "Подписки"),
    "/settings": ("settings.html", "Настройки"),
}


def _make(template: str, title: str):
    def view(request: Request, user: User = Depends(current_user)) -> HTMLResponse:
        resp = templates.TemplateResponse(request, template, {"title": title, "page": template[:-5]})
        if request.cookies.get(COOKIE) != str(user.id):      # first visit: the cookie rides on this response
            set_user_cookie(resp, user)
        return resp
    return view


for _path, (_tpl, _title) in SCREENS.items():
    router.add_api_route(_path, _make(_tpl, _title), methods=["GET"], response_class=HTMLResponse,
                         name=_tpl[:-5])


@router.get("/budget")
def budget_alias():
    return RedirectResponse("/goals")            # the guide's "Бюджет" screen is the "Цели" screen now
