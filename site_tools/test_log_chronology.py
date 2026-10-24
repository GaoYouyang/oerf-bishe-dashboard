from datetime import date
from pathlib import Path
from urllib.parse import urlsplit

from bs4 import BeautifulSoup


ROOT = Path(__file__).resolve().parents[1]


def read_page(name):
    return BeautifulSoup((ROOT / name).read_text(), "html.parser")


def test_homepage_increment_sections_lead_with_actual_october_events():
    expected = ["2026-10-20", "2026-10-19", "2026-10-17"]
    for name in ("index.html", "operator-learning/index.html"):
        section = read_page(name).select_one("#engineering-increments")
        assert section["data-reviewed-date"] == "2026-10-24"
        cards = section.select("article[data-event-date]")
        assert [card["data-event-date"] for card in cards] == expected
        for card in cards:
            assert card.time["datetime"] == card["data-event-date"]
            assert card.time.get_text() == card["data-event-date"]
            for node in card.select("[data-i18n-zh]"):
                assert node.get_text() == node["data-i18n-zh"]
                assert node.get("data-i18n-en")
            link = urlsplit(card.a["href"])
            target = (ROOT / name).parent / link.path
            assert BeautifulSoup(target.read_text(), "html.parser").find(id=link.fragment)
        assert "2.98GiB" in cards[0].get_text()
        assert "909/909" in cards[1].get_text()
        assert "统计对照仍更准" in cards[0].get_text()


def test_review_date_does_not_relabel_historical_event_dates():
    soup = read_page("operator-learning/daily-progress.html")
    review = soup.select_one("#log-date-review")
    assert review["data-reviewed-date"] == "2026-10-24"
    assert review["data-timezone"] == "Asia/Shanghai"
    assert soup.select_one("#latest")["data-date"] == "2026-10-24"
    assert soup.select_one("#log-20261011")["data-date"] == "2026-10-11"
    entries = soup.select(".day-entry")
    for entry in entries:
        date.fromisoformat(entry["data-date"])
        for time in entry.select("time[datetime]"):
            assert time["datetime"] == entry["data-date"]
    assert any(entry["data-date"].startswith("2026-08") for entry in entries)
    assert soup.select_one("#retained12-sparse-stream-resource-20261020")
    assert soup.select_one('script[src="../assets/log-chronology.js"]')


def test_calendar_uses_stored_dates_not_today_and_preserves_old_links():
    js = (ROOT / "assets/log-chronology.js").read_text()
    assert "entry.dataset.date" in js
    assert "dataset.calendarMonth" in js
    assert "dataset.calendarTarget" in js
    assert "Date.now" not in js and "new Date(" not in js
    html = (ROOT / "operator-learning/daily-progress.html").read_text()
    assert "window.OERFLogChronology.groupEntries(root, entries)" in html
    assert "linked.dataset.calendarTarget" in html
    assert 'id="phase-factor"' in html and 'id="phase-c-route"' in html


def test_plain_language_log_distinguishes_review_from_scientific_milestone():
    soup = read_page("learning_log.html")
    assert soup.select_one("#log-date-review")["data-reviewed-date"] == "2026-10-24"
    assert "10月20日" in soup.select_one("#log-date-review").get_text()
    assert soup.select_one('a[href="operator-learning/daily-progress.html#latest"]')
    text = (ROOT / "docs/operator_3d_learning_log.md").read_text()
    assert "2026-10-24" in text[:1300]
    assert "2026-10-20" in text[:1300]
