import pytest
import datetime as dt
import slate


@pytest.mark.parametrize('full,short,match', [('Nikola Jokić', 'N. Jokic', True), ('Gary Trent Jr.', 'G. Trent', True),
                                            ('Jalen Williams', 'Jaylin Williams', False), ('Desmond Bane', 'D. Bane', True)])
def test_names_normalize_accents_and_suffixes_without_collapsing_full_names(full, short, match):
    assert slate.name_match(full, short) is match


def test_slate_day_selection_observes_daylight_saving(monkeypatch):
    class Frozen(dt.datetime):
        @classmethod
        def now(cls, tz=None):
            return cls(2026, 7, 1, 4, 30, tzinfo=dt.timezone.utc).astimezone(tz)
    monkeypatch.setattr(slate.dt, 'datetime', Frozen)
    dates = []
    class Response:
        def json(self): return {'events': []}
    def get(*args, **kwargs):
        dates.append(kwargs['params']['dates']); return Response()
    monkeypatch.setattr(slate.requests, 'get', get)
    assert slate.upcoming(1) == []
    assert dates == ['20260701']
