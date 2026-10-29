"""Bounded request data for saved picks, bets and probability calculations."""
from typing import Literal

from pydantic import BaseModel, Field, FiniteFloat


class Leg(BaseModel):
    game_id: str = Field(min_length=1, max_length=64)
    player_id: str | None = Field(default=None, max_length=64)
    stat: Literal['pts', 'reb', 'ast', 'fg3m', 'stl', 'blk', 'pra', 'pr', 'pa', 'dd', 'td', 'ml', 'spread', 'total', 'tt', 'band']
    line: FiniteFloat = Field(default=0, ge=-1000, le=1000)
    side: Literal['over', 'under', 'home', 'away'] = 'over'
    team: Literal['home', 'away'] | None = None
    lo: int | None = Field(default=None, ge=1, le=1000)
    hi: int | None = Field(default=None, ge=1, le=1000)
    text: str = Field(default='', max_length=200)


class ParlayInput(BaseModel):
    legs: list[Leg] = Field(min_length=1, max_length=6)
    odds: FiniteFloat | None = Field(default=None, gt=1, le=1000000)


class PickInput(ParlayInput):
    note: str = Field(default='', max_length=500)
    model_prob: FiniteFloat | None = Field(default=None, ge=0, le=1)


class BetInput(ParlayInput):
    odds: FiniteFloat = Field(gt=1, le=1000000)
    stake: FiniteFloat = Field(gt=0, le=1000000)
    model_prob: FiniteFloat | None = Field(default=None, ge=0, le=1)
