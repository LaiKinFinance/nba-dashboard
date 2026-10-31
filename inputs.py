"""Bounded request data for saved picks, bets and probability calculations."""
from typing import Literal

from pydantic import BaseModel, Field, FiniteFloat, model_validator


class Leg(BaseModel):
    game_id: str = Field(min_length=1, max_length=64, pattern=r'^[A-Za-z0-9_.-]+$')              # ids are plain letters, digits and - _ . only: nothing that could run as page code
    player_id: str | None = Field(default=None, max_length=64, pattern=r'^[A-Za-z0-9_.-]+$')
    stat: Literal['pts', 'reb', 'ast', 'fg3m', 'stl', 'blk', 'pra', 'pr', 'pa', 'dd', 'td', 'ml', 'spread', 'total', 'tt', 'band']
    line: FiniteFloat = Field(default=0, ge=-1000, le=1000)
    side: Literal['over', 'under', 'home', 'away'] = 'over'
    team: Literal['home', 'away'] | None = None
    lo: int | None = Field(default=None, ge=1, le=1000)
    hi: int | None = Field(default=None, ge=1, le=1000)
    text: str = Field(default='', max_length=200)
    kind: Literal['atleast'] | None = None      # 'atleast' = "20 or more" (wins on 20, loses on 19); different from 'over 20'
    price: FiniteFloat | None = Field(default=None, gt=1, le=1000)    # the price shown when the leg was added (kept with saved records; not used to work out any payment)

    @model_validator(mode='after')
    def market_fields(self):
        player = self.stat not in ('ml', 'spread', 'total', 'tt', 'band')
        allowed = ('over', 'under') if player or self.stat in ('total', 'tt') else ('home', 'away')
        if self.side not in allowed:
            raise ValueError('Side does not match this market')
        if player and not self.player_id:
            raise ValueError('Player market requires a player')
        if self.stat in ('tt', 'band') and self.team is None:
            raise ValueError('This market requires a team')
        if self.stat == 'band' and (self.lo is None or (self.hi is not None and self.hi < self.lo)):
            raise ValueError('Margin band requires valid bounds')
        if self.kind == 'atleast':
            if not player or self.stat in ('dd', 'td') or self.side != 'over' or not float(self.line).is_integer() or self.line < 1:
                raise ValueError('"At least" bets need a player stat, side over and a whole number of 1 or more')
        if self.stat != 'spread' and self.line < 0:
            raise ValueError('Line must be nonnegative')
        return self


class ParlayInput(BaseModel):
    legs: list[Leg] = Field(min_length=1, max_length=6)
    odds: FiniteFloat | None = Field(default=None, gt=1, le=1000000)


class PickInput(ParlayInput):
    note: str = Field(default='', max_length=500)
    model_prob: FiniteFloat | None = Field(default=None, ge=0, le=1)


class BetInput(ParlayInput):
    """A personal pick. The old route name is kept so saved history still works."""
    model_prob: FiniteFloat | None = Field(default=None, ge=0, le=1)
