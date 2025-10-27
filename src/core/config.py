from pydantic import BaseModel, field_validator
from typing import Literal, Dict, Any


class Config(BaseModel):
    """Configuration for momentum backtest strategy."""
    
    universe_size: int
    entry_count: int
    exit_rank_threshold: int
    lookback_months: int = 3
    holding_quarters: int = 2
    rebalance_frequency: Literal["quarterly"] = "quarterly"
    transaction_cost_bps: float
    slippage_bps: float
    min_history_days: int = 63
    cash_rate: float = 0.0
    # 
    model_config = {"str_strip_whitespace": True}
    
    @field_validator("entry_count")
    @classmethod
    def validate_entry_count(cls, v: int, info) -> int:
        """Ensure entry_count <= universe_size."""
        if "universe_size" in info.data:
            if v > info.data["universe_size"]:
                raise ValueError(
                    f"entry_count ({v}) must be <= universe_size ({info.data['universe_size']})"
                )
        return v
    
    @field_validator("exit_rank_threshold")
    @classmethod
    def validate_exit_rank_threshold(cls, v: int, info) -> int:
        """Ensure exit_rank_threshold >= entry_count."""
        if "entry_count" in info.data:
            if v < info.data["entry_count"]:
                raise ValueError(
                    f"exit_rank_threshold ({v}) must be >= entry_count ({info.data['entry_count']})"
                )
        return v
    
    def summary(self) -> Dict[str, Any]:
        """Return a summary dictionary of configuration parameters."""
        return {
            "universe_size": self.universe_size,
            "entry_count": self.entry_count,
            "exit_rank_threshold": self.exit_rank_threshold,
            "lookback_months": self.lookback_months,
            "holding_quarters": self.holding_quarters,
            "rebalance_frequency": self.rebalance_frequency,
            "transaction_cost_bps": self.transaction_cost_bps,
            "slippage_bps": self.slippage_bps,
            "min_history_days": self.min_history_days,
            "cash_rate": self.cash_rate,
        }
