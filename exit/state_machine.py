"""RFX-1A: Exit State Machine — skeleton only.

Defines all states and transitions for the position exit lifecycle.
RFX-1A: states + history recording. NO execution wiring.
Execution wiring comes in RFX-1B.

Dependency: risk.policy, risk.liquidity_profile only (blocker 2).
"""
from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional
import time


class ExitState(Enum):
    """Position exit lifecycle states.

    State flow:
    OPEN -> LRT_ARM (in loss) / PT_ARM (in profit)
         -> BE_PROTECT (breakeven set)
         -> PT_TP1 -> PT_TP2 -> PT_TP3
         -> TP_FINAL (final anchor hit)
         -> SL_FINAL (final SL anchor hit)
         -> EMERGENCY (backstop)
         -> CLOSED
    """
    OPEN = "OPEN"
    LRT_ARM = "LRT_ARM"           # Loss Recovery Trail armed
    PT_ARM = "PT_ARM"             # Profit Trail armed
    BE_PROTECT = "BE_PROTECT"     # Breakeven protection set
    PT_TP1 = "PT_TP1"            # TP1 hit, partial closed
    PT_TP2 = "PT_TP2"            # TP2 hit, partial closed
    PT_TP3 = "PT_TP3"            # TP3 hit, partial closed
    TP_FINAL = "TP_FINAL"        # Final TP anchor hit
    SL_FINAL = "SL_FINAL"        # Final SL anchor hit
    EMERGENCY = "EMERGENCY"       # Emergency backstop hit
    CLOSED = "CLOSED"            # Position closed


class ExitAction(Enum):
    """Actions returned by state machine evaluation."""
    HOLD = "HOLD"                # Do nothing
    SET_BREAKEVEN = "SET_BE"     # Set breakeven SL
    ARM_LRT = "ARM_LRT"         # Arm loss recovery trail
    ARM_PT = "ARM_PT"           # Arm profit trail
    PARTIAL_TP = "PARTIAL_TP"   # Execute partial take profit
    CLOSE_TP = "CLOSE_TP"       # Full close at TP
    CLOSE_SL = "CLOSE_SL"       # Full close at SL
    CLOSE_EMERGENCY = "CLOSE_EMERGENCY"  # Emergency close
    REDUCE = "REDUCE"           # Time-based reduction


# Telemetry event names
TELEM_TRAIL_PROFIT_ARMED = "TRAIL_PROFIT_ARMED"
TELEM_TRAIL_LOSS_ARMED = "TRAIL_LOSS_ARMED"
TELEM_TRAIL_BE_SET = "TRAIL_BE_SET"
TELEM_FINAL_ANCHOR_HIT = "FINAL_ANCHOR_HIT"
TELEM_EMERGENCY_HIT = "EMERGENCY_HIT"
TELEM_TP1_HIT = "TP1_HIT"
TELEM_TP2_HIT = "TP2_HIT"
TELEM_TP3_HIT = "TP3_HIT"


@dataclass
class StateTransition:
    """Record of a state transition."""
    timestamp: float
    from_state: ExitState
    to_state: ExitState
    reason: str
    price: float = 0.0
    telemetry_event: str = ""


@dataclass
class ExitStateMachine:
    """Exit state machine for a single position.

    RFX-1A: Skeleton only — states defined, transitions recorded.
    evaluate() always returns HOLD (no execution wiring).

    RFX-1B will wire evaluate() to actual exit logic.
    """
    pos_id: str
    side: str
    entry_price: float
    state: ExitState = ExitState.OPEN
    history: list = field(default_factory=list)
    _created_at: float = field(default_factory=time.time)

    def transition(self, new_state: ExitState, reason: str, price: float = 0.0) -> bool:
        """Record a state transition.

        Returns True if transition was valid (state actually changed).
        """
        if new_state == self.state:
            return False

        record = StateTransition(
            timestamp=time.time(),
            from_state=self.state,
            to_state=new_state,
            reason=reason,
            price=price,
            telemetry_event=self._get_telemetry_event(new_state),
        )
        self.history.append(record)
        self.state = new_state
        return True

    def evaluate(
        self,
        tick_price: float,
        atr: float,
    ) -> ExitAction:
        """Evaluate current state and return action.

        RFX-1A: STUB — always returns HOLD.
        RFX-1B will implement full state machine logic.
        """
        return ExitAction.HOLD

    @property
    def is_terminal(self) -> bool:
        """Position has reached a terminal state (closed)."""
        return self.state in (
            ExitState.TP_FINAL,
            ExitState.SL_FINAL,
            ExitState.EMERGENCY,
            ExitState.CLOSED,
        )

    @property
    def last_transition(self) -> Optional[StateTransition]:
        """Most recent state transition."""
        return self.history[-1] if self.history else None

    def to_dict(self) -> dict:
        """Serialize for position storage / API response."""
        return {
            'pos_id': self.pos_id,
            'state': self.state.value,
            'side': self.side,
            'entry_price': self.entry_price,
            'created_at': self._created_at,
            'history': [
                {
                    'ts': t.timestamp,
                    'from': t.from_state.value,
                    'to': t.to_state.value,
                    'reason': t.reason,
                    'price': t.price,
                    'event': t.telemetry_event,
                }
                for t in self.history
            ],
        }

    @classmethod
    def from_dict(cls, data: dict) -> 'ExitStateMachine':
        """Deserialize from storage."""
        sm = cls(
            pos_id=data.get('pos_id', ''),
            side=data.get('side', 'LONG'),
            entry_price=data.get('entry_price', 0),
        )
        sm.state = ExitState(data.get('state', 'OPEN'))
        sm._created_at = data.get('created_at', time.time())
        # History is not deserialized in RFX-1A (skeleton)
        return sm

    @staticmethod
    def _get_telemetry_event(state: ExitState) -> str:
        """Map state to telemetry event name."""
        mapping = {
            ExitState.PT_ARM: TELEM_TRAIL_PROFIT_ARMED,
            ExitState.LRT_ARM: TELEM_TRAIL_LOSS_ARMED,
            ExitState.BE_PROTECT: TELEM_TRAIL_BE_SET,
            ExitState.TP_FINAL: TELEM_FINAL_ANCHOR_HIT,
            ExitState.SL_FINAL: TELEM_FINAL_ANCHOR_HIT,
            ExitState.EMERGENCY: TELEM_EMERGENCY_HIT,
            ExitState.PT_TP1: TELEM_TP1_HIT,
            ExitState.PT_TP2: TELEM_TP2_HIT,
            ExitState.PT_TP3: TELEM_TP3_HIT,
        }
        return mapping.get(state, "")
