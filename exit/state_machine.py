"""RFX-1B: Exit State Machine — full execution wiring.

State flow:
OPEN -> LRT_ARM (in loss) / PT_ARM (in profit)
     -> BE_PROTECT (breakeven set)
     -> PT_TP1 -> PT_TP2 -> PT_TP3
     -> TP_FINAL (final anchor hit)
     -> SL_FINAL (final SL anchor hit)
     -> EMERGENCY (backstop)
     -> CLOSED

Design invariants (blockers):
B1: SM is sole authority when wired — HOLD = "wait", not "let legacy decide"
B3: SM never writes stopLoss/trailingStop — returns suggested price via ExitDecision
B4: Terminal actions guarded by is_terminal check (idempotency)

Dependency: risk.policy, risk.liquidity_profile, risk.emergency only (no main.py imports).
"""
from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, List
import time


class ExitState(Enum):
    """Position exit lifecycle states."""
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
    HOLD = "HOLD"                # Do nothing this tick
    SET_BREAKEVEN = "SET_BE"     # Set breakeven SL (via apply_sl_floor)
    ARM_LRT = "ARM_LRT"         # Arm loss recovery trail
    ARM_PT = "ARM_PT"           # Arm profit trail
    PARTIAL_TP = "PARTIAL_TP"   # Execute partial take profit
    CLOSE_TP = "CLOSE_TP"       # Full close at TP_FINAL
    CLOSE_SL = "CLOSE_SL"       # Full close at SL
    CLOSE_EMERGENCY = "CLOSE_EMERGENCY"  # Emergency close
    TIGHTEN_TRAIL = "TIGHTEN_TRAIL"  # Tighten trailing stop (via apply_sl_floor)


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
class EvalContext:
    """All inputs needed for SM evaluate().

    Required fields per plan — no optional dependencies on main.py.
    """
    tick_price: float
    atr: float
    entry_price: float
    side: str                   # 'LONG' or 'SHORT'
    leverage: int
    trailing_stop: float        # Current trailing stop price
    is_trailing_active: bool    # Whether trail is currently active
    tp_ladder: list             # [{'key': 'tp1', 'pct': float, 'close_pct': float}, ...]
    spread_pct: float
    tick_size: float
    margin_usd: float

    # From risk modules
    risk_params: object = None  # RiskParams (optional for parity)
    liq_profile: object = None  # LiquidityProfile (optional for parity)

    # Position context
    recovery_mode: bool = False
    recovery_low: float = 0.0   # Lowest price seen (LONG loss tracking)
    recovery_high: float = 0.0  # Highest price seen (SHORT loss tracking)
    breakeven_activated: bool = False
    partial_tp_state: dict = field(default_factory=dict)
    tp1_trail_armed: bool = False


@dataclass
class ExitDecision:
    """Result of SM evaluate() — action + metadata for adapter."""
    action: ExitAction
    suggested_price: float = 0.0   # For SET_BREAKEVEN, TIGHTEN_TRAIL (adapter uses apply_sl_floor)
    tp_key: str = ""                # Which TP level hit ('tp1', 'tp2', 'tp3', 'tp_final')
    close_pct: float = 0.0         # Partial close percentage (for PARTIAL_TP)
    reason: str = ""                # Human-readable reason
    decision_source: str = "SM"     # 'SM' or 'LEGACY_FALLBACK'


@dataclass
class ExitStateMachine:
    """Exit state machine for a single position.

    RFX-1B: Full execution wiring.
    evaluate() returns ExitDecision — sole authority when RFX1B_EXIT_WIRING=true.
    """
    pos_id: str
    side: str
    entry_price: float
    state: ExitState = ExitState.OPEN
    history: list = field(default_factory=list)
    _created_at: float = field(default_factory=time.time)

    # RFX-1B added fields
    be_price: float = 0.0          # Breakeven price (0 = not set)
    trail_stop: float = 0.0       # SM's tracked trailing stop
    peak_price: float = 0.0       # Best price seen (for trail computation)
    tp_ladder: list = field(default_factory=list)  # Attached TP levels

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

    def evaluate(self, ctx: EvalContext) -> ExitDecision:
        """Evaluate current state and return decision.

        RFX-1B: Full state machine logic.
        B1: This is SOLE AUTHORITY — HOLD means "wait", not "let legacy run".
        B3: Never writes SL directly — returns suggested_price for apply_sl_floor.
        B4: Terminal states return HOLD (idempotency).
        """
        # Terminal guard (B4) — position already closed
        if self.is_terminal:
            return ExitDecision(action=ExitAction.HOLD, reason="TERMINAL_STATE")

        price = ctx.tick_price
        entry = ctx.entry_price
        side = ctx.side
        leverage = max(1, ctx.leverage)
        atr = max(entry * 0.001, ctx.atr)

        # ── Calculate current profit ──
        if side == 'LONG':
            profit_pct = ((price - entry) / entry) * 100 if entry > 0 else 0
        else:
            profit_pct = ((entry - price) / entry) * 100 if entry > 0 else 0
        profit_roi = profit_pct * leverage

        # ── Track peak price for trail ──
        if side == 'LONG':
            self.peak_price = max(self.peak_price or entry, price)
        else:
            self.peak_price = min(self.peak_price or entry, price) if self.peak_price > 0 else price

        # ══════════════════════════════════════
        # PRIORITY 1: Emergency check (preempts all)
        # ══════════════════════════════════════
        emergency_decision = self._check_emergency(ctx, profit_pct, profit_roi)
        if emergency_decision is not None:
            return emergency_decision

        # ══════════════════════════════════════
        # PRIORITY 2: SL_FINAL trailing stop hit
        # ══════════════════════════════════════
        sl_decision = self._check_sl_final(ctx, profit_pct)
        if sl_decision is not None:
            return sl_decision

        # ══════════════════════════════════════
        # PRIORITY 3: TP_FINAL anchor hit
        # ══════════════════════════════════════
        tp_final_decision = self._check_tp_final(ctx, profit_pct)
        if tp_final_decision is not None:
            return tp_final_decision

        # ══════════════════════════════════════
        # PRIORITY 4: State-specific logic
        # ══════════════════════════════════════
        if self.state == ExitState.OPEN:
            return self._eval_open(ctx, profit_pct, profit_roi)
        elif self.state == ExitState.LRT_ARM:
            return self._eval_lrt_arm(ctx, profit_pct)
        elif self.state == ExitState.PT_ARM:
            return self._eval_pt_arm(ctx, profit_pct)
        elif self.state == ExitState.BE_PROTECT:
            return self._eval_be_protect(ctx, profit_pct)
        elif self.state in (ExitState.PT_TP1, ExitState.PT_TP2, ExitState.PT_TP3):
            return self._eval_tp_progression(ctx, profit_pct)

        return ExitDecision(action=ExitAction.HOLD, reason="NO_ACTION")

    # ═══════════════════════════════════════════════════════════════════
    # Priority checks (preempt any state)
    # ═══════════════════════════════════════════════════════════════════

    def _check_emergency(self, ctx: EvalContext, profit_pct: float, profit_roi: float) -> Optional[ExitDecision]:
        """Emergency check — preempts any non-terminal state."""
        if ctx.risk_params is None:
            # Legacy fallback: 50% ROI threshold
            emergency_roi_threshold = 50.0
        else:
            emergency_roi_threshold = ctx.risk_params.emergency_roi

        # Emergency triggers on loss exceeding threshold
        if profit_pct < 0 and abs(profit_roi) >= emergency_roi_threshold:
            self.transition(ExitState.EMERGENCY, f"EMERGENCY_ROI_{abs(profit_roi):.1f}%",
                            price=ctx.tick_price)
            return ExitDecision(
                action=ExitAction.CLOSE_EMERGENCY,
                reason=f"EMERGENCY: ROI={profit_roi:.1f}% >= threshold={emergency_roi_threshold:.0f}%",
                suggested_price=ctx.tick_price,
            )
        return None

    def _check_sl_final(self, ctx: EvalContext, profit_pct: float) -> Optional[ExitDecision]:
        """SL/trailing stop hit — preempts any non-terminal state."""
        trailing = ctx.trailing_stop
        if trailing <= 0:
            return None

        hit = False
        if ctx.side == 'LONG' and ctx.tick_price <= trailing:
            hit = True
        elif ctx.side == 'SHORT' and ctx.tick_price >= trailing:
            hit = True

        if hit:
            self.transition(ExitState.SL_FINAL, f"TRAIL_HIT_{ctx.tick_price:.6f}",
                            price=ctx.tick_price)
            return ExitDecision(
                action=ExitAction.CLOSE_SL,
                reason=f"SL_FINAL: price={ctx.tick_price:.6f} hit trail={trailing:.6f}",
                suggested_price=ctx.tick_price,
            )
        return None

    def _check_tp_final(self, ctx: EvalContext, profit_pct: float) -> Optional[ExitDecision]:
        """TP_FINAL anchor hit — full close."""
        tp_final = self._get_tp_level(ctx, 'tp_final')
        if tp_final is None:
            return None

        if profit_pct >= tp_final['pct']:
            self.transition(ExitState.TP_FINAL, f"TP_FINAL_{profit_pct:.2f}%",
                            price=ctx.tick_price)
            return ExitDecision(
                action=ExitAction.CLOSE_TP,
                tp_key='tp_final',
                close_pct=1.0,  # Full close
                reason=f"TP_FINAL: profit={profit_pct:.2f}% >= target={tp_final['pct']:.2f}%",
                suggested_price=ctx.tick_price,
            )
        return None

    # ═══════════════════════════════════════════════════════════════════
    # State-specific evaluators
    # ═══════════════════════════════════════════════════════════════════

    def _eval_open(self, ctx: EvalContext, profit_pct: float, profit_roi: float) -> ExitDecision:
        """OPEN state: decide if trail should arm (PT or LRT)."""
        side = ctx.side
        atr = ctx.atr
        entry = ctx.entry_price

        # ── Check profit trail activation ──
        # Activate when profit exceeds trail activation threshold
        trail_act_threshold_pct = (atr / entry * 100) * 1.5 if entry > 0 else 2.0  # ~1.5 ATR
        if profit_pct > 0 and profit_pct >= trail_act_threshold_pct:
            self.transition(ExitState.PT_ARM, f"PROFIT_TRAIL_{profit_pct:.2f}%",
                            price=ctx.tick_price)
            return ExitDecision(
                action=ExitAction.ARM_PT,
                reason=f"PT_ARM: profit={profit_pct:.2f}% >= threshold={trail_act_threshold_pct:.2f}%",
            )

        # ── Check loss recovery trail ──
        loss_threshold = 1.0  # default
        bounce_atr = 0.3      # default
        if ctx.risk_params is not None:
            loss_threshold = ctx.risk_params.recovery_trigger_loss_pct
            bounce_atr = ctx.risk_params.recovery_bounce_atr

        if profit_pct < -loss_threshold:
            # In sufficient loss — check for bounce
            bounce_distance = atr * bounce_atr
            if side == 'LONG':
                low = ctx.recovery_low if ctx.recovery_low > 0 else ctx.tick_price
                if ctx.tick_price > low + bounce_distance:
                    self.transition(ExitState.LRT_ARM, f"LRT_BOUNCE_{profit_pct:.2f}%",
                                    price=ctx.tick_price)
                    return ExitDecision(
                        action=ExitAction.ARM_LRT,
                        reason=f"LRT_ARM: loss={profit_pct:.2f}%, bounced {bounce_distance:.4f} from low={low:.6f}",
                    )
            else:  # SHORT
                high = ctx.recovery_high if ctx.recovery_high > 0 else ctx.tick_price
                if ctx.tick_price < high - bounce_distance:
                    self.transition(ExitState.LRT_ARM, f"LRT_BOUNCE_{profit_pct:.2f}%",
                                    price=ctx.tick_price)
                    return ExitDecision(
                        action=ExitAction.ARM_LRT,
                        reason=f"LRT_ARM: loss={profit_pct:.2f}%, bounced {bounce_distance:.4f} from high={high:.6f}",
                    )

        return ExitDecision(action=ExitAction.HOLD, reason="OPEN_WAITING")

    def _eval_lrt_arm(self, ctx: EvalContext, profit_pct: float) -> ExitDecision:
        """LRT_ARM: loss recovery trailing active. Check BE transition or close."""
        # If position recovered to breakeven → transition to BE_PROTECT
        if profit_pct >= 0:
            be_price = self._compute_be_price(ctx)
            self.be_price = be_price
            self.transition(ExitState.BE_PROTECT, f"LRT_TO_BE_{profit_pct:.2f}%",
                            price=ctx.tick_price)
            return ExitDecision(
                action=ExitAction.SET_BREAKEVEN,
                suggested_price=be_price,
                reason=f"BE_PROTECT: recovered from loss, BE=${be_price:.6f}",
            )

        # If still in loss — check recovery SL (tighter trail)
        bounce_atr = 0.3
        if ctx.risk_params is not None:
            bounce_atr = ctx.risk_params.recovery_bounce_atr

        recovery_trail_dist = ctx.atr * bounce_atr
        if ctx.side == 'LONG':
            suggested_trail = ctx.tick_price - recovery_trail_dist
            if self.trail_stop > 0 and suggested_trail > self.trail_stop:
                self.trail_stop = suggested_trail
                return ExitDecision(
                    action=ExitAction.TIGHTEN_TRAIL,
                    suggested_price=suggested_trail,
                    reason=f"LRT_TIGHTEN: trail→${suggested_trail:.6f}",
                )
        else:  # SHORT
            suggested_trail = ctx.tick_price + recovery_trail_dist
            if self.trail_stop > 0 and suggested_trail < self.trail_stop:
                self.trail_stop = suggested_trail
                return ExitDecision(
                    action=ExitAction.TIGHTEN_TRAIL,
                    suggested_price=suggested_trail,
                    reason=f"LRT_TIGHTEN: trail→${suggested_trail:.6f}",
                )
            elif self.trail_stop <= 0:
                self.trail_stop = suggested_trail
                return ExitDecision(
                    action=ExitAction.TIGHTEN_TRAIL,
                    suggested_price=suggested_trail,
                    reason=f"LRT_INIT_TRAIL: trail→${suggested_trail:.6f}",
                )

        return ExitDecision(action=ExitAction.HOLD, reason="LRT_ARM_WAITING")

    def _eval_pt_arm(self, ctx: EvalContext, profit_pct: float) -> ExitDecision:
        """PT_ARM: profit trail armed. Set breakeven → transition to BE_PROTECT."""
        # On trail activation → mandatory BE (B3: via apply_sl_floor in adapter)
        be_price = self._compute_be_price(ctx)
        self.be_price = be_price
        self.transition(ExitState.BE_PROTECT, f"PT_TO_BE_{profit_pct:.2f}%",
                        price=ctx.tick_price)
        return ExitDecision(
            action=ExitAction.SET_BREAKEVEN,
            suggested_price=be_price,
            reason=f"BE_PROTECT: trail active, BE=${be_price:.6f}",
        )

    def _eval_be_protect(self, ctx: EvalContext, profit_pct: float) -> ExitDecision:
        """BE_PROTECT: breakeven set, waiting for TP1."""
        tp1 = self._get_tp_level(ctx, 'tp1')
        if tp1 is not None and profit_pct >= tp1['pct']:
            if not ctx.partial_tp_state.get('tp1', False):
                self.transition(ExitState.PT_TP1, f"TP1_{profit_pct:.2f}%",
                                price=ctx.tick_price)
                # Re-confirm BE on TP1 (idempotent)
                be_price = self._compute_be_price(ctx)
                self.be_price = be_price
                return ExitDecision(
                    action=ExitAction.PARTIAL_TP,
                    tp_key='tp1',
                    close_pct=tp1.get('close_pct', 0.40),
                    suggested_price=be_price,  # BE re-confirmation
                    reason=f"TP1_HIT: profit={profit_pct:.2f}% >= {tp1['pct']:.2f}%, close {tp1.get('close_pct', 0.40):.0%}",
                )

        # Trail tightening while waiting for TP1
        tighten = self._compute_trail_tighten(ctx, profit_pct)
        if tighten is not None:
            return tighten

        return ExitDecision(action=ExitAction.HOLD, reason="BE_PROTECT_WAITING")

    def _eval_tp_progression(self, ctx: EvalContext, profit_pct: float) -> ExitDecision:
        """PT_TP1/TP2/TP3: forward-only TP progression."""
        # Determine next TP target based on current state
        next_map = {
            ExitState.PT_TP1: ('tp2', ExitState.PT_TP2),
            ExitState.PT_TP2: ('tp3', ExitState.PT_TP3),
            ExitState.PT_TP3: (None, None),  # TP3 → TP_FINAL handled by priority check
        }
        tp_key, next_state = next_map.get(self.state, (None, None))

        if tp_key is not None and next_state is not None:
            tp = self._get_tp_level(ctx, tp_key)
            if tp is not None and profit_pct >= tp['pct']:
                if not ctx.partial_tp_state.get(tp_key, False):
                    self.transition(next_state, f"{tp_key.upper()}_{profit_pct:.2f}%",
                                    price=ctx.tick_price)
                    return ExitDecision(
                        action=ExitAction.PARTIAL_TP,
                        tp_key=tp_key,
                        close_pct=tp.get('close_pct', 0.30),
                        reason=f"{tp_key.upper()}_HIT: profit={profit_pct:.2f}% >= {tp['pct']:.2f}%",
                    )

        # Trail tightening during TP progression
        tighten = self._compute_trail_tighten(ctx, profit_pct)
        if tighten is not None:
            return tighten

        return ExitDecision(action=ExitAction.HOLD, reason=f"{self.state.value}_WAITING")

    # ═══════════════════════════════════════════════════════════════════
    # Helpers
    # ═══════════════════════════════════════════════════════════════════

    def _compute_be_price(self, ctx: EvalContext) -> float:
        """Compute breakeven price with fee+spread+slippage buffer."""
        entry = ctx.entry_price
        spread = max(0, ctx.spread_pct)
        # Fee buffer: round_trip_fee(0.08%) + 60% spread + slippage(0.02%) + safety(0.02%)
        buffer_pct = 0.08 + 0.6 * spread + 0.02 + 0.02
        buffer_pct = max(0.12, min(0.80, buffer_pct))  # Clamp
        buffer_ratio = buffer_pct / 100

        if ctx.side == 'LONG':
            return entry * (1 + buffer_ratio)
        else:
            return entry * (1 - buffer_ratio)

    def _compute_trail_tighten(self, ctx: EvalContext, profit_pct: float) -> Optional[ExitDecision]:
        """Trail tightening for states where trail is active."""
        if not ctx.is_trailing_active and self.state not in (ExitState.PT_ARM, ExitState.LRT_ARM):
            return None

        trail_mult = 1.0
        if ctx.risk_params is not None:
            trail_mult = ctx.risk_params.trail_distance_mult

        trail_dist = ctx.atr * 1.5 * trail_mult  # Base trail distance

        if ctx.side == 'LONG':
            suggested = self.peak_price - trail_dist
            suggested = max(suggested, ctx.entry_price)  # Never below entry if in profit
            if self.trail_stop > 0 and suggested > self.trail_stop:
                self.trail_stop = suggested
                return ExitDecision(
                    action=ExitAction.TIGHTEN_TRAIL,
                    suggested_price=suggested,
                    reason=f"TRAIL_UP: peak={self.peak_price:.6f} trail→${suggested:.6f}",
                )
        else:  # SHORT
            suggested = self.peak_price + trail_dist
            suggested = min(suggested, ctx.entry_price) if profit_pct > 0 else suggested
            if self.trail_stop > 0 and suggested < self.trail_stop:
                self.trail_stop = suggested
                return ExitDecision(
                    action=ExitAction.TIGHTEN_TRAIL,
                    suggested_price=suggested,
                    reason=f"TRAIL_DOWN: peak={self.peak_price:.6f} trail→${suggested:.6f}",
                )

        return None

    def _get_tp_level(self, ctx: EvalContext, key: str) -> Optional[dict]:
        """Get TP level from context's ladder."""
        for lv in (ctx.tp_ladder or []):
            if lv.get('key') == key:
                return lv
        return None

    @property
    def is_terminal(self) -> bool:
        """Position has reached a terminal state."""
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
            'be_price': self.be_price,
            'trail_stop': self.trail_stop,
            'peak_price': self.peak_price,
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
        sm.be_price = data.get('be_price', 0.0)
        sm.trail_stop = data.get('trail_stop', 0.0)
        sm.peak_price = data.get('peak_price', 0.0)
        # Deserialize history
        for h in data.get('history', []):
            sm.history.append(StateTransition(
                timestamp=h.get('ts', 0),
                from_state=ExitState(h.get('from', 'OPEN')),
                to_state=ExitState(h.get('to', 'OPEN')),
                reason=h.get('reason', ''),
                price=h.get('price', 0),
                telemetry_event=h.get('event', ''),
            ))
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
