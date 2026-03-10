import main


def test_structural_tp_long_anchors_to_resistance():
    result = main.compute_structural_tp(
        base_tp=106.0,
        signal_side="LONG",
        current_price=100.0,
        entry_price=98.0,
        sr_levels={
            "supports": [],
            "resistances": [
                {"price": 106.8, "touches": 3},
                {"price": 112.0, "touches": 4},
            ],
        },
        composite_vol={"category": "NORMAL"},
        atr=2.0,
    )

    assert result["adjusted"] is True
    assert result["source"] == "resistance_target"
    assert result["structural_level"] == 106.8
    assert abs(result["tp"] - 106.56) < 1e-9


def test_structural_tp_short_anchors_to_support():
    result = main.compute_structural_tp(
        base_tp=94.0,
        signal_side="SHORT",
        current_price=100.0,
        entry_price=102.0,
        sr_levels={
            "supports": [
                {"price": 93.2, "touches": 3},
                {"price": 88.0, "touches": 4},
            ],
            "resistances": [],
        },
        composite_vol={"category": "NORMAL"},
        atr=2.0,
    )

    assert result["adjusted"] is True
    assert result["source"] == "support_target"
    assert result["structural_level"] == 93.2
    assert abs(result["tp"] - 93.44) < 1e-9


def test_structural_tp_keeps_base_when_level_is_not_sensible():
    result = main.compute_structural_tp(
        base_tp=106.0,
        signal_side="LONG",
        current_price=100.0,
        entry_price=98.0,
        sr_levels={
            "supports": [],
            "resistances": [
                {"price": 120.0, "touches": 4},
            ],
        },
        composite_vol={"category": "NORMAL"},
        atr=2.0,
    )

    assert result["adjusted"] is False
    assert result["source"] == "atr_only"
    assert result["tp"] == 106.0
