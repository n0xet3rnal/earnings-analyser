from earnings_analyser.modules.raw_collapse_predictor import _parse_compact
from earnings_analyser.signatures import DIMENSIONS


def _line(dim: str) -> str:
    return f"1,2,0|Summary for {dim}.|3"


def test_parses_well_formed_lines():
    text = "\n".join(_line(d) for d in DIMENSIONS)
    out = _parse_compact(text)
    assert out["forward_guidance"]["weights"] == [1, 2, 0]
    assert out["forward_guidance"]["summary"] == "Summary for forward_guidance."
    assert out["forward_guidance"]["label"] == 3


def test_drops_spurious_leading_field_like_a_dimension_name_prefix():
    # A model sometimes prepends the dimension name before the weights,
    # shifting weights/summary/label off by one if parsed by bare position.
    text = "\n".join(f"{d}|1,2,0|Summary for {d}.|3" for d in DIMENSIONS)
    out = _parse_compact(text)
    for d in DIMENSIONS:
        assert out[d]["weights"] == [1, 2, 0]
        assert out[d]["summary"] == f"Summary for {d}."
        assert out[d]["label"] == 3
