"""Static analysis: shapes, MAC counts and the exact arena plan.

`simulate_arena` mirrors the C runtime's double-ended bump allocator
(runtime/tiny_nn.c) so the predicted peak can be compared byte-for-byte with
the peak measured on target.
"""
from __future__ import annotations

ARENA_ALIGN = 4


def align_up(n: int, a: int = ARENA_ALIGN) -> int:
    return (n + a - 1) // a * a


def layer_table(hp):
    """Return [(name, kind, in_len, in_ch, out_ch, k, stride, out_len)]."""
    from .dataset import NUM_CLASSES, SEQ_LEN
    l0, l1, l2 = hp.dims()
    return [
        ("conv1", "conv1d", SEQ_LEN, 1, hp.c1, hp.k1, hp.stride, l1),
        ("conv2", "conv1d", l1, hp.c1, hp.c2, hp.k2, hp.stride, l2),
        ("fc", "dense", 1, l2 * hp.c2, NUM_CLASSES, 1, 1, 1),
    ]


def mac_count(hp) -> int:
    return sum(out_len * out_ch * k * in_ch for _, _, _, in_ch, out_ch, k, _, out_len in layer_table(hp))


def param_count(hp) -> int:
    return sum(out_ch * k * in_ch + out_ch for _, _, _, in_ch, out_ch, k, _, _ in layer_table(hp))


def activation_elems(hp):
    """Element counts of each tensor in execution order: input, conv1, conv2, logits."""
    from .dataset import SEQ_LEN
    return [SEQ_LEN] + [out_len * out_ch for _, _, _, _, out_ch, _, _, out_len in layer_table(hp)]


def simulate_arena(elem_counts, bytes_per_elem: int = 1, size: int | None = None):
    """Replays the runtime's ping-pong allocation. Returns peak bytes
    (or None if `size` is given and the plan does not fit)."""
    low = high = peak = 0
    ends = [0, 0]  # used bytes at low/high end
    cur_end = 0
    ends[0] = align_up(elem_counts[0] * bytes_per_elem)
    peak = ends[0]
    for n in elem_counts[1:]:
        nxt = 1 - cur_end
        ends[nxt] = align_up(n * bytes_per_elem)
        total = ends[0] + ends[1]
        peak = max(peak, total)
        if size is not None and total > size:
            return None
        ends[cur_end] = 0
        cur_end = nxt
    return peak
