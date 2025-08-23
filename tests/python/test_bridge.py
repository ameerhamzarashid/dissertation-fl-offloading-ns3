# -*- coding: utf-8 -*-
from python_fl.co_sim.tcp_protocol import frame

def test_frame_prefix_len():
    s = b'{"hello":"world"}'
    f = frame(s)
    assert len(f) == 4 + len(s)
    # The first 4 bytes encode the length (big-endian)
    n = int.from_bytes(f[:4], "big")
    assert n == len(s)
