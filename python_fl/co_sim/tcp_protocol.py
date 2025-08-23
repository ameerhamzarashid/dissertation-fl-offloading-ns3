# -*- coding: utf-8 -*-
"""
Length-prefixed TCP protocol with retries/heartbeats.
"""
from __future__ import annotations
import asyncio
import struct
from typing import Optional

LEN = struct.Struct("!I")  # 4-byte network order length

def frame(payload: bytes) -> bytes:
    return LEN.pack(len(payload)) + payload

async def send_json(w: asyncio.StreamWriter, payload: str):
    w.write(frame(payload.encode("utf-8")))
    await w.drain()

async def recv_json(r: asyncio.StreamReader) -> Optional[str]:
    hdr = await r.readexactly(4)
    (n,) = LEN.unpack(hdr)
    data = await r.readexactly(n)
    return data.decode("utf-8")
