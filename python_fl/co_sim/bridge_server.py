# -*- coding: utf-8 -*-
"""
TCP bridge server that listens for NS-3 clients.
- Receives "state" messages: {type:"state", ue:int, size:int, cycles:int}
- Decides action (0=local,1=offload) using a simple heuristic or agent hook.
- Sends ActionMsg back.
Also aggregates per-round metrics for logging and reward computation.
"""
from __future__ import annotations
import asyncio
import json
from typing import Dict, Any
from .tcp_protocol import send_json, recv_json
from .message_schema import ActionMsg, dataclass_to_json
from ..utils.metrics import MetricsAggregator
from ..reward_functions.energy_latency_reward import EnergyLatencyReward

DEFAULT_HOST = "0.0.0.0"
DEFAULT_PORT = 50051

class BridgeServer:
    def __init__(self, host: str = DEFAULT_HOST, port: int = DEFAULT_PORT):
        self.host = host
        self.port = port
        self.metrics = MetricsAggregator()
        self.reward = EnergyLatencyReward(latency_weight=0.5, energy_weight=0.5)

    async def handle_client(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter):
        addr = writer.get_extra_info("peername")
        while True:
            try:
                payload = await recv_json(reader)
            except asyncio.IncompleteReadError:
                break
            msg = json.loads(payload)
            if msg.get("type") == "state":
                # Simple heuristic: if size <= 10MB offload else local
                action = 1 if msg["size"] <= 10*1024*1024 else 0
                out = ActionMsg(type="action", ue=msg["ue"], action=action)
                await send_json(writer, dataclass_to_json(out))
                # We could also log this decision and update metrics asynchronously
            else:
                # ignore unknown
                pass
        writer.close()
        await writer.wait_closed()

    async def run(self):
        server = await asyncio.start_server(self.handle_client, self.host, self.port)
        async with server:
            await server.serve_forever()

if __name__ == "__main__":
    bs = BridgeServer()
    asyncio.run(bs.run())
