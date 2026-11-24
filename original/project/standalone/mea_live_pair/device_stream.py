import json
import math
import os
import socket
import threading
import time


class Receiver:
    def __init__(self, directory):
        self.path = directory / 'cpp-spikes.jsonl'
        self.stop = threading.Event()
        self.last = 0.
        self.count = 0
        self.error = None
        self.thread = threading.Thread(target=self.read, daemon=True)

    def read(self):
        try:
            with socket.create_connection(('127.0.0.1', 5555), timeout=3) as client, self.path.open('x') as log:
                client.settimeout(.5)
                pending = b''
                while not self.stop.is_set():
                    try:
                        chunk = client.recv(65536)
                    except socket.timeout:
                        continue
                    if not chunk:
                        raise EOFError('C++ forwarder disconnected')
                    pending += chunk
                    while b'\n' in pending:
                        line, pending = pending.split(b'\n', 1)
                        ts, ch, amp = line.strip().split(b',')
                        ts, ch, amp = int(ts), int(ch), float(amp)
                        now = time.time_ns()
                        if not 0 <= ch < 1024 or not math.isfinite(amp) or ts <= 0:
                            raise ValueError('Invalid C++ spike')
                        log.write(json.dumps([ts, ch, amp, now])+'\n')
                        self.count += 1
                        if abs(now / 1e9 - ts / 1000) < 2:
                            self.last = time.monotonic()
                    log.flush()
                    if len(pending)>4096:
                        raise ValueError('C++ row exceeds protocol maximum')
                os.fsync(log.fileno())
        except Exception as exc:
            self.error = repr(exc)

    def fresh(self):
        if self.error or time.monotonic() - self.last > 5:
            raise RuntimeError('C++ input stale or disconnected: '+str(self.error))
