#!/bin/bash
ray start --head --port 0
python3 server.py
python3 client.py
ray stop --force 2>/dev/null || true