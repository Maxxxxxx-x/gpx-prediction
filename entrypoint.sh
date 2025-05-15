#!/bin/bash

service dbus start

setsid warp-svc > /var/log/warp-svc 2>&1 < /dev/null &

sleep 1s

warp-cli --accept-tos connect

python ./src/main.py

/bin/bash
