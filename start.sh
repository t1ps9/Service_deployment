#!/bin/bash
python grpc_my.py &
sleep 5
python task1_app.py &
wait