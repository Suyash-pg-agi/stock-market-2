#!/bin/bash
while true; do
    uvicorn app:app --host 0.0.0.0 --port 8000 --workers 1
    echo "Uvicorn crashed with exit code $?. Restarting in 5 seconds..."
    sleep 5
done
