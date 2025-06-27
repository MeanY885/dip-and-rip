# build_and_run.sh
#!/bin/bash

# Build the Docker image
docker build -t btc-backtester .

# Run the container
docker run -p 8000:8000 btc-backtester