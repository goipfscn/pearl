#!/bin/bash
# Minimal simnet mining test for sm_120
# Run after test_sm120_smoke.sh passes

set -e

echo "=== Pearl sm_120 Simnet Mining Test - $(date) ==="

# Check if pearld exists
if [ ! -f "bin/pearld" ]; then
    echo "ERROR: bin/pearld not found. Run 'task build:blockchain' first"
    exit 1
fi

# Get mining address
MINING_ADDR=${PEARLD_MINING_ADDRESS:-""}
if [ -z "$MINING_ADDR" ]; then
    echo "ERROR: Set PEARLD_MINING_ADDRESS environment variable"
    echo "Example: export PEARLD_MINING_ADDRESS=\$(./bin/prlctl getnewaddress)"
    exit 1
fi

echo "Mining address: $MINING_ADDR"

# Start pearld in simnet mode (background)
echo "Starting pearld (simnet)..."
./bin/pearld \
  --simnet \
  --rpcuser=test \
  --rpcpass=test \
  --rpclisten=127.0.0.1:18556 \
  --miningaddr="$MINING_ADDR" \
  --logdir=/tmp/pearld-simnet \
  --debuglevel=info &

PEARLD_PID=$!
echo "pearld PID: $PEARLD_PID"

# Wait for sync
echo "Waiting for pearld sync..."
sleep 5

# Check if running
if ! kill -0 $PEARLD_PID 2>/dev/null; then
    echo "ERROR: pearld failed to start"
    exit 1
fi

# Start gateway
echo "Starting pearl-gateway..."
export PEARLD_RPC_URL="http://127.0.0.1:18556"
export PEARLD_RPC_USER="test"
export PEARLD_RPC_PASSWORD="test"
export PEARLD_MINING_ADDRESS="$MINING_ADDR"

pearl-gateway start &
GATEWAY_PID=$!
echo "Gateway PID: $GATEWAY_PID"

sleep 3

# Check if running
if ! kill -0 $GATEWAY_PID 2>/dev/null; then
    echo "ERROR: gateway failed to start"
    kill $PEARLD_PID 2>/dev/null
    exit 1
fi

echo ""
echo "=== Mining Stack Running ==="
echo "pearld: http://127.0.0.1:18556"
echo "gateway: /tmp/pearlgw.sock"
echo ""
echo "To test vLLM miner:"
echo "  vllm serve pearl-ai/Llama-3.3-8B-Instruct-pearl \\"
echo "    --host 127.0.0.1 --port 8000 --max-model-len 2048 \\"
echo "    --gpu-memory-utilization 0.7 --enforce-eager"
echo ""
echo "Press Ctrl+C to stop, or run: kill $PEARLD_PID $GATEWAY_PID"

# Keep running
wait $GATEWAY_PID
