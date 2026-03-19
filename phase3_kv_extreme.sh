#!/usr/bin/env bash
set -euo pipefail

# One-shot aggressive 5090 workflow:
# 1) quick KV ranking (8,4,2,1) with reduced validation cost
# 2) finalist confirmation (4,2) with full validation + roundtrip

echo "=== Phase 3 KV Extreme: quick pass ==="
bash ./phase3_kv_quick.sh

echo
echo "=== Phase 3 KV Extreme: finalist pass ==="
bash ./phase3_kv_finalist.sh
