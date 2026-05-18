#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.." || exit 1

IMAGE_NAME="summerclaw-test"

echo "=== Building Docker image ==="
docker build -t "$IMAGE_NAME" .

echo ""
echo "=== Running 'summerclaw onboard' ==="
docker run --name summerclaw-test-run "$IMAGE_NAME" onboard

echo ""
echo "=== Running 'summerclaw status' ==="
STATUS_OUTPUT=$(docker commit summerclaw-test-run summerclaw-test-onboarded > /dev/null && \
    docker run --rm summerclaw-test-onboarded status 2>&1) || true

echo "$STATUS_OUTPUT"

echo ""
echo "=== Validating output ==="
PASS=true

check() {
    if echo "$STATUS_OUTPUT" | grep -q "$1"; then
        echo "  PASS: found '$1'"
    else
        echo "  FAIL: missing '$1'"
        PASS=false
    fi
}

check "summerclaw Status"
check "Config:"
check "Workspace:"
check "Model:"
check "OpenRouter API:"
check "Anthropic API:"
check "OpenAI API:"

echo ""
if $PASS; then
    echo "=== All checks passed ==="
else
    echo "=== Some checks FAILED ==="
    exit 1
fi

# Cleanup
echo ""
echo "=== Cleanup ==="
docker rm -f summerclaw-test-run 2>/dev/null || true
docker rmi -f summerclaw-test-onboarded 2>/dev/null || true
docker rmi -f "$IMAGE_NAME" 2>/dev/null || true
echo "Done."
