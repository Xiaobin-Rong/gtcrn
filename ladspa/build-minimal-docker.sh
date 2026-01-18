#!/bin/bash
# =============================================================================
# GTCRN - Build Minimal ONNX Runtime using Docker
# =============================================================================
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/.." # Go to project root

# Colors
GREEN='\033[0;32m'
CYAN='\033[0;36m'
NC='\033[0m'

log() { echo -e "${CYAN}[$(date '+%H:%M:%S')]${NC} $1"; }
success() { echo -e "${GREEN}✓${NC} $1"; }

log "Building Docker image (Ubuntu Rolling + ORT Latest)..."

# Build the image calling the target 'export' to ensure we get the final scratch layer if needed
docker build \
	-f ladspa/Dockerfile.minimal-ort \
	-t gtcrn-ort-builder \
	--target export \
	"$@" \
	.

log "Extracting libraries from container..."

# Create a temporary container from the image
CONTAINER_ID=$(docker create gtcrn-ort-builder)

# Clear old files to ensure we have fresh ones
rm -rf ladspa/onnxruntime-minimal
mkdir -p ladspa/onnxruntime-minimal

# Copy files from the container
# The Dockerfile copies /build/install to /onnxruntime-minimal in the scratch image
docker cp "${CONTAINER_ID}:/onnxruntime-minimal/lib" ladspa/onnxruntime-minimal/
docker cp "${CONTAINER_ID}:/onnxruntime-minimal/include" ladspa/onnxruntime-minimal/

# Copy config and converted ORT model
docker cp "${CONTAINER_ID}:/required_ops.config" ladspa/
# Make sure target dir exists (it is in root/stream/onnx_models)
docker cp "${CONTAINER_ID}:/model.ort" stream/onnx_models/gtcrn_simple.ort
docker cp "${CONTAINER_ID}:/model_full.ort" stream/onnx_models/gtcrn.ort

# Clean up
docker rm "${CONTAINER_ID}" >/dev/null

success "Libraries extracted to ladspa/onnxruntime-minimal"
ls -lh ladspa/onnxruntime-minimal/lib/

echo ""
echo "Now run: cd ladspa && ./build.sh minimal"
