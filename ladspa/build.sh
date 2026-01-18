#!/bin/bash
# GTCRN LADSPA ORT Build Script
# Supports dynamic (system), static (downloaded MS), and minimal (docker) strategies.

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

print_usage() {
	echo "Usage: $0 [dynamic|static|minimal]"
	echo ""
	echo "Options:"
	echo "  dynamic  - Use system installed ONNX Runtime (libonnxruntime.so)."
	echo "             Fast build, requires 'onnxruntime' package on the system."
	echo ""
	echo "  static   - Use ONNX Runtime downloaded from Microsoft (Bundled)."
	echo "             Downloads the 'Full' runtime (~50MB) and bundles it."
	echo "             Note: This is 'static' in the sense of 'bundled dependencies',"
	echo "             but technically links dynamically to the bundled .so file."
	echo ""
	echo "  minimal  - Use Minimal ONNX Runtime built via Docker (Statically Linked)."
	echo "             Requires running ./build-minimal-docker.sh first."
	echo "             Produces a single, small, dependency-free .so plugin."
	echo ""
	echo "Default: dynamic"
}

build_dynamic() {
	echo -e "${GREEN}Building DYNAMIC (System ORT)...${NC}"
	echo "Requires 'onnxruntime' installed on the system."
	echo ""

	CARGO_BUILD_JOBS=$(nproc)
	export CARGO_BUILD_JOBS
	cargo build --release --features dynamic --no-default-features

	BINARY="target/release/libgtcrn_ladspa_ort.so"
	if [ -f "$BINARY" ]; then
		SIZE=$(du -h "$BINARY" | cut -f1)
		echo ""
		echo -e "${GREEN}✓ Dynamic build successful!${NC}"
		echo "  Binary: $BINARY"
		echo "  Size: $SIZE"
	fi
}

build_static() {
	echo -e "${YELLOW}Building STATIC (Microsoft Downloaded)...${NC}"
	echo "Downloading/Using prebuilt ONNX Runtime from Microsoft."
	echo ""

	CARGO_BUILD_JOBS=$(nproc)
	export CARGO_BUILD_JOBS
	# 'download' feature enables ort/download-binaries
	cargo build --release --features download --no-default-features

	BINARY="target/release/libgtcrn_ladspa_ort.so"
	if [ -f "$BINARY" ]; then
		SIZE=$(du -h "$BINARY" | cut -f1)
		echo ""
		echo -e "${GREEN}✓ Static (Download) build successful!${NC}"
		echo "  Binary: $BINARY"
		echo "  Size: $SIZE"
		echo "  Note: The libonnxruntime.so is bundled/downloaded by the build process."
	fi
}

build_minimal() {
	echo -e "${YELLOW}Building MINIMAL (Docker)...${NC}"
	echo "Linking statically against minimal runtime from Docker."

	if [ ! -d "onnxruntime-minimal/lib" ]; then
		echo -e "${RED}Error: onnxruntime-minimal/lib not found.${NC}"
		echo "Please run ./build-minimal-docker.sh first."
		exit 1
	fi

	# Create unified libonnxruntime.a to satisfy ort-sys
	echo "Creating unified libonnxruntime.a..."
	cd onnxruntime-minimal/lib

	echo "CREATE libonnxruntime.a" >lib_script.mri
	for lib in libonnxruntime_*.a libonnx.a libonnx_proto.a; do
		if [ -f "$lib" ]; then
			echo "ADDLIB $lib" >>lib_script.mri
		fi
	done
	echo "SAVE" >>lib_script.mri
	echo "END" >>lib_script.mri

	ar -M <lib_script.mri
	rm lib_script.mri
	cd ../..

	echo "Embedding ONNX Runtime statically..."

	ORT_STRATEGY=system
	export ORT_STRATEGY
	ORT_LIB_LOCATION=$(pwd)/onnxruntime-minimal/lib
	export ORT_LIB_LOCATION
	CARGO_BUILD_JOBS=$(nproc)
	export CARGO_BUILD_JOBS

	# 'static' feature enables the logic in build.rs to link against these libs
	cargo build --release --features static --no-default-features

	if [ -f "target/release/libgtcrn_ladspa_ort.so" ]; then
		SIZE=$(du -h target/release/libgtcrn_ladspa_ort.so | cut -f1)
		echo ""
		echo -e "${GREEN}✓ Minimal build successful!${NC}"
		echo "  Plugin: target/release/libgtcrn_ladspa_ort.so ($SIZE)"
		echo "  Dependencies: Single file (Static Link)"
	else
		echo -e "${RED}Build failed!${NC}"
		exit 1
	fi
}

# Main
case "${1:-dynamic}" in
dynamic)
	build_dynamic
	;;
static)
	build_static
	;;
minimal)
	build_minimal
	;;
-h | --help | help)
	print_usage
	;;
*)
	echo -e "${RED}Unknown option: $1${NC}"
	print_usage
	exit 1
	;;
esac
