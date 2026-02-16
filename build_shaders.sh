#!/usr/bin/env bash
set -e  # stop on first error

SLANGC="./cmake-build-debug/_deps/Slang-linux-x86_64-2026.1.1/bin/slangc"
SHADER="rt2d_triangles.slang"

echo "== Slang RT shader build =="

# Check slangc exists
if [ ! -f "$SLANGC" ]; then
  echo "Error: slangc not found at $SLANGC"
  exit 1
fi

# Check shader exists
if [ ! -f "$SHADER" ]; then
  echo "Error: shader file $SHADER not found"
  exit 1
fi

COMMON_FLAGS="-profile sm_6_6 -target spirv -fvk-use-scalar-layout"

echo "Compiling raygen..."
$SLANGC $SHADER $COMMON_FLAGS \
  -entry raygenMain -stage raygeneration -o raygen.spv

echo "Compiling miss..."
$SLANGC $SHADER $COMMON_FLAGS \
  -entry missMain -stage miss -o miss.spv

echo "Compiling closest hit..."
$SLANGC $SHADER $COMMON_FLAGS \
  -entry chitMain -stage closesthit -o chit.spv

echo "== Done =="

ls -lh raygen.spv miss.spv chit.spv
