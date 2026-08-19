#!/bin/bash
set -e

# TinyGiant Service Installer
# Sets up TinyGiant as a persistent macOS LaunchAgent

TINYGIANT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
PLIST_SRC="$TINYGIANT_DIR/service/com.tinygiant.server.plist"
PLIST_DST="$HOME/Library/LaunchAgents/com.tinygiant.server.plist"
CONFIG_DIR="$HOME/.tinygiant"
LOG_DIR="$CONFIG_DIR/logs"
CONFIG_FILE="$CONFIG_DIR/config.json"
PYTHON="$(which python3)"
LABEL="com.tinygiant.server"

echo "TinyGiant Service Installer"
echo "============================"
echo "  TinyGiant dir: $TINYGIANT_DIR"
echo "  Python:        $PYTHON"
echo "  Config dir:    $CONFIG_DIR"
echo ""

# Check for existing service
if launchctl list 2>/dev/null | grep -q "$LABEL"; then
    echo "Stopping existing service..."
    launchctl unload "$PLIST_DST" 2>/dev/null || true
fi

# Create directories
mkdir -p "$CONFIG_DIR" "$LOG_DIR"

# Create config if it doesn't exist
if [ ! -f "$CONFIG_FILE" ]; then
    echo "Creating default config at $CONFIG_FILE"
    echo "  Edit this file to set your model and cache paths."
    cat > "$CONFIG_FILE" << 'CONF'
{
  "model": "~/models/Qwen3-30B-A3B-Q4_K_M.gguf",
  "cache": "~/.tinygiant/cache",
  "host": "0.0.0.0",
  "port": 8000,
  "pin": 48,
  "calibrate": 10,
  "model_name": "tinygiant-qwen3-30b"
}
CONF
    echo ""
    echo "  *** IMPORTANT: Edit $CONFIG_FILE before starting the service ***"
    echo "  Set 'model' to your GGUF model path"
    echo "  Set 'cache' to your NWS expert cache directory"
    echo ""
fi

# Build C library if needed
LIB="$TINYGIANT_DIR/tinygiant/libtinygiant.dylib"
if [ ! -f "$LIB" ]; then
    echo "Building libtinygiant..."
    make -C "$TINYGIANT_DIR" all
fi

# Generate plist from template
echo "Installing LaunchAgent..."
sed -e "s|__PYTHON__|$PYTHON|g" \
    -e "s|__HOME__|$HOME|g" \
    -e "s|__TINYGIANT_DIR__|$TINYGIANT_DIR|g" \
    "$PLIST_SRC" > "$PLIST_DST"

echo "  Installed: $PLIST_DST"

echo ""
echo "Service installed. Commands:"
echo ""
echo "  Start:   launchctl load $PLIST_DST"
echo "  Stop:    launchctl unload $PLIST_DST"
echo "  Status:  curl http://localhost:8000/health"
echo "  Logs:    tail -f $LOG_DIR/server.log"
echo "  Errors:  tail -f $LOG_DIR/server.err"
echo "  Config:  $CONFIG_FILE"
echo ""
echo "The service will start automatically on login."
echo "To start now: launchctl load $PLIST_DST"
