#!/bin/bash
# Manual square pass simulator launcher (bypasses control panel cache)

cd /home/balaji/paparazzi

echo "Starting components for bebop_square_pass_guided simulation..."

# Kill any existing processes
pkill -f "simsitl" 2>/dev/null
pkill -f "server" 2>/dev/null  
pkill -f "link" 2>/dev/null
pkill -f "gcs" 2>/dev/null
sleep 1

# Start Data Link
./sw/ground_segment/tmtc/link -udp -ping_period 100 &
LINK_PID=$!
sleep 1

# Start Server
./sw/ground_segment/tmtc/server -n &
SERVER_PID=$!
sleep 1

# Start GCS (no speech flag)
./sw/ground_segment/cockpit/gcs &
GCS_PID=$!
sleep 2

# Start Simulator
./sw/simulator/pprzsim-launch -a bebop_square_pass_guided -t nps &
SIM_PID=$!

echo "========================================" 
echo "Session started (NO joystick, NO GST spam, NO speech)"
echo "In GCS:"
echo "  1. Click 'Start Engine' (or press 'r')"
echo "  2. Click 'Takeoff' (or press 't')"  
echo "  3. Wait for altitude > 0.8m"
echo "  4. Click 'START' (or press 'g') to enable guided mode"
echo "========================================" 
echo "Press Ctrl+C to stop all processes"

# Wait and cleanup on exit
trap "kill $LINK_PID $SERVER_PID $GCS_PID $SIM_PID 2>/dev/null; exit" INT TERM
wait
