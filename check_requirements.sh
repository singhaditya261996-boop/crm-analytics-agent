#!/bin/bash

echo ""
echo "======================================"
echo "  CRM Agent — Pre-Flight Checker"
echo "======================================"
echo ""

PASS=0
FAIL=0
WARN=0

pass() { echo "  ✅ $1"; PASS=$((PASS+1)); }
fail() { echo "  ❌ $1"; echo "     → Fix: $2"; echo ""; FAIL=$((FAIL+1)); }
warn() { echo "  ⚠️  $1"; echo "     → Note: $2"; echo ""; WARN=$((WARN+1)); }

echo "Checking Docker..."

# Docker installed
if command -v docker > /dev/null 2>&1; then
    pass "Docker is installed ($(docker --version | cut -d' ' -f3 | tr -d ','))"
else
    fail "Docker is not installed" "Download from: docker.com/products/docker-desktop"
fi

# Docker running
if docker info > /dev/null 2>&1; then
    pass "Docker Desktop is running"
else
    fail "Docker Desktop is not running" "Open Docker Desktop and wait for it to show 'Running'"
fi

# docker-compose available
if command -v docker-compose > /dev/null 2>&1; then
    pass "docker-compose is available"
else
    fail "docker-compose not found" "Reinstall Docker Desktop — it should be included"
fi

echo ""
echo "Checking system resources..."

# RAM check
if [[ "$OSTYPE" == "darwin"* ]]; then
    RAM_BYTES=$(sysctl -n hw.memsize)
    RAM_GB=$((RAM_BYTES / 1024 / 1024 / 1024))
elif [[ "$OSTYPE" == "linux-gnu"* ]]; then
    RAM_GB=$(( $(grep MemTotal /proc/meminfo | awk '{print $2}') / 1024 / 1024 ))
else
    RAM_BYTES=$(wmic computersystem get TotalPhysicalMemory /value 2>/dev/null | grep = | cut -d= -f2 | tr -d '\r')
    RAM_GB=$((RAM_BYTES / 1024 / 1024 / 1024))
fi

if [ "$RAM_GB" -ge 16 ]; then
    pass "RAM: ${RAM_GB}GB — excellent, full performance expected"
elif [ "$RAM_GB" -ge 8 ]; then
    warn "RAM: ${RAM_GB}GB — minimum met" \
         "Performance may be slow. Close other apps before running."
elif [ "$RAM_GB" -ge 4 ]; then
    warn "RAM: ${RAM_GB}GB — below recommended" \
         "Switch to smaller model after setup: run 'make switch-small'"
else
    fail "RAM: ${RAM_GB}GB — insufficient" \
         "Need at least 4GB free RAM. This machine may not run the agent well."
fi

# Disk space check (need 12GB free)
if [[ "$OSTYPE" == "darwin"* ]]; then
    FREE_GB=$(df -g / | tail -1 | awk '{print $4}')
elif [[ "$OSTYPE" == "linux-gnu"* ]]; then
    FREE_GB=$(df -BG / | tail -1 | awk '{print $4}' | tr -d 'G')
else
    FREE_GB=$(wmic logicaldisk where "DeviceID='C:'" get FreeSpace /value 2>/dev/null | grep = | cut -d= -f2 | tr -d '\r')
    FREE_GB=$((FREE_GB / 1024 / 1024 / 1024))
fi

if [ "$FREE_GB" -ge 15 ]; then
    pass "Disk space: ${FREE_GB}GB free — sufficient"
elif [ "$FREE_GB" -ge 10 ]; then
    warn "Disk space: ${FREE_GB}GB free — tight" \
         "Models take ~9GB. Consider freeing more space if download fails."
else
    fail "Disk space: ${FREE_GB}GB free — insufficient" \
         "Need at least 10GB free. Clear disk space before continuing."
fi

echo ""
echo "Checking network..."

# Port 8501
if lsof -i :8501 > /dev/null 2>&1 || netstat -an 2>/dev/null | grep -q ":8501 "; then
    fail "Port 8501 is already in use" \
         "Mac/Linux: lsof -ti:8501 | xargs kill -9   Windows: netstat -ano | findstr :8501 then taskkill /PID [number] /F"
else
    pass "Port 8501 is available"
fi

# Port 11434 (Ollama)
if lsof -i :11434 > /dev/null 2>&1; then
    warn "Port 11434 is in use" \
         "Ollama may already be running locally — this is usually fine"
else
    pass "Port 11434 is available"
fi

echo ""
echo "Checking project files..."

# Required files exist
for file in app.py requirements.txt docker-compose.yml entrypoint.sh config/settings.yaml; do
    if [ -f "$file" ]; then
        pass "Found: $file"
    else
        fail "Missing: $file" "Re-download the project folder from your team lead"
    fi
done

# Uploads folder
if [ -d "data/uploads" ]; then
    pass "Uploads folder exists"
else
    mkdir -p data/uploads
    pass "Uploads folder created"
fi

echo ""
echo "======================================"
printf "  Results: ✅ %d passed  " "$PASS"
printf "❌ %d failed  " "$FAIL"
printf "⚠️  %d warnings\n" "$WARN"
echo "======================================"
echo ""

if [ $FAIL -eq 0 ] && [ $WARN -eq 0 ]; then
    echo "  🚀 Everything looks great!"
    echo "     Run: docker-compose up"
    echo "     Then open: http://localhost:8501"
elif [ $FAIL -eq 0 ]; then
    echo "  🟡 Checks passed with warnings — review notes above"
    echo "     Run: docker-compose up"
    echo "     Then open: http://localhost:8501"
else
    echo "  ⛔ Fix the failed checks above before continuing"
    echo "     Run this script again after fixing to verify"
    exit 1
fi
echo ""
