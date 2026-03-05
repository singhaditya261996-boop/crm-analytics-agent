#!/bin/bash
set -e

echo ""
echo "======================================"
echo "  CRM Analytics Agent — Starting Up"
echo "======================================"
echo ""

# Start Ollama service in background
echo "⚙️  Starting local AI service..."
ollama serve &
OLLAMA_PID=$!

# Wait for Ollama to be ready with timeout
echo "⏳ Waiting for AI service to be ready..."
TIMEOUT=30
ELAPSED=0
until curl -s http://localhost:11434/api/tags > /dev/null 2>&1; do
    sleep 2
    ELAPSED=$((ELAPSED + 2))
    if [ $ELAPSED -ge $TIMEOUT ]; then
        echo "❌ AI service failed to start within ${TIMEOUT}s — check Docker memory allocation"
        exit 1
    fi
done
echo "✅ AI service ready"
echo ""

# Pull main reasoning model if not already cached
if ! ollama list | grep -q "llama3.1:8b"; then
    echo "📥 Downloading main AI model (4.7GB)"
    echo "   This only happens once — subsequent starts are instant"
    echo "   Please wait, this may take 10-30 minutes depending on your connection..."
    ollama pull llama3.1:8b
    echo "✅ Main model ready"
else
    echo "✅ Main model already cached — skipping download"
fi

echo ""

# Pull vision model for screenshot/image support if not cached
if ! ollama list | grep -q "llava"; then
    echo "📥 Downloading vision model for image/screenshot support (4.5GB)"
    echo "   This also only happens once..."
    ollama pull llava
    echo "✅ Vision model ready"
else
    echo "✅ Vision model already cached — skipping download"
fi

echo ""
echo "======================================"
echo "  🚀 Launching CRM Analytics Agent"
echo "======================================"
echo ""
echo "  👉 Open your browser and go to:"
echo "     http://localhost:8501"
echo ""
echo "  📁 Drop your data files into:"
echo "     the 'uploads' folder in the project directory"
echo ""
echo "  🛑 To stop: press Ctrl+C in this window"
echo ""

# Launch Streamlit
streamlit run app.py \
    --server.address 0.0.0.0 \
    --server.port 8501 \
    --server.headless true \
    --server.fileWatcherType none \
    --server.maxUploadSize 500
