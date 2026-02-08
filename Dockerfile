FROM python:3.12-slim

# Install Node.js 22 + system deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl git build-essential && \
    curl -fsSL https://deb.nodesource.com/setup_22.x | bash - && \
    apt-get install -y nodejs && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# Create workspace dirs
RUN mkdir -p /root/.clawdbot /root/clawd /root/.clawdbot-bin

# Install clawdbot globally via npm
RUN npm install -g @anthropic-ai/claude-code 2>/dev/null || \
    npm install -g clawdbot 2>/dev/null || \
    echo "clawdbot will be installed on first use"

WORKDIR /app/backend

# Install Python deps
COPY backend/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy backend code
COPY backend/ .

# Expose port
ENV PORT=8080
EXPOSE 8080

# Run FastAPI
CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8080"]
