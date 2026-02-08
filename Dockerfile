FROM python:3.11-slim

# Install system deps
RUN apt-get update && apt-get install -y \
    curl xz-utils git build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install Node.js 22 for clawdbot
ENV NODE_DIR=/root/nodejs
RUN mkdir -p $NODE_DIR && \
    ARCH=$(dpkg --print-architecture) && \
    if [ "$ARCH" = "amd64" ]; then NODE_ARCH="x64"; else NODE_ARCH="arm64"; fi && \
    curl -sL https://nodejs.org/dist/v22.12.0/node-v22.12.0-linux-${NODE_ARCH}.tar.xz -o /tmp/node.tar.xz && \
    tar xf /tmp/node.tar.xz -C /tmp && \
    cp -r /tmp/node-v22.12.0-linux-${NODE_ARCH}/* $NODE_DIR/ && \
    rm -rf /tmp/node*

ENV PATH="$NODE_DIR/bin:/root/.clawdbot-bin:$PATH"

# Try to install clawdbot (may or may not be available)
RUN npm install -g clawdbot 2>/dev/null || echo "clawdbot npm package not available - will use built-in AI tools instead"

# Create workspace
RUN mkdir -p /root/clawd /root/.clawdbot

# Set up app
WORKDIR /app
COPY backend/requirements_railway.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY backend/ /app/

# Rename server_railway.py to server.py for deployment
RUN cp /app/server_railway.py /app/server.py

EXPOSE 8000

CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8000"]
