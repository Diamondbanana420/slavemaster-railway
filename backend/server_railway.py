"""
SlavemasterBot Backend - Railway Port
Ported from Emergent to Railway standalone deployment.
Changes: Emergent auth → API key auth, supervisor → subprocess, WhatsApp removed
All dropship automation preserved intact.
"""
from fastapi import FastAPI, APIRouter, HTTPException, WebSocket, WebSocketDisconnect, Request, Response
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from dotenv import load_dotenv
from starlette.middleware.cors import CORSMiddleware
from starlette.websockets import WebSocketState
from motor.motor_asyncio import AsyncIOMotorClient
import os
import logging
import json
import secrets
import subprocess
import asyncio
import httpx
import shutil
from pathlib import Path
from pydantic import BaseModel, Field, ConfigDict
from typing import List, Optional
import uuid
from datetime import datetime, timezone, timedelta

ROOT_DIR = Path(__file__).parent
load_dotenv(ROOT_DIR / '.env')

# MongoDB connection
mongo_url = os.environ.get('MONGO_URL', os.environ.get('MONGODB_URL', 'mongodb://localhost:27017'))
db_name = os.environ.get('DB_NAME', 'slavemaster')
client = AsyncIOMotorClient(mongo_url)
db = client[db_name]

# Auth config
API_KEY = os.environ.get('API_KEY', secrets.token_hex(32))
AUTHORIZED_USERS = os.environ.get('AUTHORIZED_USERS', 'ben,steve374').split(',')

app = FastAPI(title="SlavemasterBot API", version="2.0.0")
api_router = APIRouter(prefix="/api")

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ClawdBot Gateway config
MOLTBOT_PORT = int(os.environ.get('CLAWDBOT_PORT', 18789))
MOLTBOT_CONTROL_PORT = 18791
CONFIG_DIR = os.path.expanduser("~/.clawdbot")
CONFIG_FILE = os.path.join(CONFIG_DIR, "clawdbot.json")
WORKSPACE_DIR = os.path.expanduser("~/clawd")
NODE_DIR = "/root/nodejs"
CLAWDBOT_DIR = "/root/.clawdbot-bin"

# Gateway state
gateway_process = None
gateway_state = {
    "token": None,
    "provider": None,
    "started_at": None,
    "pid": None
}

# Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# ============== Pydantic Models ==============

class StatusCheck(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    client_name: str
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

class StatusCheckCreate(BaseModel):
    client_name: str

class OpenClawStartRequest(BaseModel):
    apiKey: Optional[str] = None
    provider: str = "openai"

class OpenClawStartResponse(BaseModel):
    ok: bool
    controlUrl: Optional[str] = None
    token: Optional[str] = None
    message: str

class OpenClawStatusResponse(BaseModel):
    running: bool
    pid: Optional[int] = None
    provider: Optional[str] = None
    started_at: Optional[str] = None
    controlUrl: Optional[str] = None


# ============== Simple API Key Auth ==============

async def verify_api_key(request: Request) -> bool:
    """Verify API key from header or cookie"""
    # Check header
    auth_header = request.headers.get("Authorization")
    if auth_header:
        if auth_header.startswith("Bearer "):
            token = auth_header.split(" ")[1]
            if token == API_KEY:
                return True
    
    # Check cookie
    cookie_key = request.cookies.get("api_key")
    if cookie_key == API_KEY:
        return True
    
    # Check query param (for dashboard access)
    key = request.query_params.get("key")
    if key == API_KEY:
        return True
    
    return False


async def require_auth(request: Request):
    """Require valid API key"""
    if not await verify_api_key(request):
        raise HTTPException(status_code=401, detail="Invalid API key")
    return True


# ============== Auth Endpoints ==============

@api_router.post("/auth/login")
async def login(request: Request, response: Response):
    """Simple API key login"""
    body = await request.json()
    key = body.get("api_key", "")
    
    if key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API key")
    
    response.set_cookie(
        key="api_key",
        value=API_KEY,
        httponly=True,
        secure=True,
        samesite="none",
        path="/",
        max_age=30 * 24 * 60 * 60  # 30 days
    )
    
    return {"ok": True, "message": "Logged in"}


@api_router.get("/auth/me")
async def get_me(request: Request):
    """Check auth status"""
    if await verify_api_key(request):
        return {"authenticated": True, "user": "admin"}
    raise HTTPException(status_code=401, detail="Not authenticated")


@api_router.post("/auth/logout")
async def logout(response: Response):
    """Logout"""
    response.delete_cookie(key="api_key", path="/")
    return {"ok": True}


@api_router.get("/auth/instance")
async def get_instance_status():
    """Instance status - always accessible"""
    return {"locked": False, "railway": True, "version": "2.0.0"}


# ============== ClawdBot Gateway ==============

def get_clawdbot_command():
    """Find clawdbot executable"""
    paths = [
        f"{CLAWDBOT_DIR}/clawdbot",
        f"{NODE_DIR}/bin/clawdbot",
        shutil.which("clawdbot"),
        "/usr/local/bin/clawdbot",
    ]
    for p in paths:
        if p and os.path.exists(p):
            return p
    return shutil.which("clawdbot")


def install_clawdbot():
    """Install Node.js and clawdbot"""
    logger.info("Installing Node.js and clawdbot...")
    
    # Install Node.js if not present
    if not os.path.exists(f"{NODE_DIR}/bin/node"):
        os.makedirs(NODE_DIR, exist_ok=True)
        
        import platform
        arch = platform.machine()
        node_arch = "x64" if arch == "x86_64" else "arm64"
        node_version = "22.22.0"
        
        cmds = [
            f"cd /tmp && curl -sL https://nodejs.org/dist/v{node_version}/node-v{node_version}-linux-{node_arch}.tar.xz -o node.tar.xz",
            f"cd /tmp && tar xf node.tar.xz && cp -r node-v{node_version}-linux-{node_arch}/* {NODE_DIR}/",
            f"rm -rf /tmp/node.tar.xz /tmp/node-v{node_version}-linux-{node_arch}",
        ]
        
        for cmd in cmds:
            result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=120)
            if result.returncode != 0:
                logger.error(f"Failed: {cmd}\n{result.stderr}")
                return False
    
    # Install clawdbot
    os.environ["PATH"] = f"{NODE_DIR}/bin:{CLAWDBOT_DIR}:{os.environ.get('PATH', '')}"
    
    os.makedirs(CLAWDBOT_DIR, exist_ok=True)
    result = subprocess.run(
        f"{NODE_DIR}/bin/npm install -g @anthropic/clawdbot 2>/dev/null || {NODE_DIR}/bin/npm install -g clawdbot 2>/dev/null || echo 'clawdbot not available as npm package'",
        shell=True, capture_output=True, text=True, timeout=120,
        env={**os.environ, "PATH": f"{NODE_DIR}/bin:{os.environ.get('PATH', '')}"}
    )
    logger.info(f"clawdbot install: {result.stdout[:200]}")
    
    return get_clawdbot_command() is not None


def generate_token():
    return secrets.token_hex(32)


def create_clawdbot_config(token=None, api_key=None, provider="openai"):
    """Create clawdbot config for the chosen provider"""
    os.makedirs(CONFIG_DIR, exist_ok=True)
    os.makedirs(WORKSPACE_DIR, exist_ok=True)
    
    # Load existing or create new
    config = {}
    if os.path.exists(CONFIG_FILE):
        try:
            with open(CONFIG_FILE) as f:
                config = json.load(f)
        except:
            pass
    
    final_token = token or config.get("gateway", {}).get("auth", {}).get("token") or generate_token()
    
    config["gateway"] = {
        "mode": "local",
        "port": MOLTBOT_PORT,
        "bind": "lan",
        "auth": {"mode": "token", "token": final_token},
        "controlUi": {"enabled": True, "allowInsecureAuth": True}
    }
    
    if "models" not in config:
        config["models"] = {"mode": "merge", "providers": {}}
    config["models"]["mode"] = "merge"
    if "providers" not in config["models"]:
        config["models"]["providers"] = {}
    
    if "agents" not in config:
        config["agents"] = {"defaults": {}}
    config["agents"]["defaults"]["workspace"] = WORKSPACE_DIR
    
    # Configure provider
    if provider == "openai":
        key = api_key or os.environ.get('OPENAI_API_KEY', '')
        config["models"]["providers"]["openai"] = {
            "baseUrl": "https://api.openai.com/v1/",
            "apiKey": key,
            "api": "openai-completions",
            "models": [
                {"id": "gpt-4o", "name": "GPT-4o", "reasoning": False,
                 "input": ["text", "image"],
                 "cost": {"input": 0.0000025, "output": 0.00001},
                 "contextWindow": 128000, "maxTokens": 16384},
                {"id": "o4-mini-2025-04-16", "name": "o4-mini", "reasoning": True,
                 "input": ["text", "image"],
                 "cost": {"input": 0.0000011, "output": 0.0000044},
                 "contextWindow": 200000, "maxTokens": 100000},
            ]
        }
        config["agents"]["defaults"]["model"] = {"primary": "openai/gpt-4o"}
    
    elif provider == "anthropic":
        key = api_key or os.environ.get('ANTHROPIC_API_KEY', '')
        config["models"]["providers"]["anthropic"] = {
            "baseUrl": "https://api.anthropic.com",
            "apiKey": key,
            "api": "anthropic-messages",
            "models": [
                {"id": "claude-sonnet-4-5-20250514", "name": "Claude Sonnet 4.5",
                 "input": ["text", "image"],
                 "cost": {"input": 0.000003, "output": 0.000015},
                 "contextWindow": 200000, "maxTokens": 64000},
            ]
        }
        config["agents"]["defaults"]["model"] = {"primary": "anthropic/claude-sonnet-4-5-20250514"}
    
    with open(CONFIG_FILE, "w") as f:
        json.dump(config, f, indent=2)
    
    logger.info(f"ClawdBot config written for provider: {provider}")
    return final_token


async def start_gateway(api_key=None, provider="openai"):
    """Start the ClawdBot gateway process"""
    global gateway_process, gateway_state
    
    # Check if already running
    if gateway_process and gateway_process.poll() is None:
        logger.info("Gateway already running")
        return gateway_state["token"]
    
    # Find clawdbot
    cmd = get_clawdbot_command()
    if not cmd:
        if not install_clawdbot():
            raise HTTPException(status_code=500, detail="ClawdBot not available. The clawdbot npm package may not be publicly accessible.")
        cmd = get_clawdbot_command()
        if not cmd:
            raise HTTPException(status_code=500, detail="ClawdBot installation failed")
    
    # Create config
    token = create_clawdbot_config(api_key=api_key, provider=provider)
    
    # Start process
    env = {**os.environ, "PATH": f"{NODE_DIR}/bin:{CLAWDBOT_DIR}:{os.environ.get('PATH', '')}"}
    
    gateway_process = subprocess.Popen(
        [cmd, "gateway"],
        cwd=WORKSPACE_DIR,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE
    )
    
    gateway_state["token"] = token
    gateway_state["provider"] = provider
    gateway_state["started_at"] = datetime.now(timezone.utc).isoformat()
    gateway_state["pid"] = gateway_process.pid
    
    # Wait for ready
    async with httpx.AsyncClient() as http_client:
        for _ in range(30):
            try:
                resp = await http_client.get(f"http://127.0.0.1:{MOLTBOT_PORT}/", timeout=2.0)
                if resp.status_code == 200:
                    logger.info(f"ClawdBot gateway ready on port {MOLTBOT_PORT}")
                    
                    # Persist config
                    await db.gateway_config.update_one(
                        {"_id": "config"},
                        {"$set": {"should_run": True, "provider": provider, "token": token,
                                  "started_at": gateway_state["started_at"]}},
                        upsert=True
                    )
                    return token
            except:
                pass
            await asyncio.sleep(1)
    
    # Check if process died
    if gateway_process.poll() is not None:
        stderr = gateway_process.stderr.read().decode()[:500]
        raise HTTPException(status_code=500, detail=f"Gateway process exited: {stderr}")
    
    raise HTTPException(status_code=500, detail="Gateway did not become ready in 30s")


def check_gateway_running():
    """Check if gateway process is alive"""
    global gateway_process
    return gateway_process is not None and gateway_process.poll() is None


# ============== OpenClaw API Endpoints ==============

@api_router.get("/")
async def root():
    return {
        "message": "SlavemasterBot API",
        "version": "2.0.0",
        "platform": "Railway",
        "dropship": True,
        "gateway": check_gateway_running()
    }


@api_router.post("/openclaw/start", response_model=OpenClawStartResponse)
async def start_openclaw(request: OpenClawStartRequest, req: Request):
    """Start the ClawdBot gateway"""
    await require_auth(req)
    
    if request.provider not in ["openai", "anthropic"]:
        raise HTTPException(status_code=400, detail="Invalid provider. Use 'openai' or 'anthropic'")
    
    if not request.apiKey and not os.environ.get(f'{request.provider.upper()}_API_KEY'):
        raise HTTPException(status_code=400, detail=f"API key required for {request.provider}")
    
    try:
        token = await start_gateway(request.apiKey, request.provider)
        return OpenClawStartResponse(
            ok=True,
            controlUrl="/api/openclaw/ui/",
            token=token,
            message=f"OpenClaw started with {request.provider}"
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to start gateway: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@api_router.get("/openclaw/status", response_model=OpenClawStatusResponse)
async def get_openclaw_status():
    """Get gateway status"""
    running = check_gateway_running()
    if running:
        return OpenClawStatusResponse(
            running=True,
            pid=gateway_state["pid"],
            provider=gateway_state["provider"],
            started_at=gateway_state["started_at"],
            controlUrl="/api/openclaw/ui/"
        )
    return OpenClawStatusResponse(running=False)


@api_router.post("/openclaw/stop")
async def stop_openclaw(request: Request):
    """Stop the gateway"""
    await require_auth(request)
    global gateway_process, gateway_state
    
    if gateway_process and gateway_process.poll() is None:
        gateway_process.terminate()
        try:
            gateway_process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            gateway_process.kill()
    
    gateway_process = None
    gateway_state = {"token": None, "provider": None, "started_at": None, "pid": None}
    
    await db.gateway_config.update_one(
        {"_id": "config"},
        {"$set": {"should_run": False}},
        upsert=True
    )
    
    return {"ok": True, "message": "OpenClaw stopped"}


@api_router.get("/openclaw/token")
async def get_openclaw_token(request: Request):
    """Get gateway token"""
    await require_auth(request)
    if not check_gateway_running():
        raise HTTPException(status_code=404, detail="OpenClaw not running")
    return {"token": gateway_state.get("token")}


# ============== OpenClaw Proxy ==============

@api_router.api_route("/openclaw/ui/{path:path}", methods=["GET", "POST", "PUT", "DELETE", "PATCH", "OPTIONS", "HEAD"])
async def proxy_openclaw_ui(request: Request, path: str = ""):
    """Proxy to ClawdBot Control UI"""
    if not check_gateway_running():
        return HTMLResponse("<html><body><h1>OpenClaw not running</h1><p>Start it via /api/openclaw/start</p></body></html>", status_code=503)
    
    target_url = f"http://127.0.0.1:{MOLTBOT_PORT}/{path}"
    if request.url.query:
        target_url += f"?{request.url.query}"
    
    try:
        body = await request.body()
        headers = dict(request.headers)
        headers.pop("host", None)
        
        if gateway_state.get("token"):
            headers["Authorization"] = f"Bearer {gateway_state['token']}"
        
        async with httpx.AsyncClient() as http_client:
            resp = await http_client.request(
                method=request.method,
                url=target_url,
                headers=headers,
                content=body,
                timeout=30.0
            )
            
            response_headers = dict(resp.headers)
            response_headers.pop("transfer-encoding", None)
            response_headers.pop("content-encoding", None)
            
            return Response(
                content=resp.content,
                status_code=resp.status_code,
                headers=response_headers,
                media_type=resp.headers.get("content-type")
            )
    except Exception as e:
        return HTMLResponse(f"<html><body><h1>Proxy Error</h1><p>{e}</p></body></html>", status_code=502)


@api_router.websocket("/openclaw/ws")
async def websocket_proxy(websocket: WebSocket):
    """WebSocket proxy to ClawdBot"""
    if not check_gateway_running():
        await websocket.close(code=1013)
        return
    
    await websocket.accept()
    
    try:
        import websockets
        ws_url = f"ws://127.0.0.1:{MOLTBOT_PORT}/ws"
        extra_headers = {}
        if gateway_state.get("token"):
            extra_headers["Authorization"] = f"Bearer {gateway_state['token']}"
        
        async with websockets.connect(ws_url, extra_headers=extra_headers) as ws:
            async def client_to_gateway():
                try:
                    while True:
                        data = await websocket.receive_text()
                        await ws.send(data)
                except WebSocketDisconnect:
                    pass
            
            async def gateway_to_client():
                try:
                    async for msg in ws:
                        await websocket.send_text(msg)
                except:
                    pass
            
            await asyncio.gather(client_to_gateway(), gateway_to_client())
    except Exception as e:
        logger.error(f"WebSocket proxy error: {e}")
    finally:
        try:
            await websocket.close()
        except:
            pass


# ============== Status Endpoints ==============

@api_router.post("/status", response_model=StatusCheck)
async def create_status_check(input: StatusCheckCreate):
    check = StatusCheck(client_name=input.client_name)
    await db.status_checks.insert_one(check.model_dump())
    return check

@api_router.get("/status", response_model=List[StatusCheck])
async def get_status_checks():
    checks = await db.status_checks.find().sort("timestamp", -1).limit(50).to_list(50)
    return [StatusCheck(**c) for c in checks]


# ============== Dropshipping Automation ==============

# Import dropship system
from dropship.system import DropshipSystem
dropship_system = DropshipSystem()


@api_router.get("/dropship/status")
async def get_dropship_status():
    """Get dropshipping system status"""
    return {
        "initialized": dropship_system.initialized,
        "scheduler_running": dropship_system.scheduler.is_running if hasattr(dropship_system.scheduler, 'is_running') else False,
        "components": {
            "shopify": dropship_system.shopify is not None,
            "site_monitor": dropship_system.site_monitor is not None,
            "competitor_monitor": dropship_system.competitor_monitor is not None,
            "analytics": dropship_system.analytics is not None,
            "pricing": dropship_system.pricing is not None,
            "inventory": dropship_system.inventory is not None,
            "fulfillment": dropship_system.fulfillment is not None,
        }
    }


@api_router.post("/dropship/initialize")
async def initialize_dropship(request: Request):
    """Initialize the dropshipping system"""
    await require_auth(request)
    dropship_system.db = db
    await dropship_system.initialize()
    return {"ok": True, "message": "Dropship system initialized"}


@api_router.get("/dropship/site-audit")
async def run_site_audit():
    return await dropship_system.site_monitor.full_audit()


@api_router.get("/dropship/competitors")
async def check_competitors():
    return await dropship_system.competitor_monitor.check_all()


@api_router.post("/dropship/competitors/add")
async def add_competitor(request: Request):
    body = await request.json()
    return await dropship_system.competitor_monitor.add_competitor(body.get("name"), body.get("url"))


@api_router.get("/dropship/reports/daily")
async def get_daily_report():
    return await dropship_system.reports.generate_daily()


@api_router.get("/dropship/reports/weekly")
async def get_weekly_report():
    return await dropship_system.reports.generate_weekly()


@api_router.post("/dropship/pricing/analyze")
async def analyze_pricing(request: Request):
    body = await request.json()
    return await dropship_system.pricing.analyze(body.get("products", []))


@api_router.post("/dropship/inventory/check")
async def check_inventory(request: Request):
    body = await request.json()
    return await dropship_system.inventory.check(body.get("products", []))


@api_router.post("/dropship/profit/calculate")
async def calculate_profit(request: Request):
    body = await request.json()
    return await dropship_system.pricing.calculate_profit(body)


@api_router.post("/dropship/funnel/analyze")
async def analyze_funnel(request: Request):
    body = await request.json()
    return await dropship_system.analytics.analyze_funnel(body)


@api_router.get("/dropship/alerts")
async def get_alerts(unacknowledged: bool = False, severity: str = None):
    return await dropship_system.alerts.get_alerts(unacknowledged=unacknowledged, severity=severity)


@api_router.post("/dropship/alerts/{alert_id}/acknowledge")
async def acknowledge_alert(alert_id: str):
    return await dropship_system.alerts.acknowledge(alert_id)


@api_router.get("/dropship/scheduler/status")
async def get_scheduler_status():
    return dropship_system.scheduler.get_status()


@api_router.post("/dropship/scheduler/start")
async def start_scheduler(request: Request):
    await require_auth(request)
    return await dropship_system.scheduler.start()


@api_router.post("/dropship/scheduler/stop")
async def stop_scheduler(request: Request):
    await require_auth(request)
    return dropship_system.scheduler.stop()


@api_router.post("/dropship/scheduler/run/{task_name}")
async def run_task(task_name: str, request: Request):
    await require_auth(request)
    return await dropship_system.scheduler.run_now(task_name)


# ============== Health & System ==============

@api_router.get("/health")
async def health_check():
    """System health check"""
    checks = {
        "api": True,
        "mongodb": False,
        "gateway": check_gateway_running(),
        "dropship": dropship_system.initialized,
    }
    
    try:
        await db.command("ping")
        checks["mongodb"] = True
    except:
        pass
    
    return {
        "status": "healthy" if all([checks["api"], checks["mongodb"]]) else "degraded",
        "checks": checks,
        "version": "2.0.0",
        "platform": "railway",
        "timestamp": datetime.now(timezone.utc).isoformat()
    }


# ============== Startup / Shutdown ==============

@app.on_event("startup")
async def startup_event():
    """Server startup"""
    global gateway_state
    logger.info("SlavemasterBot API starting on Railway...")
    
    # Initialize dropship system with db
    dropship_system.db = db
    
    # Log API key for first-time setup
    logger.info(f"API Key: {API_KEY[:8]}...{API_KEY[-4:]}")
    
    # Check for clawdbot
    cmd = get_clawdbot_command()
    if cmd:
        logger.info(f"ClawdBot found: {cmd}")
    else:
        logger.info("ClawdBot not installed - will install on first /openclaw/start")
    
    # Auto-start gateway if configured
    config = await db.gateway_config.find_one({"_id": "config"})
    if config and config.get("should_run"):
        logger.info("Auto-starting gateway from saved config...")
        try:
            await start_gateway(provider=config.get("provider", "openai"))
            logger.info("Gateway auto-started successfully")
        except Exception as e:
            logger.warning(f"Gateway auto-start failed: {e}")
    
    logger.info("SlavemasterBot API ready!")


@app.on_event("shutdown")
async def shutdown_event():
    """Graceful shutdown"""
    global gateway_process
    
    if gateway_process and gateway_process.poll() is None:
        logger.info("Stopping gateway on shutdown...")
        gateway_process.terminate()
        try:
            gateway_process.wait(timeout=5)
        except:
            gateway_process.kill()
    
    client.close()
    logger.info("SlavemasterBot API shut down")


# Mount router
app.include_router(api_router)

# Serve frontend static files if present
frontend_dir = ROOT_DIR.parent / "frontend" / "build"
if frontend_dir.exists():
    app.mount("/", StaticFiles(directory=str(frontend_dir), html=True), name="frontend")
    logger.info(f"Serving frontend from {frontend_dir}")


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
