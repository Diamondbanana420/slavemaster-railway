from fastapi import FastAPI, APIRouter, HTTPException, WebSocket, WebSocketDisconnect, Request, Response
from fastapi.responses import HTMLResponse
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
import shutil
import httpx
import websockets
from websockets.exceptions import ConnectionClosed
from pathlib import Path
from pydantic import BaseModel, Field, ConfigDict
from typing import List, Optional
import uuid
from datetime import datetime, timezone, timedelta

# Gateway management (subprocess-based for Railway)
from gateway_config import (
    write_gateway_env, clear_gateway_env, get_clawdbot_command,
    install_clawdbot, gateway_manager, CONFIG_FILE, CONFIG_DIR,
    WORKSPACE_DIR, ProcessManager
)

ROOT_DIR = Path(__file__).parent
load_dotenv(ROOT_DIR / '.env')

# MongoDB connection
mongo_url = os.environ['MONGO_URL']
client = AsyncIOMotorClient(mongo_url)
db = client[os.environ.get('DB_NAME', 'moltbot_app')]

# Create the main app without a prefix
app = FastAPI()

# Create a router with the /api prefix
api_router = APIRouter(prefix="/api")

# Moltbot Gateway Management (config paths in gateway_config.py)
MOLTBOT_PORT = 18789
MOLTBOT_CONTROL_PORT = 18791

# Global state for gateway (per-user)
# Note: Process managed by subprocess (Railway deployment)
gateway_state = {
    "token": None,
    "provider": None,
    "started_at": None,
    "owner_user_id": None  # Track which user owns this instance
}

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ============== Pydantic Models ==============

class StatusCheck(BaseModel):
    model_config = ConfigDict(extra="ignore")
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    client_name: str
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class StatusCheckCreate(BaseModel):
    client_name: str


class OpenClawStartRequest(BaseModel):
    provider: str = "openai"  # "anthropic" or "openai"
    apiKey: Optional[str] = None  # API key for the provider


class OpenClawStartResponse(BaseModel):
    ok: bool
    controlUrl: str
    token: str
    message: str


class OpenClawStatusResponse(BaseModel):
    running: bool
    pid: Optional[int] = None
    provider: Optional[str] = None
    started_at: Optional[str] = None
    controlUrl: Optional[str] = None
    owner_user_id: Optional[str] = None
    is_owner: Optional[bool] = None


class User(BaseModel):
    model_config = ConfigDict(extra="ignore")
    user_id: str
    email: str
    name: str
    picture: Optional[str] = None
    created_at: Optional[datetime] = None


class SessionRequest(BaseModel):
    session_id: str


# ============== Authentication Helpers ==============

# Simple API key auth for Railway deployment (replaces Emergent OAuth)
AUTH_API_KEY = os.environ.get('AUTH_API_KEY', 'xeriaco-admin-2026')
AUTHORIZED_USERS = json.loads(os.environ.get('AUTHORIZED_USERS', '["ben", "steve374"]'))
SESSION_EXPIRY_DAYS = 7


async def get_instance_owner() -> Optional[dict]:
    """Get the instance owner from database."""
    doc = await db.instance_config.find_one({"_id": "instance_owner"})
    return doc


async def set_instance_owner(user: User) -> None:
    """Lock the instance to a specific user."""
    await db.instance_config.update_one(
        {"_id": "instance_owner"},
        {
            "$setOnInsert": {
                "user_id": user.user_id,
                "email": user.email,
                "name": user.name,
                "locked_at": datetime.now(timezone.utc)
            }
        },
        upsert=True
    )


async def check_instance_access(user: User) -> bool:
    """Check if user is allowed to access this instance."""
    owner = await get_instance_owner()
    if not owner:
        return True
    return owner.get("user_id") == user.user_id


async def get_current_user(request: Request) -> Optional[User]:
    """Get current user from API key or session token."""
    # Check API key header
    api_key = request.headers.get("X-API-Key")
    if api_key == AUTH_API_KEY:
        return User(user_id="admin", email="xeriaco@outlook.com", name="Ben")

    # Check session token (cookie or Bearer)
    session_token = request.cookies.get("session_token")
    if not session_token:
        auth_header = request.headers.get("Authorization")
        if auth_header and auth_header.startswith("Bearer "):
            session_token = auth_header.split(" ")[1]

    if not session_token:
        return None

    session_doc = await db.user_sessions.find_one(
        {"session_token": session_token}, {"_id": 0}
    )
    if not session_doc:
        return None

    expires_at = session_doc.get("expires_at")
    if isinstance(expires_at, str):
        expires_at = datetime.fromisoformat(expires_at)
    if expires_at.tzinfo is None:
        expires_at = expires_at.replace(tzinfo=timezone.utc)
    if expires_at < datetime.now(timezone.utc):
        return None

    user_doc = await db.users.find_one(
        {"user_id": session_doc["user_id"]}, {"_id": 0}
    )
    if not user_doc:
        return None
    return User(**user_doc)


async def require_auth(request: Request) -> User:
    """Dependency that requires authentication and instance access"""
    user = await get_current_user(request)
    if not user:
        raise HTTPException(status_code=401, detail="Not authenticated")
    
    # Check if user is allowed to access this instance
    if not await check_instance_access(user):
        owner = await get_instance_owner()
        raise HTTPException(
            status_code=403, 
            detail=f"This instance is locked to {owner.get('email', 'another user')}. Access denied."
        )
    return user


# ============== Auth Endpoints ==============

@api_router.get("/auth/instance")
async def get_instance_status():
    owner = await get_instance_owner()
    if owner:
        return {"locked": True}
    return {"locked": False}


@api_router.post("/auth/session")
async def create_session(request: SessionRequest, response: Response):
    """Simple API key login (replaces Emergent OAuth)."""
    try:
        # Accept API key or username from session_id field
        session_id = request.session_id
        
        # Check if it's the admin API key
        if session_id == AUTH_API_KEY:
            user_id = "admin"
            email = "xeriaco@outlook.com"
            name = "Ben"
        elif session_id in AUTHORIZED_USERS:
            user_id = session_id
            email = f"{session_id}@xeriaco.com"
            name = session_id
        else:
            raise HTTPException(status_code=401, detail="Invalid API key or username")

        # Check instance lock
        owner = await get_instance_owner()
        if owner and owner.get("user_id") != user_id:
            raise HTTPException(status_code=403, detail="Instance locked to another user")

        # Upsert user
        await db.users.update_one(
            {"user_id": user_id},
            {"$set": {"user_id": user_id, "email": email, "name": name}},
            upsert=True
        )

        # Create session
        session_token = secrets.token_hex(32)
        expires_at = datetime.now(timezone.utc) + timedelta(days=SESSION_EXPIRY_DAYS)
        await db.user_sessions.insert_one({
            "user_id": user_id,
            "session_token": session_token,
            "expires_at": expires_at,
            "created_at": datetime.now(timezone.utc)
        })

        response.set_cookie(
            key="session_token", value=session_token,
            httponly=True, secure=True, samesite="none",
            path="/", max_age=SESSION_EXPIRY_DAYS * 24 * 60 * 60
        )

        user_doc = await db.users.find_one({"user_id": user_id}, {"_id": 0})
        return {"ok": True, "user": user_doc}

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Session creation error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@api_router.get("/auth/me")
async def get_me(request: Request):
    """Get current authenticated user"""
    user = await get_current_user(request)
    if not user:
        raise HTTPException(status_code=401, detail="Not authenticated")
    return user.model_dump()


@api_router.post("/auth/logout")
async def logout(request: Request, response: Response):
    """Logout - delete session and clear cookie"""
    session_token = request.cookies.get("session_token")

    if session_token:
        await db.user_sessions.delete_one({"session_token": session_token})

    response.delete_cookie(
        key="session_token",
        path="/",
        secure=True,
        samesite="none"
    )

    return {"ok": True, "message": "Logged out"}


# ============== Moltbot Helpers (Railway-adapted) ==============

# Functions moved to gateway_config.py: get_clawdbot_command, install_clawdbot, ProcessManager


def generate_token():
    """Generate a secure random token"""
    return secrets.token_hex(32)


def ensure_moltbot_installed():
    """Ensure clawdbot is installed"""
    cmd = get_clawdbot_command()
    if cmd:
        logger.info(f"clawdbot found: {cmd}")
        return True
    return install_clawdbot()


def create_moltbot_config(token: str = None, api_key: str = None, provider: str = "openai", force_new_token: bool = False):
    """Update clawdbot.json with gateway config and provider settings"""
    os.makedirs(CONFIG_DIR, exist_ok=True)
    os.makedirs(WORKSPACE_DIR, exist_ok=True)

    existing_config = {}
    if os.path.exists(CONFIG_FILE):
        try:
            with open(CONFIG_FILE, "r") as f:
                existing_config = json.load(f)
        except:
            pass

    existing_token = None
    if not force_new_token:
        try:
            existing_token = existing_config.get("gateway", {}).get("auth", {}).get("token")
        except:
            pass

    final_token = existing_token or token or generate_token()
    logger.info(f"Config token: {'reusing existing' if existing_token else 'new token'}, provider: {provider}")

    gateway_config = {
        "mode": "local",
        "port": MOLTBOT_PORT,
        "bind": "lan",
        "auth": {"mode": "token", "token": final_token},
        "controlUi": {"enabled": True, "allowInsecureAuth": True}
    }

    existing_config["gateway"] = gateway_config

    if "models" not in existing_config:
        existing_config["models"] = {"mode": "merge", "providers": {}}
    existing_config["models"]["mode"] = "merge"
    if "providers" not in existing_config["models"]:
        existing_config["models"]["providers"] = {}

    if "agents" not in existing_config:
        existing_config["agents"] = {"defaults": {}}
    if "defaults" not in existing_config["agents"]:
        existing_config["agents"]["defaults"] = {}
    existing_config["agents"]["defaults"]["workspace"] = WORKSPACE_DIR

    if provider == "openai":
        openai_key = api_key or os.environ.get('OPENAI_API_KEY', '')
        openai_provider = {
            "baseUrl": "https://api.openai.com/v1/",
            "apiKey": openai_key,
            "api": "openai-completions",
            "models": [
                {"id": "gpt-4o", "name": "GPT-4o", "reasoning": False, "input": ["text", "image"],
                 "cost": {"input": 0.0000025, "output": 0.00001}, "contextWindow": 128000, "maxTokens": 16384},
                {"id": "gpt-4o-mini", "name": "GPT-4o Mini", "reasoning": False, "input": ["text", "image"],
                 "cost": {"input": 0.00000015, "output": 0.0000006}, "contextWindow": 128000, "maxTokens": 16384}
            ]
        }
        existing_config["models"]["providers"]["openai"] = openai_provider
        existing_config["agents"]["defaults"]["model"] = {"primary": "openai/gpt-4o"}

    elif provider == "anthropic":
        anthropic_key = api_key or os.environ.get('ANTHROPIC_API_KEY', '')
        anthropic_provider = {
            "baseUrl": "https://api.anthropic.com",
            "apiKey": anthropic_key,
            "api": "anthropic-messages",
            "models": [
                {"id": "claude-sonnet-4-20250514", "name": "Claude Sonnet 4", "input": ["text", "image"],
                 "cost": {"input": 0.000003, "output": 0.000015}, "contextWindow": 200000, "maxTokens": 64000},
            ]
        }
        existing_config["models"]["providers"]["anthropic"] = anthropic_provider
        existing_config["agents"]["defaults"]["model"] = {"primary": "anthropic/claude-sonnet-4-20250514"}

    with open(CONFIG_FILE, "w") as f:
        json.dump(existing_config, f, indent=2)

    logger.info(f"Updated config at {CONFIG_FILE} for provider: {provider}")
    return final_token


async def start_gateway_process(api_key: str, provider: str, owner_user_id: str):
    """Start the gateway process (Railway-adapted: uses subprocess instead of supervisor)"""
    global gateway_state

    if gateway_manager.status():
        logger.info("Gateway already running, recovering state...")
        token = None
        try:
            with open(CONFIG_FILE, 'r') as f:
                config = json.load(f)
            token = config.get("gateway", {}).get("auth", {}).get("token")
        except:
            pass

        if not token:
            token = generate_token()
            create_moltbot_config(token=token, api_key=api_key, provider=provider, force_new_token=True)

        gateway_state["token"] = token
        gateway_state["provider"] = provider
        gateway_state["started_at"] = datetime.now(timezone.utc).isoformat()
        gateway_state["owner_user_id"] = owner_user_id

        await db.moltbot_configs.update_one(
            {"_id": "gateway_config"},
            {"$set": {"should_run": True, "owner_user_id": owner_user_id, "provider": provider,
                      "token": token, "started_at": gateway_state["started_at"],
                      "updated_at": datetime.now(timezone.utc)}},
            upsert=True
        )
        return token

    clawdbot_cmd = get_clawdbot_command()
    if not clawdbot_cmd:
        if not ensure_moltbot_installed():
            raise HTTPException(status_code=500, detail="clawdbot not installed")
        clawdbot_cmd = get_clawdbot_command()
        if not clawdbot_cmd:
            raise HTTPException(status_code=500, detail="clawdbot not found after install")

    token = create_moltbot_config(api_key=api_key, provider=provider)
    write_gateway_env(token=token, api_key=api_key, provider=provider)

    logger.info(f"Starting gateway on port {MOLTBOT_PORT}...")

    if not gateway_manager.start(clawdbot_cmd):
        raise HTTPException(status_code=500, detail="Failed to start gateway")

    gateway_state["token"] = token
    gateway_state["provider"] = provider
    gateway_state["started_at"] = datetime.now(timezone.utc).isoformat()
    gateway_state["owner_user_id"] = owner_user_id

    max_wait = 60
    start_time = asyncio.get_event_loop().time()
    async with httpx.AsyncClient() as http_client:
        while asyncio.get_event_loop().time() - start_time < max_wait:
            try:
                response = await http_client.get(f"http://127.0.0.1:{MOLTBOT_PORT}/", timeout=2.0)
                if response.status_code == 200:
                    logger.info("Gateway is ready!")
                    await db.moltbot_configs.update_one(
                        {"_id": "gateway_config"},
                        {"$set": {"should_run": True, "owner_user_id": owner_user_id,
                                  "provider": provider, "token": token,
                                  "started_at": gateway_state["started_at"],
                                  "updated_at": datetime.now(timezone.utc)}},
                        upsert=True
                    )
                    return token
            except:
                pass
            await asyncio.sleep(1)

    if not gateway_manager.status():
        raise HTTPException(status_code=500, detail="Gateway failed to start")
    raise HTTPException(status_code=500, detail="Gateway not ready in time")


def check_gateway_running():
    """Check if the gateway process is running"""
    return gateway_manager.status()


# ============== Moltbot API Endpoints (Protected) ==============

@api_router.get("/")
async def root():
    return {"message": "OpenClaw Hosting API"}


@api_router.post("/openclaw/start", response_model=OpenClawStartResponse)
async def start_moltbot(request: OpenClawStartRequest, req: Request):
    """Start the OpenClaw gateway (requires auth)"""
    user = await require_auth(req)

    if request.provider not in ["anthropic", "openai"]:
        raise HTTPException(status_code=400, detail="Invalid provider. Use 'anthropic' or 'openai'")

    # API key required for provider
    if request.provider in ["anthropic", "openai"] and (not request.apiKey or len(request.apiKey) < 10):
        raise HTTPException(status_code=400, detail="API key required for anthropic/openai providers")

    # Check if Moltbot is already running by another user
    if check_gateway_running() and gateway_state["owner_user_id"] != user.user_id:
        raise HTTPException(
            status_code=403,
            detail="OpenClaw is already running by another user. Please wait for them to stop it."
        )

    try:
        token = await start_gateway_process(request.apiKey, request.provider, user.user_id)

        # Lock the instance to this user on first successful start
        await set_instance_owner(user)
        logger.info(f"Instance locked to user: {user.email}")

        return OpenClawStartResponse(
            ok=True,
            controlUrl="/api/openclaw/ui/",
            token=token,
            message="OpenClaw started successfully"
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to start Moltbot: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@api_router.get("/openclaw/status", response_model=OpenClawStatusResponse)
async def get_moltbot_status(request: Request):
    """Get the current status of the Moltbot gateway"""
    user = await get_current_user(request)
    running = check_gateway_running()

    if running:
        is_owner = user and gateway_state["owner_user_id"] == user.user_id
        return OpenClawStatusResponse(
            running=True,
            pid=gateway_manager.get_pid(),
            provider=gateway_state["provider"],
            started_at=gateway_state["started_at"],
            controlUrl="/api/openclaw/ui/",
            owner_user_id=gateway_state["owner_user_id"],
            is_owner=is_owner
        )
    else:
        return OpenClawStatusResponse(running=False)


@api_router.get("/openclaw/whatsapp/status")
async def get_whatsapp_connection_status():
    """Get basic WhatsApp connection status. Auto-fix handled by background watcher."""
    return get_whatsapp_status()


@api_router.post("/openclaw/stop")
async def stop_moltbot(request: Request):
    """Stop the Moltbot gateway (only owner can stop)"""
    user = await require_auth(request)

    global gateway_state

    if not check_gateway_running():
        # Clear should_run flag even if not running
        await db.moltbot_configs.update_one(
            {"_id": "gateway_config"},
            {"$set": {"should_run": False, "updated_at": datetime.now(timezone.utc)}}
        )
        return {"ok": True, "message": "OpenClaw is not running"}

    # Check if user is the owner
    if gateway_state["owner_user_id"] != user.user_id:
        raise HTTPException(status_code=403, detail="Only the owner can stop OpenClaw")

    # Stop via supervisor
    if not gateway_manager.stop():
        logger.error("Failed to stop gateway process")

    # Clear the gateway env file
    clear_gateway_env()

    # Clear should_run flag in database
    await db.moltbot_configs.update_one(
        {"_id": "gateway_config"},
        {"$set": {"should_run": False, "updated_at": datetime.now(timezone.utc)}}
    )

    # Clear in-memory state
    gateway_state["token"] = None
    gateway_state["provider"] = None
    gateway_state["started_at"] = None
    gateway_state["owner_user_id"] = None

    return {"ok": True, "message": "OpenClaw stopped"}


@api_router.get("/openclaw/token")
async def get_moltbot_token(request: Request):
    """Get the current gateway token for authentication (only owner)"""
    user = await require_auth(request)

    if not check_gateway_running():
        raise HTTPException(status_code=404, detail="OpenClaw not running")

    # Only owner can get the token
    if gateway_state["owner_user_id"] != user.user_id:
        raise HTTPException(status_code=403, detail="Only the owner can access the token")

    return {"token": gateway_state.get("token")}


# ============== Moltbot Proxy (Protected) ==============

@api_router.api_route("/openclaw/ui/{path:path}", methods=["GET", "POST", "PUT", "DELETE", "PATCH", "OPTIONS", "HEAD"])
async def proxy_moltbot_ui(request: Request, path: str = ""):
    """Proxy requests to the Moltbot Control UI (only owner can access)"""
    user = await get_current_user(request)

    if not check_gateway_running():
        return HTMLResponse(
            content="<html><body><h1>OpenClaw not running</h1><p>Please start OpenClaw first.</p><a href='/'>Go to setup</a></body></html>",
            status_code=503
        )

    # Check if user is the owner
    if not user or gateway_state["owner_user_id"] != user.user_id:
        return HTMLResponse(
            content="<html><body><h1>Access Denied</h1><p>This OpenClaw instance is owned by another user.</p><a href='/'>Go back</a></body></html>",
            status_code=403
        )

    target_url = f"http://127.0.0.1:{MOLTBOT_PORT}/{path}"

    # Handle query string
    if request.query_params:
        target_url += f"?{request.query_params}"

    async with httpx.AsyncClient() as client:
        try:
            # Forward the request
            headers = dict(request.headers)
            headers.pop("host", None)
            headers.pop("content-length", None)

            body = await request.body()

            response = await client.request(
                method=request.method,
                url=target_url,
                headers=headers,
                content=body,
                timeout=30.0
            )

            # Filter response headers
            exclude_headers = {"content-encoding", "content-length", "transfer-encoding", "connection"}
            response_headers = {
                k: v for k, v in response.headers.items()
                if k.lower() not in exclude_headers
            }

            # Get content and rewrite WebSocket URLs if HTML
            content = response.content
            content_type = response.headers.get("content-type", "")

            # Get the current gateway token
            current_token = gateway_state.get("token", "")

            # If it's HTML, rewrite any WebSocket URLs to use our proxy
            if "text/html" in content_type:
                content_str = content.decode('utf-8', errors='ignore')
                # Inject WebSocket URL override script with token
                ws_override = f'''
<script>
// OpenClaw Proxy Configuration
window.__MOLTBOT_PROXY_TOKEN__ = "{current_token}";
window.__MOLTBOT_PROXY_WS_URL__ = (window.location.protocol === 'https:' ? 'wss:' : 'ws:') + '//' + window.location.host + '/api/openclaw/ws';

// Override WebSocket to use proxy path
(function() {{
    const originalWS = window.WebSocket;
    const proxyWsUrl = window.__MOLTBOT_PROXY_WS_URL__;

    window.WebSocket = function(url, protocols) {{
        let finalUrl = url;

        // Rewrite any OpenClaw gateway URLs to use our proxy
        if (url.includes('127.0.0.1:18789') ||
            url.includes('localhost:18789') ||
            url.includes('0.0.0.0:18789') ||
            (url.includes(':18789') && !url.includes('/api/openclaw/'))) {{
            finalUrl = proxyWsUrl;
        }}

        // If it's a relative URL or same-origin, redirect to proxy
        try {{
            const urlObj = new URL(url, window.location.origin);
            if (urlObj.port === '18789' || urlObj.pathname === '/' && !url.startsWith(proxyWsUrl)) {{
                finalUrl = proxyWsUrl;
            }}
        }} catch (e) {{}}

        console.log('[OpenClaw Proxy] WebSocket:', url, '->', finalUrl);
        return new originalWS(finalUrl, protocols);
    }};

    // Copy static properties
    window.WebSocket.prototype = originalWS.prototype;
    window.WebSocket.CONNECTING = originalWS.CONNECTING;
    window.WebSocket.OPEN = originalWS.OPEN;
    window.WebSocket.CLOSING = originalWS.CLOSING;
    window.WebSocket.CLOSED = originalWS.CLOSED;
}})();
</script>
'''
                # Insert before </head> or at start of <body>
                if '</head>' in content_str:
                    content_str = content_str.replace('</head>', ws_override + '</head>')
                elif '<body>' in content_str:
                    content_str = content_str.replace('<body>', '<body>' + ws_override)
                else:
                    content_str = ws_override + content_str
                content = content_str.encode('utf-8')

            return Response(
                content=content,
                status_code=response.status_code,
                headers=response_headers,
                media_type=response.headers.get("content-type")
            )
        except httpx.RequestError as e:
            logger.error(f"Proxy error: {e}")
            raise HTTPException(status_code=502, detail="Failed to connect to OpenClaw")


# Root proxy for Moltbot UI (handles /api/moltbot/ui without trailing path)
@api_router.get("/openclaw/ui")
async def proxy_moltbot_ui_root(request: Request):
    """Redirect to Moltbot UI with trailing slash"""
    return Response(
        status_code=307,
        headers={"Location": "/api/openclaw/ui/"}
    )


# WebSocket proxy for Moltbot (Protected)
@api_router.websocket("/openclaw/ws")
async def websocket_proxy(websocket: WebSocket):
    """WebSocket proxy for Moltbot Control UI"""
    await websocket.accept()

    if not check_gateway_running():
        await websocket.close(code=1013, reason="OpenClaw not running")
        return

    # Note: WebSocket auth is handled by the token in the connection itself
    # The Control UI passes the token in the connect message

    # Get the token from state
    token = gateway_state.get("token")

    # Moltbot expects WebSocket connection with optional auth in query params
    moltbot_ws_url = f"ws://127.0.0.1:{MOLTBOT_PORT}/"

    logger.info(f"WebSocket proxy connecting to: {moltbot_ws_url}")

    try:
        # Additional headers for connection
        extra_headers = {}
        if token:
            extra_headers["X-Auth-Token"] = token

        async with websockets.connect(
            moltbot_ws_url,
            ping_interval=20,
            ping_timeout=20,
            close_timeout=10,
            additional_headers=extra_headers if extra_headers else None
        ) as moltbot_ws:

            async def client_to_moltbot():
                try:
                    while True:
                        try:
                            data = await websocket.receive()
                            if data["type"] == "websocket.receive":
                                if "text" in data:
                                    await moltbot_ws.send(data["text"])
                                elif "bytes" in data:
                                    await moltbot_ws.send(data["bytes"])
                            elif data["type"] == "websocket.disconnect":
                                break
                        except WebSocketDisconnect:
                            break
                except Exception as e:
                    logger.error(f"Client to Moltbot error: {e}")

            async def moltbot_to_client():
                try:
                    async for message in moltbot_ws:
                        if websocket.client_state == WebSocketState.CONNECTED:
                            if isinstance(message, str):
                                await websocket.send_text(message)
                            else:
                                await websocket.send_bytes(message)
                except ConnectionClosed as e:
                    logger.info(f"Moltbot WebSocket closed: {e}")
                except Exception as e:
                    logger.error(f"Moltbot to client error: {e}")

            # Run both directions concurrently
            done, pending = await asyncio.wait(
                [
                    asyncio.create_task(client_to_moltbot()),
                    asyncio.create_task(moltbot_to_client())
                ],
                return_when=asyncio.FIRST_COMPLETED
            )

            # Cancel pending tasks
            for task in pending:
                task.cancel()

    except Exception as e:
        logger.error(f"WebSocket proxy error: {e}")
    finally:
        try:
            if websocket.client_state == WebSocketState.CONNECTED:
                await websocket.close(code=1011, reason="Proxy connection ended")
        except:
            pass


# ============== Legacy Status Endpoints ==============

@api_router.post("/status", response_model=StatusCheck)
async def create_status_check(input: StatusCheckCreate):
    status_dict = input.model_dump()
    status_obj = StatusCheck(**status_dict)

    doc = status_obj.model_dump()
    doc['timestamp'] = doc['timestamp'].isoformat()

    _ = await db.status_checks.insert_one(doc)
    return status_obj


@api_router.get("/status", response_model=List[StatusCheck])
async def get_status_checks():
    status_checks = await db.status_checks.find({}, {"_id": 0}).to_list(1000)

    for check in status_checks:
        if isinstance(check['timestamp'], str):
            check['timestamp'] = datetime.fromisoformat(check['timestamp'])

    return status_checks


# ============== Dropshipping Automation Endpoints ==============
from dropship.system import get_system, DropshipSystem
from dropship.models import Product, Competitor

# Initialize dropship system
dropship_system: DropshipSystem = None

@api_router.get("/dropship/status")
async def get_dropship_status():
    """Get dropshipping system status"""
    global dropship_system
    if dropship_system is None:
        dropship_system = get_system(db)
    return dropship_system.get_status()

@api_router.post("/dropship/initialize")
async def initialize_dropship():
    """Initialize the dropshipping system"""
    global dropship_system
    if dropship_system is None:
        dropship_system = get_system(db)
    return await dropship_system.initialize()

@api_router.get("/dropship/site-audit")
async def run_site_audit():
    """Run a full site audit"""
    global dropship_system
    if dropship_system is None:
        dropship_system = get_system(db)
    return await dropship_system.run_site_audit()

@api_router.get("/dropship/competitors")
async def check_competitors():
    """Check all competitors"""
    global dropship_system
    if dropship_system is None:
        dropship_system = get_system(db)
    return await dropship_system.check_competitors()

@api_router.post("/dropship/competitors/add")
async def add_competitor(name: str, url: str):
    """Add a competitor to monitor"""
    global dropship_system
    if dropship_system is None:
        dropship_system = get_system(db)
    dropship_system.competitor_monitor.add_competitor(name, url)
    return {"status": "added", "competitor": name, "url": url}

@api_router.get("/dropship/reports/daily")
async def get_daily_report():
    """Get daily performance report"""
    global dropship_system
    if dropship_system is None:
        dropship_system = get_system(db)
    return await dropship_system.get_daily_report()

@api_router.get("/dropship/reports/weekly")
async def get_weekly_report():
    """Get weekly performance report"""
    global dropship_system
    if dropship_system is None:
        dropship_system = get_system(db)
    return await dropship_system.get_weekly_report()

@api_router.post("/dropship/pricing/analyze")
async def analyze_pricing(products: List[dict]):
    """Analyze pricing for products"""
    global dropship_system
    if dropship_system is None:
        dropship_system = get_system(db)
    return await dropship_system.analyze_pricing(products)

@api_router.post("/dropship/inventory/check")
async def check_inventory(products: List[dict]):
    """Check inventory levels"""
    global dropship_system
    if dropship_system is None:
        dropship_system = get_system(db)
    return dropship_system.check_inventory(products)

@api_router.post("/dropship/profit/calculate")
async def calculate_profit(product: dict):
    """Calculate profit margin for a product"""
    global dropship_system
    if dropship_system is None:
        dropship_system = get_system(db)
    return dropship_system.calculate_profit(product)

@api_router.post("/dropship/funnel/analyze")
async def analyze_funnel(funnel_data: dict):
    """Analyze conversion funnel"""
    global dropship_system
    if dropship_system is None:
        dropship_system = get_system(db)
    return dropship_system.analyze_funnel(funnel_data)

@api_router.get("/dropship/alerts")
async def get_alerts(unacknowledged: bool = False, severity: str = None):
    """Get system alerts"""
    global dropship_system
    if dropship_system is None:
        dropship_system = get_system(db)
    return dropship_system.alerts.get_alerts(unacknowledged, severity)

@api_router.post("/dropship/alerts/{alert_id}/acknowledge")
async def acknowledge_alert(alert_id: str):
    """Acknowledge an alert"""
    global dropship_system
    if dropship_system is None:
        dropship_system = get_system(db)
    success = dropship_system.alerts.acknowledge_alert(alert_id)
    return {"acknowledged": success}

@api_router.get("/dropship/scheduler/status")
async def get_scheduler_status():
    """Get scheduler status"""
    global dropship_system
    if dropship_system is None:
        dropship_system = get_system(db)
    return dropship_system.scheduler.get_status()

@api_router.post("/dropship/scheduler/start")
async def start_scheduler():
    """Start the automation scheduler"""
    global dropship_system
    if dropship_system is None:
        dropship_system = get_system(db)
    return dropship_system.start_automation()

@api_router.post("/dropship/scheduler/stop")
async def stop_scheduler():
    """Stop the automation scheduler"""
    global dropship_system
    if dropship_system is None:
        dropship_system = get_system(db)
    return dropship_system.stop_automation()

@api_router.post("/dropship/scheduler/run/{task_name}")
async def run_task(task_name: str):
    """Run a specific scheduled task"""
    global dropship_system
    if dropship_system is None:
        dropship_system = get_system(db)
    return await dropship_system.scheduler.run_task(task_name)


# Include the router in the main app
app.include_router(api_router)

app.add_middleware(
    CORSMiddleware,
    allow_credentials=True,
    allow_origins=os.environ.get('CORS_ORIGINS', '*').split(','),
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
async def startup_event():
    """Run on server startup - auto-start gateway if configured"""
    global gateway_state
    logger.info("Server starting up on Railway...")

    # Check and install clawdbot dependencies
    clawdbot_cmd = get_clawdbot_command()
    if clawdbot_cmd:
        logger.info(f"clawdbot ready: {clawdbot_cmd}")
    else:
        logger.info("clawdbot not found, will install on first use")

    # Check database for persistent gateway config
    config_doc = None
    try:
        config_doc = await db.moltbot_configs.find_one({"_id": "gateway_config"})
    except Exception as e:
        logger.warning(f"Could not read gateway config: {e}")

    should_run = config_doc.get("should_run", False) if config_doc else False
    logger.info(f"Gateway should_run flag: {should_run}")

    # Check if gateway already running
    if gateway_manager.status():
        pid = gateway_manager.get_pid()
        logger.info(f"Gateway already running (PID: {pid})")
        gateway_state["provider"] = config_doc.get("provider", "openai") if config_doc else "openai"
        try:
            with open(CONFIG_FILE, 'r') as f:
                config = json.load(f)
            gateway_state["token"] = config.get("gateway", {}).get("auth", {}).get("token")
        except:
            pass
        if config_doc:
            gateway_state["owner_user_id"] = config_doc.get("owner_user_id")
            gateway_state["started_at"] = config_doc.get("started_at")

    elif should_run and config_doc:
        logger.info("Gateway should_run=True but not running - auto-starting...")
        token = config_doc.get("token")
        if not token:
            try:
                with open(CONFIG_FILE, 'r') as f:
                    config = json.load(f)
                token = config.get("gateway", {}).get("auth", {}).get("token")
            except:
                token = generate_token()

        provider = config_doc.get("provider", "openai")
        write_gateway_env(token=token, provider=provider)

        if gateway_manager.start():
            logger.info("Gateway auto-started successfully")
            await asyncio.sleep(3)
            gateway_state["token"] = token
            gateway_state["provider"] = provider
            gateway_state["owner_user_id"] = config_doc.get("owner_user_id")
            gateway_state["started_at"] = config_doc.get("started_at")
        else:
            logger.error("Failed to auto-start gateway")

    logger.info("Server startup complete")


@app.on_event("shutdown")
async def shutdown_db_client():
    """Cleanup on shutdown"""
    # Stop gateway process
    if gateway_manager.status():
        gateway_manager.stop()
        logger.info("Gateway stopped")

    client.close()
