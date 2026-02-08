"""
Gateway configuration utilities for Railway deployment.
Replaces supervisor-based management with direct subprocess control.
"""

import os
import stat
import json
import signal
import subprocess
import asyncio
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

GATEWAY_ENV_FILE = os.path.expanduser("~/.clawdbot/gateway.env")
GATEWAY_ENV_DIR = os.path.expanduser("~/.clawdbot")
CONFIG_DIR = os.path.expanduser("~/.clawdbot")
CONFIG_FILE = os.path.join(CONFIG_DIR, "clawdbot.json")
WORKSPACE_DIR = os.path.expanduser("~/clawd")
NODE_DIR = "/usr/local"
CLAWDBOT_DIR = os.path.expanduser("~/.clawdbot-bin")

# Global process reference
_gateway_process = None


def write_gateway_env(token: str, api_key: str = None, provider: str = "openai") -> None:
    os.makedirs(GATEWAY_ENV_DIR, exist_ok=True)
    lines = [f'export CLAWDBOT_GATEWAY_TOKEN="{token}"']
    if api_key:
        if provider == "anthropic":
            lines.append(f'export ANTHROPIC_API_KEY="{api_key}"')
        elif provider == "openai":
            lines.append(f'export OPENAI_API_KEY="{api_key}"')
    content = "\n".join(lines) + "\n"
    with open(GATEWAY_ENV_FILE, 'w') as f:
        f.write(content)
    os.chmod(GATEWAY_ENV_FILE, stat.S_IRUSR | stat.S_IWUSR)


def clear_gateway_env() -> None:
    if os.path.exists(GATEWAY_ENV_FILE):
        os.remove(GATEWAY_ENV_FILE)


def get_clawdbot_command():
    """Find the clawdbot executable."""
    # Check common locations
    for path in [
        f"{CLAWDBOT_DIR}/clawdbot",
        "/usr/local/bin/clawdbot",
        "/root/.clawdbot-bin/clawdbot",
    ]:
        if os.path.exists(path):
            return path
    
    # Check PATH
    import shutil
    cmd = shutil.which("clawdbot")
    if cmd:
        return cmd
    
    # Check npm global
    try:
        result = subprocess.run(["npm", "root", "-g"], capture_output=True, text=True, timeout=5)
        npm_global = result.stdout.strip()
        candidate = os.path.join(os.path.dirname(npm_global), "bin", "clawdbot")
        if os.path.exists(candidate):
            return candidate
    except:
        pass
    
    return None


def install_clawdbot():
    """Install clawdbot via npm."""
    logger.info("Installing clawdbot via npm...")
    try:
        result = subprocess.run(
            ["npm", "install", "-g", "@anthropic/clawdbot"],
            capture_output=True, text=True, timeout=120
        )
        if result.returncode == 0:
            logger.info(f"clawdbot installed: {result.stdout[-200:]}")
            return True
        else:
            logger.error(f"npm install failed: {result.stderr[-500:]}")
            # Try alternative package name
            result2 = subprocess.run(
                ["npm", "install", "-g", "clawdbot"],
                capture_output=True, text=True, timeout=120
            )
            if result2.returncode == 0:
                logger.info(f"clawdbot installed (alt): {result2.stdout[-200:]}")
                return True
            logger.error(f"Alt install also failed: {result2.stderr[-500:]}")
            return False
    except Exception as e:
        logger.error(f"Installation error: {e}")
        return False


class ProcessManager:
    """Manages gateway process directly (replaces supervisor)."""
    
    def __init__(self):
        self.process = None
        self.log_task = None
    
    def start(self, cmd=None) -> bool:
        if self.process and self.process.poll() is None:
            logger.info("Gateway already running")
            return True
        
        if not cmd:
            cmd = get_clawdbot_command()
        if not cmd:
            logger.error("clawdbot not found")
            return False
        
        env = os.environ.copy()
        # Load gateway env
        if os.path.exists(GATEWAY_ENV_FILE):
            with open(GATEWAY_ENV_FILE) as f:
                for line in f:
                    line = line.strip()
                    if line.startswith('export '):
                        line = line[7:]
                    if '=' in line:
                        key, val = line.split('=', 1)
                        env[key] = val.strip('"')
        
        try:
            self.process = subprocess.Popen(
                [cmd, "gateway", "--config", CONFIG_FILE],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                env=env,
                cwd=WORKSPACE_DIR
            )
            logger.info(f"Gateway started (PID: {self.process.pid})")
            return True
        except Exception as e:
            logger.error(f"Failed to start gateway: {e}")
            return False
    
    def stop(self) -> bool:
        if not self.process:
            return True
        try:
            self.process.terminate()
            try:
                self.process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait(timeout=5)
            logger.info("Gateway stopped")
            self.process = None
            return True
        except Exception as e:
            logger.error(f"Failed to stop gateway: {e}")
            return False
    
    def status(self) -> bool:
        if not self.process:
            return False
        return self.process.poll() is None
    
    def get_pid(self):
        if self.process and self.process.poll() is None:
            return self.process.pid
        return None
    
    def restart(self) -> bool:
        self.stop()
        return self.start()
    
    def reload_config(self):
        pass  # No-op for subprocess manager


# Global instance
gateway_manager = ProcessManager()
