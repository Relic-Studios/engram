#!/usr/bin/env python3
"""
LLM Backend & OAuth2 Setup

Supports two LLM backends for the soul system:

1. Claude (OAuth2 Bearer token via Claude Max PKCE flow)
   - Token stored as CLAUDE_OAUTH_TOKEN in config/secrets.env
   - Used with: Authorization: Bearer {token}
   -            anthropic-beta: oauth-2025-04-20

2. Ollama (local models, no auth needed)
   - Model name stored as OLLAMA_MODEL in config/secrets.env
   - Talks to http://localhost:11434/api/chat
   - Default model: llama3.1 (configurable)

Both power:
- Soul Creation GUI (generating soul prose)
- Self-realization (AI articulating discoveries about itself)
- Any future tool needing LLM intervention

Usage:
    python oauth_setup.py              # Interactive OAuth flow
    python oauth_setup.py --code CODE  # Apply OAuth code directly
    python oauth_setup.py --ollama     # Configure Ollama backend
    python oauth_setup.py --ollama MODEL_NAME  # Set specific Ollama model
"""

import json
import webbrowser
import urllib.parse
import urllib.request
import urllib.error
import hashlib
import base64
import secrets
import sys
import os
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Dict

# ── OAuth constants (Claude Max flow, identical to claude_code.py) ──
CLIENT_ID = "9d1c250a-e61b-44d9-88ed-5944d1962f5e"
AUTH_URL = "https://claude.ai/oauth/authorize"
TOKEN_URL = "https://console.anthropic.com/v1/oauth/token"
REDIRECT_URI = "https://console.anthropic.com/oauth/code/callback"
CLAUDE_API_BASE = "https://api.anthropic.com/v1"

# ── Ollama constants ──
OLLAMA_BASE_URL = "http://localhost:11434"
OLLAMA_DEFAULT_MODEL = "llama3.1"

# ── Where config lives ──
# Check thomas-soul config first (global), fall back to engram project config
_THOMAS_SECRETS = Path("C:/Dev/thomas-soul/config/secrets.env")
_LOCAL_SECRETS = Path(__file__).parent.parent / "config" / "secrets.env"
SECRETS_ENV = _THOMAS_SECRETS if _THOMAS_SECRETS.exists() else _LOCAL_SECRETS


# ═══════════════════════════════════════════════════════════════
# SECRETS.ENV READ/WRITE
# ═══════════════════════════════════════════════════════════════


def _read_secrets() -> Dict[str, str]:
    """Read all key=value pairs from secrets.env."""
    result = {}
    if SECRETS_ENV.exists():
        for line in SECRETS_ENV.read_text(encoding="utf-8").split("\n"):
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, val = line.split("=", 1)
                result[key.strip()] = val.strip()
    return result


def _write_secret(key: str, value: str, comment: str = "") -> str:
    """
    Write a key=value to config/secrets.env.
    Preserves all existing entries. Overwrites if key exists.
    Returns path written to.
    """
    SECRETS_ENV.parent.mkdir(parents=True, exist_ok=True)

    existing = ""
    if SECRETS_ENV.exists():
        existing = SECRETS_ENV.read_text(encoding="utf-8")

    lines = existing.split("\n")
    new_lines = []
    found = False
    for line in lines:
        if line.startswith(f"{key}="):
            new_lines.append(f"{key}={value}")
            found = True
        else:
            new_lines.append(line)

    if not found:
        if new_lines and new_lines[-1].strip():
            new_lines.append("")
        if comment:
            for c in comment.split("\n"):
                new_lines.append(f"# {c}")
        new_lines.append(f"{key}={value}")
        new_lines.append("")

    SECRETS_ENV.write_text("\n".join(new_lines), encoding="utf-8")
    return str(SECRETS_ENV)


def load_token() -> str:
    """Load CLAUDE_OAUTH_TOKEN from secrets.env."""
    return _read_secrets().get("CLAUDE_OAUTH_TOKEN", "")


def load_ollama_model() -> str:
    """Load OLLAMA_MODEL from secrets.env."""
    return _read_secrets().get("OLLAMA_MODEL", "")


def load_active_backend() -> str:
    """Load LLM_BACKEND from secrets.env. Returns 'claude' or 'ollama'."""
    return _read_secrets().get("LLM_BACKEND", "")


def write_token_to_secrets(token: str) -> str:
    """Write CLAUDE_OAUTH_TOKEN and set backend to claude."""
    _write_secret(
        "CLAUDE_OAUTH_TOKEN",
        token,
        f"Claude OAuth2 Bearer Token (Claude Max flow)\n"
        f"Applied by oauth_setup.py on {datetime.now().strftime('%Y-%m-%d %H:%M')}\n"
        f"Used with: Authorization: Bearer <token>\n"
        f"           anthropic-beta: oauth-2025-04-20",
    )
    _write_secret("LLM_BACKEND", "claude", "Active LLM backend (claude or ollama)")
    return str(SECRETS_ENV)


def write_ollama_to_secrets(model: str) -> str:
    """Write OLLAMA_MODEL and set backend to ollama."""
    _write_secret(
        "OLLAMA_MODEL",
        model,
        f"Ollama local model\n"
        f"Set by oauth_setup.py on {datetime.now().strftime('%Y-%m-%d %H:%M')}\n"
        f"Used with: http://localhost:11434/api/chat",
    )
    _write_secret("LLM_BACKEND", "ollama", "Active LLM backend (claude or ollama)")
    return str(SECRETS_ENV)


# ═══════════════════════════════════════════════════════════════
# CLAUDE OAUTH2 PKCE FLOW
# ═══════════════════════════════════════════════════════════════


def generate_pkce_challenge() -> tuple:
    """Generate PKCE code verifier and challenge."""
    code_verifier = (
        base64.urlsafe_b64encode(secrets.token_bytes(32)).decode("utf-8").rstrip("=")
    )
    code_challenge = (
        base64.urlsafe_b64encode(hashlib.sha256(code_verifier.encode("utf-8")).digest())
        .decode("utf-8")
        .rstrip("=")
    )
    return code_verifier, code_challenge


def build_auth_url(code_challenge: str, code_verifier: str) -> str:
    """Build the browser auth URL."""
    auth_params = {
        "code": "true",
        "client_id": CLIENT_ID,
        "code_challenge_method": "S256",
        "code_challenge": code_challenge,
        "redirect_uri": REDIRECT_URI,
        "response_type": "code",
        "scope": "org:create_api_key user:profile user:inference",
        "state": code_verifier,
    }
    return f"{AUTH_URL}?{urllib.parse.urlencode(auth_params)}"


def exchange_code_for_token(auth_code: str, code_verifier: str) -> str:
    """Exchange authorization code for a Bearer access token."""
    code_parts = auth_code.split("#")
    actual_code = code_parts[0]
    state = code_parts[1] if len(code_parts) > 1 else code_verifier

    token_data = {
        "code": actual_code,
        "state": state,
        "grant_type": "authorization_code",
        "client_id": CLIENT_ID,
        "redirect_uri": REDIRECT_URI,
        "code_verifier": code_verifier,
    }

    data = json.dumps(token_data).encode("utf-8")
    req = urllib.request.Request(
        TOKEN_URL,
        data=data,
        headers={
            "Content-Type": "application/json",
            "User-Agent": "reqwest/0.11.0",
        },
    )

    try:
        with urllib.request.urlopen(req) as response:
            result = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        error_body = ""
        try:
            error_body = e.read().decode("utf-8")
        except Exception:
            pass
        raise RuntimeError(
            f"Token exchange failed (HTTP {e.code}): {e.reason}\n{error_body}"
        )

    if "access_token" in result:
        return result["access_token"]
    else:
        raise RuntimeError(f"Token exchange failed: {result}")


def verify_claude_token(token: str) -> bool:
    """Quick check that the Bearer token works."""
    try:
        request_data = {
            "model": "claude-sonnet-4-20250514",
            "max_tokens": 16,
            "messages": [{"role": "user", "content": "Say 'ok'"}],
        }
        data = json.dumps(request_data).encode("utf-8")
        req = urllib.request.Request(
            f"{CLAUDE_API_BASE}/messages",
            data=data,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {token}",
                "anthropic-version": "2023-06-01",
                "anthropic-beta": "oauth-2025-04-20",
            },
        )
        with urllib.request.urlopen(req, timeout=15) as response:
            result = json.loads(response.read().decode("utf-8"))
            return "content" in result
    except Exception:
        return False


# ═══════════════════════════════════════════════════════════════
# OLLAMA
# ═══════════════════════════════════════════════════════════════


def ollama_is_running() -> bool:
    """Check if Ollama is reachable at localhost:11434."""
    try:
        req = urllib.request.Request(f"{OLLAMA_BASE_URL}/api/tags")
        with urllib.request.urlopen(req, timeout=5) as response:
            return response.status == 200
    except Exception:
        return False


def ollama_list_models() -> List[str]:
    """Get list of locally available Ollama models."""
    try:
        req = urllib.request.Request(f"{OLLAMA_BASE_URL}/api/tags")
        with urllib.request.urlopen(req, timeout=10) as response:
            result = json.loads(response.read().decode("utf-8"))
            return [m["name"] for m in result.get("models", [])]
    except Exception:
        return []


def ollama_verify_model(model: str) -> bool:
    """Check that a specific model is available in Ollama."""
    models = ollama_list_models()
    # Match with or without :latest tag
    for m in models:
        if m == model or m.startswith(f"{model}:"):
            return True
    return False


def call_ollama(
    model: str,
    system_prompt: str,
    user_message: str,
    max_tokens: int = 4096,
) -> str:
    """
    Call Ollama's chat API with the given model.
    Uses /api/chat endpoint (OpenAI-compatible chat format).
    """
    request_data = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ],
        "stream": False,
        "options": {
            "num_predict": max_tokens,
            "temperature": 0.8,
        },
    }

    data = json.dumps(request_data).encode("utf-8")
    req = urllib.request.Request(
        f"{OLLAMA_BASE_URL}/api/chat",
        data=data,
        headers={"Content-Type": "application/json"},
    )

    with urllib.request.urlopen(req, timeout=120) as response:
        result = json.loads(response.read().decode("utf-8"))

    if "message" in result and "content" in result["message"]:
        return result["message"]["content"]
    else:
        raise RuntimeError(f"Unexpected Ollama response: {result}")


# ═══════════════════════════════════════════════════════════════
# UNIFIED CALL — picks the right backend automatically
# ═══════════════════════════════════════════════════════════════


def call_llm(
    system_prompt: str,
    user_message: str,
    max_tokens: int = 4096,
    backend: Optional[str] = None,
) -> str:
    """
    Call the active LLM backend (Claude or Ollama).

    If backend is not specified, reads LLM_BACKEND from secrets.env.
    Falls back: Claude token present -> claude, Ollama running -> ollama.

    This is THE function everything should use.
    """
    if backend is None:
        backend = load_active_backend()

    # Auto-detect if not configured
    if not backend:
        token = load_token()
        if token:
            backend = "claude"
        elif ollama_is_running():
            backend = "ollama"
        else:
            raise RuntimeError(
                "No LLM backend configured. Run oauth_setup.py or "
                "start Ollama and run: python oauth_setup.py --ollama"
            )

    if backend == "claude":
        token = load_token()
        if not token:
            raise RuntimeError(
                "Claude backend selected but no CLAUDE_OAUTH_TOKEN found."
            )
        return call_claude(token, system_prompt, user_message, max_tokens)
    elif backend == "ollama":
        model = load_ollama_model() or OLLAMA_DEFAULT_MODEL
        if not ollama_is_running():
            raise RuntimeError(
                "Ollama backend selected but Ollama is not running on localhost:11434."
            )
        return call_ollama(model, system_prompt, user_message, max_tokens)
    else:
        raise RuntimeError(f"Unknown LLM backend: {backend}")


def call_claude(
    token: str,
    system_prompt: str,
    user_message: str,
    max_tokens: int = 4096,
) -> str:
    """Make a Claude API call using the OAuth Bearer token."""
    request_data = {
        "model": "claude-sonnet-4-20250514",
        "max_tokens": max_tokens,
        "temperature": 0.8,
        "system": system_prompt,
        "messages": [{"role": "user", "content": user_message}],
    }

    data = json.dumps(request_data).encode("utf-8")
    req = urllib.request.Request(
        f"{CLAUDE_API_BASE}/messages",
        data=data,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {token}",
            "anthropic-version": "2023-06-01",
            "anthropic-beta": "oauth-2025-04-20",
        },
    )

    with urllib.request.urlopen(req, timeout=60) as response:
        result = json.loads(response.read().decode("utf-8"))

    if "content" in result and result["content"]:
        return result["content"][0]["text"]
    else:
        raise RuntimeError(f"Unexpected Claude API response: {result}")


# ═══════════════════════════════════════════════════════════════
# INTERACTIVE CLI FLOWS
# ═══════════════════════════════════════════════════════════════


def run_oauth_flow() -> str:
    """Interactive OAuth2 PKCE flow. Returns access token."""
    print("=" * 60)
    print("  Claude OAuth2 Setup for Soul System")
    print("=" * 60)
    print()
    print("This authenticates via the Claude Max OAuth2 PKCE flow")
    print("and stores the Bearer token for soul creation + self-realization.")
    print()

    code_verifier, code_challenge = generate_pkce_challenge()
    auth_url = build_auth_url(code_challenge, code_verifier)

    print("Opening browser for authentication...")
    webbrowser.open(auth_url)
    print()
    print("After authorizing, copy the ENTIRE code (including # parts).")
    print()

    auth_code = input("Paste authorization code: ").strip()
    if not auth_code:
        print("No code provided. Aborting.")
        sys.exit(1)

    print("\nExchanging code for Bearer token...")

    try:
        token = exchange_code_for_token(auth_code, code_verifier)
    except Exception as e:
        print(f"ERROR: {e}")
        sys.exit(1)

    path = write_token_to_secrets(token)
    print(f"Bearer token written to: {path}")

    print("Verifying token...")
    if verify_claude_token(token):
        print("Token verified — Claude API reachable.")
    else:
        print("WARNING: Could not verify. Token may still work.")

    print(f"\nBackend set to: claude")
    print("Run: python soul_creation_gui.py")
    return token


def run_ollama_setup(model: Optional[str] = None) -> str:
    """Interactive Ollama setup. Returns model name."""
    print("=" * 60)
    print("  Ollama Setup for Soul System")
    print("=" * 60)
    print()

    if not ollama_is_running():
        print("ERROR: Ollama is not running on localhost:11434.")
        print("Start it with: ollama serve")
        sys.exit(1)

    models = ollama_list_models()
    if not models:
        print("ERROR: No models found. Pull one with: ollama pull llama3.1")
        sys.exit(1)

    print(f"Ollama is running. Available models:")
    for i, m in enumerate(models, 1):
        print(f"  {i}. {m}")
    print()

    if model:
        chosen = model
    else:
        choice = input(
            f"Enter model name or number [default: {OLLAMA_DEFAULT_MODEL}]: "
        ).strip()
        if not choice:
            chosen = OLLAMA_DEFAULT_MODEL
        elif choice.isdigit() and 1 <= int(choice) <= len(models):
            chosen = models[int(choice) - 1]
        else:
            chosen = choice

    # Verify
    if not ollama_verify_model(chosen):
        print(f"WARNING: Model '{chosen}' not found locally. It may need to be pulled.")
        print(f"  ollama pull {chosen}")

    path = write_ollama_to_secrets(chosen)
    print(f"\nOllama model '{chosen}' written to: {path}")
    print(f"Backend set to: ollama")
    print("Run: python soul_creation_gui.py")
    return chosen


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--ollama":
        model = sys.argv[2] if len(sys.argv) > 2 else None
        run_ollama_setup(model)
    elif len(sys.argv) > 2 and sys.argv[1] == "--code":
        code_verifier, code_challenge = generate_pkce_challenge()
        try:
            token = exchange_code_for_token(sys.argv[2], code_verifier)
            path = write_token_to_secrets(token)
            print(f"Token written to {path}")
        except Exception as e:
            print(f"ERROR: {e}")
            sys.exit(1)
    else:
        run_oauth_flow()
