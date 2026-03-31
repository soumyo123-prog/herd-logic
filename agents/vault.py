"""
Vault — secure secret storage using macOS Keychain.

Instead of storing API keys in .env files (which can accidentally get committed
to git), we store them in the macOS Keychain. The Keychain is:
- Encrypted at rest
- Protected by your system password
- Never at risk of being git-committed

Usage:
    from agents.vault import Vault

    vault = Vault()

    # Store a secret (one-time setup)
    vault.set("OPENROUTER_API_KEY", "sk-or-v1-your-key")

    # Retrieve a secret (used at runtime)
    key = vault.get("OPENROUTER_API_KEY")
"""

import subprocess


# All keys are stored under this service name in Keychain
SERVICE_NAME = "herd-logic"


class Vault:
    def __init__(self, service: str = SERVICE_NAME):
        self.service = service

    def get(self, key: str) -> str:
        """Retrieve a secret from macOS Keychain."""
        result = subprocess.run(
            ["security", "find-generic-password", "-s", self.service, "-a", key, "-w"],
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            raise KeyError(
                f"Secret '{key}' not found in Keychain (service='{self.service}'). "
                f"Add it with: python -m scripts.vault_setup set {key} <value>"
            )
        return result.stdout.strip()

    def set(self, key: str, value: str) -> None:
        """Store a secret in macOS Keychain. Overwrites if it already exists."""
        # Delete existing entry if present (security add fails on duplicates)
        subprocess.run(
            ["security", "delete-generic-password", "-s", self.service, "-a", key],
            capture_output=True,
        )
        result = subprocess.run(
            ["security", "add-generic-password", "-s", self.service, "-a", key, "-w", value],
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            raise RuntimeError(f"Failed to store secret '{key}': {result.stderr}")

    def delete(self, key: str) -> None:
        """Remove a secret from macOS Keychain."""
        result = subprocess.run(
            ["security", "delete-generic-password", "-s", self.service, "-a", key],
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            raise KeyError(f"Secret '{key}' not found in Keychain.")

    def list_keys(self) -> list[str]:
        """List all keys stored under our service name."""
        result = subprocess.run(
            ["security", "dump-keychain"],
            capture_output=True,
            text=True,
        )
        keys = []
        in_our_service = False
        for line in result.stdout.splitlines():
            if f'"svce"<blob>="{self.service}"' in line:
                in_our_service = True
            elif in_our_service and '"acct"<blob>=' in line:
                # Extract the key name from: "acct"<blob>="OPENROUTER_API_KEY"
                key = line.split('=')[1].strip().strip('"')
                keys.append(key)
                in_our_service = False
            elif 'class:' in line:
                in_our_service = False
        return keys
