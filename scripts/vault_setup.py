"""
CLI tool to manage secrets in macOS Keychain.

Usage:
    python -m scripts.vault_setup set OPENROUTER_API_KEY sk-or-v1-your-key
    python -m scripts.vault_setup get OPENROUTER_API_KEY
    python -m scripts.vault_setup list
    python -m scripts.vault_setup delete OPENROUTER_API_KEY
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agents.vault import Vault


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    vault = Vault()
    command = sys.argv[1]

    if command == "set":
        if len(sys.argv) != 4:
            print("Usage: python -m scripts.vault_setup set <KEY> <VALUE>")
            sys.exit(1)
        key, value = sys.argv[2], sys.argv[3]
        vault.set(key, value)
        print(f"Stored '{key}' in Keychain (service='herd-logic').")

    elif command == "get":
        if len(sys.argv) != 3:
            print("Usage: python -m scripts.vault_setup get <KEY>")
            sys.exit(1)
        key = sys.argv[2]
        try:
            value = vault.get(key)
            print(value)
        except KeyError as e:
            print(f"ERROR: {e}")
            sys.exit(1)

    elif command == "list":
        keys = vault.list_keys()
        if keys:
            print("Secrets stored in Keychain (service='herd-logic'):")
            for k in keys:
                print(f"  - {k}")
        else:
            print("No secrets stored yet.")

    elif command == "delete":
        if len(sys.argv) != 3:
            print("Usage: python -m scripts.vault_setup delete <KEY>")
            sys.exit(1)
        key = sys.argv[2]
        try:
            vault.delete(key)
            print(f"Deleted '{key}' from Keychain.")
        except KeyError as e:
            print(f"ERROR: {e}")
            sys.exit(1)

    else:
        print(f"Unknown command: {command}")
        print(__doc__)
        sys.exit(1)


if __name__ == "__main__":
    main()
