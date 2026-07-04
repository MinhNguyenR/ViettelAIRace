from __future__ import annotations

import json
import platform
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


@dataclass
class Check:
    name: str
    ok: bool
    detail: str


def _run(args: list[str], timeout: float = 8.0) -> tuple[bool, str]:
    try:
        proc = subprocess.run(
            args,
            check=False,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=timeout,
        )
    except FileNotFoundError:
        return False, "not found"
    except subprocess.TimeoutExpired:
        return False, "timeout"
    return proc.returncode == 0, proc.stdout.strip()


def _command_check(name: str, args: list[str]) -> Check:
    if shutil.which(args[0]) is None:
        return Check(name, False, f"{args[0]} not found")
    ok, output = _run(args)
    first_line = output.splitlines()[0] if output else ""
    return Check(name, ok, first_line)


def _nvidia_query() -> Check:
    if shutil.which("nvidia-smi") is None:
        return Check("nvidia-smi", False, "nvidia-smi not found")
    ok, output = _run(
        [
            "nvidia-smi",
            "--query-gpu=name,memory.total,driver_version,pci.bus_id",
            "--format=csv,noheader,nounits",
        ]
    )
    if ok:
        return Check("nvidia-smi", True, output or "GPU visible")
    return Check("nvidia-smi", False, output.splitlines()[0] if output else "GPU not visible")


def run_doctor() -> int:
    inside_container = Path("/.dockerenv").exists()
    checks = [Check("python", sys.version_info >= (3, 11), platform.python_version())]
    if not inside_container:
        checks.extend(
            [
                _command_check("docker", ["docker", "--version"]),
                _command_check("docker compose", ["docker", "compose", "version"]),
            ]
        )
    checks.append(_nvidia_query())
    if not inside_container:
        checks.append(_command_check("nvidia-container-cli", ["nvidia-container-cli", "--version"]))

    mode = "container" if inside_container else "host"
    print(f"LLM Race Lab Doctor ({mode})")
    print("=" * 48)
    for check in checks:
        status = "OK" if check.ok else "WARN"
        print(f"{status:4} {check.name:22} {check.detail}")

    print("\nJSON")
    print(json.dumps([check.__dict__ for check in checks], ensure_ascii=False, indent=2))

    hard_required = {"python"} if inside_container else {"python", "docker"}
    hard_fail = [check for check in checks if check.name in hard_required and not check.ok]
    return 1 if hard_fail else 0
