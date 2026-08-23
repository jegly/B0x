"""Subprocess sandboxing: layered, runtime-probed, honestly reported.

Engine-tier, pure Python (ctypes syscalls, no gi) — used by the
llama-server supervisor and the stable-diffusion.cpp subprocess, and
written for reuse by any future subprocess backend.

Box runs on anyone's machine, and what actually enforces varies per
machine — measured on the dev box: Landlock compiled but absent from the
boot LSM list (EOPNOTSUPP), systemd user-level ``ProtectSystem=strict``
genuinely enforced while ``ProtectHome=``/``InaccessiblePaths=`` were
*silently ignored*. So nothing here is trusted by configuration alone:

- **Baseline** (always): no-new-privs, RLIMIT_CORE=0, PR_SET_PDEATHSIG.
- **Landlock** (when the LSM is active): the precise layer — per-spawn
  read grants on exactly the files served, TCP bind on one port, outbound
  connect denied. Applied by the child itself between fork and exec; no
  namespaces, no root, no helper — which is also why bubblewrap/firejail
  are NOT used: they need unprivileged userns, blocked by Ubuntu's
  AppArmor default (killed the Phase 4 code interpreter; don't go back).
- **systemd-run --user** (when Landlock isn't active and a user session
  exists): unit-level properties, each verified by a one-time per-boot
  enforcement probe — a property that fails its probe is still applied
  but reported as unverified, never counted as protection.

:func:`launch` picks the strongest available mechanism and returns the
process plus a :class:`SandboxReport` saying exactly what is (and is not)
in force. Fail-open happens only for "mechanism unavailable", loudly;
unexpected errors while applying rules fail the spawn.
"""
from __future__ import annotations

import ctypes
import dataclasses
import errno
import json
import os
import resource
import secrets
import shutil
import struct
import subprocess
from pathlib import Path
from typing import Callable

__all__ = [
    "LaunchedProcess",
    "Policy",
    "SandboxError",
    "SandboxReport",
    "landlock_abi",
    "launch",
    "make_preexec",
    "probe_systemd_properties",
    "systemd_user_available",
]

# x86_64 syscall numbers (amd64-only product; other arches degrade).
_SYS_LANDLOCK_CREATE_RULESET = 444
_SYS_LANDLOCK_ADD_RULE = 445
_SYS_LANDLOCK_RESTRICT_SELF = 446

_LANDLOCK_CREATE_RULESET_VERSION = 1

_RULE_PATH_BENEATH = 1
_RULE_NET_PORT = 2

_FS_EXECUTE = 1 << 0
_FS_WRITE_FILE = 1 << 1
_FS_READ_FILE = 1 << 2
_FS_READ_DIR = 1 << 3
_FS_REMOVE_DIR = 1 << 4
_FS_REMOVE_FILE = 1 << 5
_FS_MAKE_CHAR = 1 << 6
_FS_MAKE_DIR = 1 << 7
_FS_MAKE_REG = 1 << 8
_FS_MAKE_SOCK = 1 << 9
_FS_MAKE_FIFO = 1 << 10
_FS_MAKE_BLOCK = 1 << 11
_FS_MAKE_SYM = 1 << 12
_FS_REFER = 1 << 13  # ABI 2
_FS_TRUNCATE = 1 << 14  # ABI 3
_FS_IOCTL_DEV = 1 << 15  # ABI 5

_NET_BIND_TCP = 1 << 0  # ABI 4
_NET_CONNECT_TCP = 1 << 1  # ABI 4

_SCOPE_ABSTRACT_UNIX_SOCKET = 1 << 0  # ABI 6
_SCOPE_SIGNAL = 1 << 1  # ABI 6

_PR_SET_NO_NEW_PRIVS = 38
_PR_SET_PDEATHSIG = 1
_SIGTERM = 15

_READ = _FS_READ_FILE
_READ_DIR_ACC = _FS_READ_FILE | _FS_READ_DIR
_EXEC_DIR_ACC = _READ_DIR_ACC | _FS_EXECUTE
_WRITE_FILE_ACC = _FS_WRITE_FILE | _FS_TRUNCATE
_WRITE_DIR_ACC = (
    _READ_DIR_ACC
    | _FS_WRITE_FILE
    | _FS_REMOVE_DIR
    | _FS_REMOVE_FILE
    | _FS_MAKE_DIR
    | _FS_MAKE_REG
    | _FS_MAKE_SYM
    | _FS_MAKE_FIFO
    | _FS_MAKE_SOCK
    | _FS_REFER
    | _FS_TRUNCATE
)


def _fs_mask_for_abi(abi: int) -> int:
    mask = (1 << 13) - 1  # ABI 1: EXECUTE..MAKE_SYM
    if abi >= 2:
        mask |= _FS_REFER
    if abi >= 3:
        mask |= _FS_TRUNCATE
    if abi >= 5:
        mask |= _FS_IOCTL_DEV
    return mask


_libc = ctypes.CDLL(None, use_errno=True)


def _syscall(nr: int, *args) -> int:
    res = _libc.syscall(ctypes.c_long(nr), *args)
    if res < 0:
        raise OSError(ctypes.get_errno(), os.strerror(ctypes.get_errno()))
    return res


def landlock_abi() -> int:
    """Highest Landlock ABI the running kernel supports, 0 if none."""
    try:
        return _syscall(
            _SYS_LANDLOCK_CREATE_RULESET,
            None,
            ctypes.c_size_t(0),
            ctypes.c_uint32(_LANDLOCK_CREATE_RULESET_VERSION),
        )
    except OSError:
        return 0


class SandboxError(Exception):
    """Policy couldn't be built (bad path) or applied (unexpected failure)."""


@dataclasses.dataclass(frozen=True)
class Policy:
    """What a sandboxed child is allowed to touch. Everything else: denied."""

    read_files: tuple[str, ...] = ()
    read_dirs: tuple[str, ...] = ()
    exec_dirs: tuple[str, ...] = ()
    write_files: tuple[str, ...] = ()
    write_dirs: tuple[str, ...] = ()
    bind_tcp: tuple[int, ...] = ()
    connect_tcp: tuple[int, ...] = ()

    def resolved_rules(self) -> list[tuple[str, int]]:
        """Canonicalized (path, access-mask) pairs; raises on missing paths."""
        out: list[tuple[str, int]] = []
        for paths, acc in (
            (self.read_files, _READ),
            (self.read_dirs, _READ_DIR_ACC),
            (self.exec_dirs, _EXEC_DIR_ACC),
            (self.write_files, _WRITE_FILE_ACC),
            (self.write_dirs, _WRITE_DIR_ACC),
        ):
            for p in paths:
                real = os.path.realpath(p)
                if not os.path.exists(real):
                    raise SandboxError(f"sandbox policy path does not exist: {p}")
                out.append((real, acc))
        return out

    @staticmethod
    def for_compute_subprocess(
        exec_dir: str | Path,
        read_files: tuple[str, ...] | list[str] = (),
        read_dirs: tuple[str, ...] | list[str] = (),
        write_dirs: tuple[str, ...] | list[str] = (),
    ) -> "Policy":
        """Policy for a one-shot compute subprocess with NO network at all
        (e.g. stable-diffusion.cpp's ``sd-cli``). Grants the system read-only
        baseline + the binary dir (executable), the given read files/dirs,
        and read-write on the given scratch/output dirs. Both TCP bind and
        connect stay denied."""
        base_read = tuple(
            d for d in ("/etc", "/proc", "/sys", "/dev") if os.path.isdir(d)
        )
        exec_dirs = tuple(
            str(d)
            for d in (exec_dir, "/usr/lib", "/lib", "/lib64", "/usr/lib64")
            if os.path.isdir(d)
        )
        return Policy(
            read_files=tuple(str(f) for f in read_files),
            read_dirs=base_read + tuple(str(d) for d in read_dirs),
            exec_dirs=exec_dirs,
            write_files=("/dev/null",) if os.path.exists("/dev/null") else (),
            write_dirs=tuple(str(d) for d in write_dirs),
            bind_tcp=(),
            connect_tcp=(),
        )

    @staticmethod
    def for_local_server(
        exec_dir: str | Path,
        model_files: tuple[str, ...] | list[str],
        port: int,
        write_dirs: tuple[str, ...] = (),
    ) -> "Policy":
        """Standard policy for a bundled local inference server: read-only on
        the binary dir + model file(s) + system baseline, TCP bind on exactly
        one port, no outbound connects, no writes outside ``write_dirs``."""
        read_dirs = tuple(
            d for d in ("/etc", "/proc", "/sys", "/dev") if os.path.isdir(d)
        )
        exec_dirs = tuple(
            str(d)
            for d in (exec_dir, "/usr/lib", "/lib", "/lib64", "/usr/lib64")
            if os.path.isdir(d)
        )
        return Policy(
            read_files=tuple(str(m) for m in model_files),
            read_dirs=read_dirs,
            exec_dirs=exec_dirs,
            write_files=("/dev/null",) if os.path.exists("/dev/null") else (),
            write_dirs=tuple(str(d) for d in write_dirs),
            bind_tcp=(port,),
            connect_tcp=(),
        )


def _apply_baseline() -> None:
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    if _libc.prctl(_PR_SET_NO_NEW_PRIVS, 1, 0, 0, 0) != 0:
        raise OSError(ctypes.get_errno(), "prctl(PR_SET_NO_NEW_PRIVS) failed")
    # Die with Box: if the app is SIGKILLed or crashes, the kernel delivers
    # SIGTERM to the child — no multi-GB model process left squatting in RAM.
    _libc.prctl(_PR_SET_PDEATHSIG, _SIGTERM, 0, 0, 0)


def _apply_landlock(
    abi: int, rules: list[tuple[str, int]], bind: tuple[int, ...], conn: tuple[int, ...]
) -> None:
    fs_mask = _fs_mask_for_abi(abi)
    handled_fs = fs_mask
    handled_net = (_NET_BIND_TCP | _NET_CONNECT_TCP) if abi >= 4 else 0
    scoped = (_SCOPE_ABSTRACT_UNIX_SOCKET | _SCOPE_SIGNAL) if abi >= 6 else 0

    if abi >= 6:
        attr = struct.pack("<QQQ", handled_fs, handled_net, scoped)
    elif abi >= 4:
        attr = struct.pack("<QQ", handled_fs, handled_net)
    else:
        attr = struct.pack("<Q", handled_fs)

    ruleset_fd = _syscall(
        _SYS_LANDLOCK_CREATE_RULESET,
        attr,
        ctypes.c_size_t(len(attr)),
        ctypes.c_uint32(0),
    )
    try:
        for path, access in rules:
            access &= fs_mask
            if not access:
                continue
            fd = os.open(path, os.O_PATH | os.O_CLOEXEC)
            try:
                battr = struct.pack("<Qi", access, fd)
                _syscall(
                    _SYS_LANDLOCK_ADD_RULE,
                    ctypes.c_int(ruleset_fd),
                    ctypes.c_int(_RULE_PATH_BENEATH),
                    battr,
                    ctypes.c_uint32(0),
                )
            finally:
                os.close(fd)
        if handled_net:
            for port_list, right in ((bind, _NET_BIND_TCP), (conn, _NET_CONNECT_TCP)):
                for port in port_list:
                    nattr = struct.pack("<QQ", right, port)
                    _syscall(
                        _SYS_LANDLOCK_ADD_RULE,
                        ctypes.c_int(ruleset_fd),
                        ctypes.c_int(_RULE_NET_PORT),
                        nattr,
                        ctypes.c_uint32(0),
                    )
        _syscall(
            _SYS_LANDLOCK_RESTRICT_SELF, ctypes.c_int(ruleset_fd), ctypes.c_uint32(0)
        )
    finally:
        os.close(ruleset_fd)


def make_preexec(policy: Policy) -> Callable[[], None]:
    """Build the ``preexec_fn`` for ``subprocess.Popen``."""
    rules = policy.resolved_rules()  # raises SandboxError in the parent
    abi = landlock_abi()
    bind, conn = tuple(policy.bind_tcp), tuple(policy.connect_tcp)

    def preexec() -> None:
        _apply_baseline()
        if abi <= 0:
            os.write(
                2,
                b"box-sandbox: WARNING kernel lacks Landlock; running with "
                b"baseline hardening only\n",
            )
            return
        try:
            _apply_landlock(abi, rules, bind, conn)
        except OSError as exc:
            if exc.errno in (errno.ENOSYS, errno.EOPNOTSUPP):
                os.write(
                    2,
                    b"box-sandbox: WARNING Landlock unavailable; running with "
                    b"baseline hardening only\n",
                )
                return
            raise  # unexpected → fail the spawn, never fail open silently

    return preexec


# ── systemd-run --user layer ────────────────────────────────────────────────
_PY_NET_PROBE = (
    "import socket\n"
    "s = socket.socket(); s.settimeout(3)\n"
    "try:\n"
    "    s.connect(('192.0.2.1', 9))\n"
    "    print('OPEN')\n"
    "except PermissionError:\n"
    "    print('ENFORCED')\n"
    "except OSError as e:\n"
    "    import errno as E\n"
    "    print('ENFORCED' if e.errno == E.EPERM else 'OPEN')\n"
)
_PY_AF_PROBE = (
    "import socket\n"
    "try:\n"
    "    socket.socket(16, 3)\n"
    "    print('OPEN')\n"
    "except OSError:\n"
    "    print('ENFORCED')\n"
)
_PY_WX_PROBE = (
    "import mmap\n"
    "try:\n"
    "    mmap.mmap(-1, 4096, prot=mmap.PROT_READ|mmap.PROT_WRITE|mmap.PROT_EXEC)\n"
    "    print('OPEN')\n"
    "except (PermissionError, OSError):\n"
    "    print('ENFORCED')\n"
)
_PY_FS_PROBE = (
    "try:\n"
    "    open('/usr/.box-sandbox-probe', 'w')\n"
    "    print('OPEN')\n"
    "except OSError:\n"
    "    print('ENFORCED')\n"
)

_PROBES: dict[str, tuple[tuple[str, ...], str]] = {
    "ProtectSystem=strict": (("ProtectSystem=strict",), _PY_FS_PROBE),
    "IPAddressDeny=any": (
        ("IPAddressDeny=any", "IPAddressAllow=localhost"),
        _PY_NET_PROBE,
    ),
    "RestrictAddressFamilies=AF_UNIX AF_INET AF_INET6": (
        ("RestrictAddressFamilies=AF_UNIX AF_INET AF_INET6",),
        _PY_AF_PROBE,
    ),
    "MemoryDenyWriteExecute=yes": (("MemoryDenyWriteExecute=yes",), _PY_WX_PROBE),
}

_UNPROBED_PROPS = (
    "NoNewPrivileges=yes",
    "LimitCORE=0",
    "PrivateTmp=yes",
    "RestrictNamespaces=yes",
    "LockPersonality=yes",
    "RestrictRealtime=yes",
)

_PROBE_TIMEOUT = 15


def systemd_user_available() -> bool:
    if shutil.which("systemd-run") is None:
        return False
    try:
        r = subprocess.run(
            ["systemctl", "--user", "is-system-running"],
            capture_output=True,
            timeout=5,
        )
    except (OSError, subprocess.TimeoutExpired):
        return False
    return r.returncode == 0 or r.stdout.strip() == b"degraded"


def _probe_cache_path() -> Path:
    runtime = os.environ.get("XDG_RUNTIME_DIR", f"/run/user/{os.getuid()}")
    return Path(runtime) / "box-sandbox-probes.json"


def probe_systemd_properties(use_cache: bool = True) -> dict[str, bool]:
    """Which candidate unit properties *actually enforce* on this machine."""
    cache = _probe_cache_path()
    if use_cache:
        try:
            data = json.loads(cache.read_text())
            if set(data) == set(_PROBES) and all(
                isinstance(v, bool) for v in data.values()
            ):
                return data
        except (OSError, ValueError):
            pass

    if not systemd_user_available():
        return {}

    results: dict[str, bool] = {}
    for name, (props, payload) in _PROBES.items():
        cmd = ["systemd-run", "--user", "--pipe", "--wait", "--collect", "--quiet"]
        for p in props:
            cmd += ["-p", p]
        cmd += ["/usr/bin/python3", "-I", "-c", payload]
        try:
            r = subprocess.run(cmd, capture_output=True, timeout=_PROBE_TIMEOUT)
            results[name] = b"ENFORCED" in r.stdout
        except (OSError, subprocess.TimeoutExpired):
            results[name] = False

    try:
        fd = os.open(cache, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        with os.fdopen(fd, "w") as f:
            json.dump(results, f)
    except OSError:
        pass
    return results


@dataclasses.dataclass
class SandboxReport:
    """What is actually in force for one launched child — Security page food."""

    mechanism: str  # "landlock" | "systemd" | "baseline"
    landlock_abi: int = 0
    verified: tuple[str, ...] = ()
    unverified: tuple[str, ...] = ()
    skipped: tuple[str, ...] = ()

    def summary(self) -> str:
        if self.mechanism == "landlock":
            return f"Landlock LSM (ABI v{self.landlock_abi}) + baseline hardening"
        if self.mechanism == "systemd":
            return (
                f"systemd user sandbox ({len(self.verified)} verified, "
                f"{len(self.unverified)} unverified properties) + baseline"
            )
        return "baseline hardening only (no kernel sandbox available)"


@dataclasses.dataclass
class LaunchedProcess:
    """A sandboxed child plus the handle needed to stop it properly."""

    popen: subprocess.Popen
    report: SandboxReport
    unit: str | None = None
    env_file: Path | None = None

    @property
    def pid(self) -> int:
        return self.popen.pid

    def cleanup_env_file(self) -> None:
        if self.env_file is not None:
            try:
                self.env_file.unlink(missing_ok=True)
            except OSError:
                pass
            self.env_file = None

    def terminate(self) -> None:
        if self.unit is not None:
            subprocess.run(
                ["systemctl", "--user", "stop", "--no-block", self.unit],
                capture_output=True,
                timeout=10,
            )
        else:
            try:
                self.popen.terminate()
            except ProcessLookupError:
                pass

    def kill(self) -> None:
        if self.unit is not None:
            subprocess.run(
                ["systemctl", "--user", "kill", "-s", "SIGKILL", self.unit],
                capture_output=True,
                timeout=10,
            )
        try:
            self.popen.kill()
        except (ProcessLookupError, OSError):
            pass


def launch(
    argv: list[str],
    policy: Policy,
    env: dict[str, str] | None = None,
    secret_env: dict[str, str] | None = None,
    stop_grace_seconds: int = 5,
) -> LaunchedProcess:
    """Spawn ``argv`` under the strongest sandbox this machine supports."""
    full_env = dict(os.environ)
    if env:
        full_env.update(env)

    abi = landlock_abi()
    if abi > 0:
        if secret_env:
            full_env.update(secret_env)
        popen = subprocess.Popen(
            argv,
            preexec_fn=make_preexec(policy),
            env=full_env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
        return LaunchedProcess(
            popen=popen,
            report=SandboxReport(mechanism="landlock", landlock_abi=abi),
        )

    if systemd_user_available():
        probes = probe_systemd_properties()
        verified = tuple(name for name, ok in probes.items() if ok)
        skipped = tuple(name for name, ok in probes.items() if not ok)
        unit = f"box-sbx-{secrets.token_hex(4)}.service"
        cmd = [
            "systemd-run", "--user", "--pipe", "--wait", "--collect", "--quiet",
            f"--unit={unit}", "-p", f"TimeoutStopSec={stop_grace_seconds}",
        ]
        # PrivateTmp gives the unit a private /tmp — which HIDES any write
        # target under /tmp. Drop it when a write target is under /tmp so the
        # output is visible afterward.
        write_targets = list(policy.write_dirs) + [
            str(Path(f).parent) for f in policy.write_files
            if f not in ("/dev/null",)
        ]
        under_tmp = any(
            str(Path(d).resolve()).startswith(("/tmp/", "/var/tmp/"))
            for d in write_targets
        )
        for prop in _UNPROBED_PROPS:
            if prop == "PrivateTmp=yes" and under_tmp:
                continue
            cmd += ["-p", prop]
        for name in verified:
            for p in _PROBES[name][0]:
                cmd += ["-p", p]
        # ProtectSystem=strict makes the WHOLE filesystem read-only (incl.
        # /home) — re-grant the policy's write dirs via ReadWritePaths, else
        # the subprocess can't write its output.
        for d in write_targets:
            rp = str(Path(d).resolve())
            if rp not in ("/dev", "/proc", "/sys"):
                cmd += ["-p", f"ReadWritePaths={rp}"]
        env_file: Path | None = None
        if env:
            for k, v in env.items():
                cmd += ["-p", f"Environment={k}={v}"]
        if secret_env:
            runtime = os.environ.get("XDG_RUNTIME_DIR", f"/run/user/{os.getuid()}")
            env_file = Path(runtime) / f"box-sbx-{secrets.token_hex(4)}.env"
            fd = os.open(env_file, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
            with os.fdopen(fd, "w") as f:
                for k, v in secret_env.items():
                    f.write(f"{k}={v}\n")
            cmd += ["-p", f"EnvironmentFile={env_file}"]
        # Lifeline: the unit outlives our process by design (systemd owns it),
        # so killing Box would orphan a multi-GB server. --pipe wires our
        # stdin through; this wrapper tears the server down on EOF, i.e. the
        # moment Box dies. (fd 3 dance: POSIX gives background jobs stdin from
        # /dev/null, so the watcher must read the real stdin via a dup.)
        lifeline = (
            'exec 3<&0; "$@" & srv=$!; '
            "{ while read -r _ <&3; do :; done; kill -TERM $srv 2>/dev/null; } & "
            "wait $srv"
        )
        cmd += ["/bin/sh", "-c", lifeline, "box-lifeline"]
        cmd += argv
        popen = subprocess.Popen(
            cmd,
            env=full_env,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
        return LaunchedProcess(
            popen=popen,
            report=SandboxReport(
                mechanism="systemd",
                verified=verified,
                unverified=_UNPROBED_PROPS,
                skipped=skipped,
            ),
            unit=unit,
            env_file=env_file,
        )

    if secret_env:
        full_env.update(secret_env)
    baseline_policy = Policy()

    popen = subprocess.Popen(
        argv,
        preexec_fn=make_preexec(baseline_policy),
        env=full_env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    return LaunchedProcess(popen=popen, report=SandboxReport(mechanism="baseline"))
