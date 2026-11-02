"""Read-only system information and local date/time tools."""
from __future__ import annotations
import ctypes
import json
import os
import platform
from collections.abc import Mapping
from datetime import datetime
from core.agent import ToolSpec
from tools.registry import ToolRegistry

_EMPTY_SCHEMA: dict[str, object] = {
    "type": "object", "properties": {}, "required": [], "additionalProperties": False,
}


def get_system_info(arguments: Mapping[str, object] | None = None) -> str:
    """Return basic OS, CPU, and total memory details without spawning processes."""
    if arguments not in (None, {}):
        raise ValueError("get_system_info takes no arguments")
    info = {
        "os": platform.system(),
        "os_release": platform.release(),
        "architecture": platform.machine(),
        "processor": platform.processor() or None,
        "logical_cpu_count": os.cpu_count(),
        "total_memory_bytes": _total_memory_bytes(),
    }
    return json.dumps(info, sort_keys=True)


def get_datetime(arguments: Mapping[str, object] | None = None) -> str:
    """Return the current local date/time and timezone in ISO 8601 form."""
    if arguments not in (None, {}):
        raise ValueError("get_datetime takes no arguments")
    current = datetime.now().astimezone()
    return json.dumps({"datetime": current.isoformat(), "timezone": current.tzname()},
                      sort_keys=True)


def _total_memory_bytes() -> int | None:
    if os.name == "nt":
        class MemoryStatus(ctypes.Structure):
            _fields_ = [
                ("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
                ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong),
                ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
            ]
        status = MemoryStatus()
        status.dwLength = ctypes.sizeof(MemoryStatus)
        if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
            return int(status.ullTotalPhys)
        return None
    try:
        return os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
    except (AttributeError, OSError, ValueError):
        return None


def create_default_tool_registry() -> ToolRegistry:
    registry = ToolRegistry()
    registry.register(ToolSpec("get_system_info",
        "Return basic operating system, CPU, and memory details.", _EMPTY_SCHEMA), get_system_info)
    registry.register(ToolSpec("get_datetime",
        "Return the current local date, time, and timezone.", _EMPTY_SCHEMA), get_datetime)
    return registry
