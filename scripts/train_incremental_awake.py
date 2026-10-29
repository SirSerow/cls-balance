"""Process-lifetime idle-sleep guard for sequential Phase 7 training."""

import ctypes
import sys


def main() -> None:
    request = None
    if sys.platform == "win32":
        request = ctypes.WinDLL("kernel32", use_last_error=True).SetThreadExecutionState
        request.argtypes = [ctypes.c_uint]
        request.restype = ctypes.c_uint
        if not request(0x80000001):
            raise ctypes.WinError(ctypes.get_last_error())
        print("Holding temporary Windows idle-sleep request; power plan unchanged.", flush=True)
    try:
        from cls_balance.incremental import main as incremental_main
        incremental_main()
    finally:
        if request:
            request(0x80000000)


if __name__ == "__main__":
    main()
