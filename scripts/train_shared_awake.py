"""Run shared training with a process-lifetime Windows idle-sleep request.

Releases the request on exit. Does not alter power plans, display, or clocks.
On other platforms this simply invokes the same training entry point.
"""

import ctypes
import multiprocessing
import runpy
import sys


def identity(value: int) -> int:
    return value


def main() -> None:
    request = None
    if sys.platform == "win32":
        kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        request = kernel.SetThreadExecutionState
        request.argtypes = [ctypes.c_uint]
        request.restype = ctypes.c_uint
        if not request(0x80000001):
            raise ctypes.WinError(ctypes.get_last_error())
        print("Holding temporary Windows idle-sleep request; power plan unchanged.", flush=True)
    try:
        if sys.argv[1:] == ["--smoke-test"]:
            with multiprocessing.get_context("spawn").Pool(1) as pool:
                assert pool.map(identity, [7]) == [7]
            print("Spawn-safe power guard passed.", flush=True)
        else:
            sys.argv = ["cls_balance.train_shared", *sys.argv[1:]]
            runpy.run_module("cls_balance.train_shared", run_name="__main__")
    finally:
        if request:
            request(0x80000000)


if __name__ == "__main__":
    main()
