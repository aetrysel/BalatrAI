"""
Debug utilities for timestamped print statements and debugging helpers
"""

import datetime

# dprint lines are hidden unless debug output is on (settings menu "Debug output", or key 'd' during training)
_debug_output = False


def set_debug_output(enabled: bool) -> None:
    """Show (True) or hide (False) dprint lines."""
    global _debug_output
    _debug_output = bool(enabled)


def debug_output_enabled() -> bool:
    return _debug_output


def timestamp() -> str:
    """
    Get current timestamp formatted as HH:MM:SS.mmm
    
    Returns:
        str: Formatted timestamp string
    """
    now = datetime.datetime.now()
    return now.strftime("%H:%M:%S.%f")[:-3]  # Remove last 3 digits to get milliseconds


def tprint(*args, **kwargs):
    """
    Print with timestamp prefix
    
    Args:
        *args: Arguments to print
        **kwargs: Keyword arguments for print()
    """
    print(f"[{timestamp()}]", *args, **kwargs)


def dprint(prefix: str, *args, **kwargs):
    """
    Debug print with custom prefix and timestamp
    
    Args:
        prefix: Custom prefix string
        *args: Arguments to print  
        **kwargs: Keyword arguments for print()
    """
    # print(f"[{timestamp()}] [{prefix}]", *args, **kwargs)  # previous: always printed
    if _debug_output:
        print(f"[{timestamp()}] [{prefix}]", *args, **kwargs)