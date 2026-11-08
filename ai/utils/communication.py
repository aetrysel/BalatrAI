import json
import logging
import os
import platform
from typing import Dict, Any, Optional

import select
import socket
import threading

from .lifecycle import (
    load_bridge_settings, remove_session_flag, write_session_flag, StopTrainingRequested,  # noqa: F401
)

def print_waiting_message() -> None:
    settings = load_bridge_settings()
    if settings["auto_start"]:
        print(f"Waiting for Balatro… (auto-wake, or {settings['start_key']} in game)")
    else:
        print(f"Waiting for Balatro… press {settings['start_key']} in game to start")


SHUTDOWN_RESPONSE = {"action": "SHUTDOWN"}
# After SHUTDOWN, wait briefly for a request Lua may already be sending so its write doesn't hit a closed pipe
SHUTDOWN_DRAIN_SECS = 0.5


class BalatroSocketIO:
    """
    TCP socket transport for Balatro communication.
    Drop-in replacement for BalatroPipeIO — identical interface.
    
    Used for the viewer instance where Balatro runs on a different
    machine (your PC) and connects to the training server over TCP.
    
    Python acts as the SERVER, Balatro (Lua) connects as the CLIENT.
    """

    def __init__(self, host: str = '0.0.0.0', port: int = 9000):
        self.logger = logging.getLogger(__name__)
        self.host = host
        self.port = port

        self.request_handle = None  # socket file object for reading
        self.response_handle = None  # socket file object for writing
        self._conn = None
        self._server_sock = None

        remove_session_flag()  # drop a stale flag from a crashed session
        self._wait_for_connection()

    def _wait_for_connection(self):
        self.logger.info(f"Viewer socket listening on {self.host}:{self.port}")

        self._server_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._server_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._server_sock.bind((self.host, self.port))
        self._server_sock.listen(1)

        write_session_flag()
        print_waiting_message()
        self._conn, addr = self._server_sock.accept()
        self._conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        self.logger.info(f"Balatro connected from {addr}")

        # Wrap socket in file objects — same readline()/write() interface as pipes
        self.request_handle = self._conn.makefile('r', encoding='utf-8')
        self.response_handle = self._conn.makefile('w', encoding='utf-8')

    # --- Identical interface to BalatroPipeIO below ---

    def wait_for_request(self) -> Optional[Dict[str, Any]]:
        if not self.request_handle:
            self.logger.error("Request handle not open")
            return None
        try:
            request_line = self.request_handle.readline().strip()
            if not request_line:
                return None
            request_data = json.loads(request_line)
            self.logger.debug(f"📥 RECEIVED: {request_line}")
            return request_data
        except json.JSONDecodeError as e:
            self.logger.error(f"Invalid JSON in request: {e}")
            return None
        except Exception as e:
            self.logger.error(f"Error reading from socket: {e}")
            return None

    def send_response(self, response_data: Dict[str, Any]) -> bool:
        if not self.response_handle:
            self.logger.error("Response handle not open")
            return False
        try:
            json.dump(response_data, self.response_handle)
            self.response_handle.write('\n')
            self.response_handle.flush()
            self.logger.debug(f"📤 SENT: {json.dumps(response_data)}")
            return True
        except Exception as e:
            self.logger.error(f"Error sending response: {e}")
            return False

    def send_shutdown(self) -> bool:
        """Best-effort SHUTDOWN response so Lua returns to the menu and idles. Never raises or hangs."""
        if not self._conn:
            return False
        try:
            self._conn.settimeout(1.0)
            self._conn.sendall((json.dumps(SHUTDOWN_RESPONSE) + "\n").encode("utf-8"))
            return True
        except Exception as e:
            self.logger.debug(f"SHUTDOWN not delivered: {e}")
            return False

    def cleanup_handles(self):
        for handle in [self.request_handle, self.response_handle]:
            if handle:
                try:
                    handle.close()
                except Exception:
                    pass
        self.request_handle = None
        self.response_handle = None
        if self._conn:
            try:
                self._conn.close()
            except Exception:
                pass
            self._conn = None

    def cleanup(self):
        remove_session_flag()
        self.cleanup_handles()
        if self._server_sock:
            try:
                self._server_sock.close()
            except Exception:
                pass
            self._server_sock = None
        self.logger.info("Socket communication closed")

class BalatroPipeIO:
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.request_handle = None
        self.response_handle = None
        
        if platform.system() == "Windows":
            self.request_pipe = r'\\.\pipe\balatro_request'
            self.response_pipe = r'\\.\pipe\balatro_response'
        else:
            self.request_pipe = '/tmp/balatro_request'
            self.response_pipe = '/tmp/balatro_response'

        remove_session_flag()  # drop a stale flag from a crashed session
        self.create_pipes()
        write_session_flag()
        try:
            self.open_persistent_handles()
        except BaseException:
            # Ctrl+C while waiting for Balatro: don't leave the flag or FIFOs behind
            self.cleanup()
            raise

    def create_pipes(self) -> None:
        if platform.system() == "Windows":
            # On Windows, named pipes are created via Win32 API
            # We do this in open_persistent_handles using pywin32
            self.logger.info("Windows: pipes will be created as named pipe server")
            return
        
        # Linux/WSL: use mkfifo
        for pipe_path in [self.request_pipe, self.response_pipe]:
            try:
                if not os.path.exists(pipe_path):
                    os.mkfifo(pipe_path)
                    self.logger.info(f"Created pipe: {pipe_path}")
                else:
                    self.logger.info(f"Pipe already exists: {pipe_path}")
                    self.logger.debug(f"Using existing pipe: {pipe_path}")
            except Exception as e:
                self.logger.error(f"Failed to create pipe {pipe_path}: {e}")
                raise RuntimeError(f"Could not create pipe: {pipe_path}")

    def open_persistent_handles(self) -> None:
        try:
            if platform.system() == "Windows":
                self._open_windows_pipes()
            else:
                self._open_unix_pipes()
        except Exception as e:
            self.logger.error(f"Failed to open persistent handles: {e}")
            self.cleanup_handles()
            raise RuntimeError(f"Could not open persistent pipe handles: {e}")

    def _open_windows_pipes(self):
        import msvcrt
        import pywintypes
        import win32pipe
        import win32file

        self.logger.info("Creating Windows named pipe servers...")
        print_waiting_message()

        self._req_pipe_handle = win32pipe.CreateNamedPipe(
            self.request_pipe,
            win32pipe.PIPE_ACCESS_INBOUND,
            win32pipe.PIPE_TYPE_MESSAGE | win32pipe.PIPE_READMODE_MESSAGE | win32pipe.PIPE_WAIT,
            1, 65536, 65536, 0, None
        )

        self._res_pipe_handle = win32pipe.CreateNamedPipe(
            self.response_pipe,
            win32pipe.PIPE_ACCESS_OUTBOUND,
            win32pipe.PIPE_TYPE_MESSAGE | win32pipe.PIPE_READMODE_MESSAGE | win32pipe.PIPE_WAIT,
            1, 65536, 65536, 0, None
        )

        self.logger.info("Waiting for Balatro to connect to pipes...")
        win32pipe.ConnectNamedPipe(self._req_pipe_handle, None)
        win32pipe.ConnectNamedPipe(self._res_pipe_handle, None)
        self.logger.info("Balatro connected!")

        # Use msvcrt.open_osfhandle instead of the broken GetOSHandle approach
        req_fd = msvcrt.open_osfhandle(int(self._req_pipe_handle), os.O_RDONLY)
        res_fd = msvcrt.open_osfhandle(int(self._res_pipe_handle), os.O_WRONLY)
        self.request_handle = os.fdopen(req_fd, 'r', encoding='utf-8')
        self.response_handle = os.fdopen(res_fd, 'w', encoding='utf-8')

    def _open_unix_pipes(self):
        import threading

        self.logger.info("Creating Unix pipes...")
        print_waiting_message()

        req_fd = None
        res_fd = None

        def open_request():
            nonlocal req_fd
            req_fd = os.open(self.request_pipe, os.O_RDONLY)

        def open_response():
            nonlocal res_fd
            res_fd = os.open(self.response_pipe, os.O_WRONLY)

        # Open both ends concurrently to avoid blocking
        # Daemon threads so Ctrl+C while waiting for Balatro can still exit the process
        req_thread = threading.Thread(target=open_request, daemon=True)
        res_thread = threading.Thread(target=open_response, daemon=True)
        req_thread.start()
        res_thread.start()
        req_thread.join()
        res_thread.join()

        self.request_handle = os.fdopen(req_fd, 'r', encoding='utf-8')
        self.response_handle = os.fdopen(res_fd, 'w', encoding='utf-8')
        self.logger.info("Balatro connected!")

    def wait_for_request(self) -> Optional[Dict[str, Any]]:
        if not self.request_handle:
            self.logger.error("Request handle not open")
            return None
        try:
            request_line = self.request_handle.readline().strip()
            if not request_line:
                return None
            request_data = json.loads(request_line)
            self.logger.debug(f"📥 RECEIVED: {request_line}")
            return request_data
        except json.JSONDecodeError as e:
            self.logger.error(f"Invalid JSON in request: {e}")
            return None
        except Exception as e:
            self.logger.error(f"Error reading from request pipe: {e}")
            return None

    def send_response(self, response_data: Dict[str, Any]) -> bool:
        if not self.response_handle:
            self.logger.error("Response handle not open")
            return False
        try:
            json.dump(response_data, self.response_handle)
            self.response_handle.write('\n')
            self.response_handle.flush()
            self.logger.debug(f"📤 SENT: {json.dumps(response_data)}")
            return True
        except Exception as e:
            self.logger.error(f"Error sending response: {e}")
            return False

    def send_shutdown(self) -> bool:
        """
        Best-effort SHUTDOWN response so Lua returns to the menu and idles. Never raises or hangs.

        Writes straight to the pipe fd in non-blocking mode (a 22-byte write fits the pipe buffer),
        so it is skipped silently if the write end is closed, nobody reads, or the pipe is full.
        Afterwards waits briefly for a request Lua may already be sending, so its write does not
        land on a closed pipe.
        """
        try:
            handle = self.response_handle
            if handle is None:
                return False
            data = (json.dumps(SHUTDOWN_RESPONSE) + "\n").encode("utf-8")
            if platform.system() == "Windows":
                handle.write(data.decode("utf-8"))
                handle.flush()
                return True
            fd = handle.fileno()
            os.set_blocking(fd, False)
            os.write(fd, data)
        except Exception as e:  # BrokenPipeError, BlockingIOError, closed file, ...
            self.logger.debug(f"SHUTDOWN not delivered: {e}")
            return False
        self._drain_pending_request()
        return True

    def _drain_pending_request(self) -> None:
        try:
            if self.request_handle is None:
                return
            ready, _, _ = select.select([self.request_handle.fileno()], [], [], SHUTDOWN_DRAIN_SECS)
            if ready:
                os.read(self.request_handle.fileno(), 1 << 20)
        except Exception:
            pass

    def cleanup_handles(self):
        for handle_name in ['request_handle', 'response_handle']:
            h = getattr(self, handle_name, None)
            if h:
                try:
                    h.close()
                    self.logger.debug(f"Closed {handle_name}")
                except Exception:
                    pass
                setattr(self, handle_name, None)

        # Also close Windows PyHANDLE objects if they exist
        if platform.system() == "Windows":
            for handle_name in ['_req_pipe_handle', '_res_pipe_handle']:
                h = getattr(self, handle_name, None)
                if h:
                    try:
                        import win32file
                        win32file.CloseHandle(h)
                    except Exception:
                        pass
                    setattr(self, handle_name, None)

    def cleanup(self):
        remove_session_flag()
        self.cleanup_handles()
        if platform.system() != "Windows":
            for pipe_path in [self.request_pipe, self.response_pipe]:
                try:
                    if os.path.exists(pipe_path):
                        os.unlink(pipe_path)
                        self.logger.debug(f"Removed pipe: {pipe_path}")
                except Exception as e:
                    self.logger.warning(f"Failed to remove pipe {pipe_path}: {e}")
        

            self.logger.debug("Dual pipe communication cleanup complete")
