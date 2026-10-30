import errno
import json
import logging
import os
import platform
import select
import time
from typing import Callable, Dict, Any, Optional

import socket
import threading
from typing import Dict, Any, Optional


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

        self._wait_for_connection()

    def _wait_for_connection(self):
        self.logger.info(f"Viewer socket listening on {self.host}:{self.port}")
        self.logger.info("Start Balatro on your PC and press 'R' to connect...")

        self._server_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._server_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._server_sock.bind((self.host, self.port))
        self._server_sock.listen(1)

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

        self.create_pipes()
        self.open_persistent_handles()

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
        self.logger.info("Press 'R' in Balatro now to activate RL training!")

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
        self.logger.info("Press 'R' in Balatro now to activate RL training!")

        req_fd = None
        res_fd = None

        def open_request():
            nonlocal req_fd
            req_fd = os.open(self.request_pipe, os.O_RDONLY)

        def open_response():
            nonlocal res_fd
            res_fd = os.open(self.response_pipe, os.O_WRONLY)

        # Open both ends concurrently to avoid blocking
        req_thread = threading.Thread(target=open_request)
        res_thread = threading.Thread(target=open_response)
        req_thread.start()
        res_thread.start()
        req_thread.join()
        res_thread.join()

        self.logger.info("Waiting for Balatro to connect to pipes...")
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


class ManagedPipeIO(BalatroPipeIO):
    """
    Linux FIFO transport for launcher-managed instances (one pipe pair per worker).

    Unlike BalatroPipeIO it never blocks forever: opening and reading poll the
    game process, so a crashed or hung instance surfaces as a None request
    that the environment turns into an instance restart.
    """

    POLL_SECS = 1.0

    def __init__(self, worker_id, alive_check: Callable[[], bool], read_timeout: float = 300.0):
        self.logger = logging.getLogger(__name__)
        self.worker_id = worker_id
        self.alive_check = alive_check
        self.read_timeout = read_timeout
        self.request_handle = None
        self.response_handle = None
        self._req_fd = None
        self._buf = b""
        self.request_pipe = f'/tmp/balatro_request_{worker_id}'
        self.response_pipe = f'/tmp/balatro_response_{worker_id}'
        self.create_pipes()

    def connect(self, timeout: float = 120.0) -> None:
        """Open both FIFO ends once the game process has opened its side."""
        self.cleanup_handles()
        # O_RDWR keeps our own writer on the request FIFO: open never blocks and reads never hit EOF
        self._req_fd = os.open(self.request_pipe, os.O_RDWR)
        self._buf = b""
        deadline = time.monotonic() + timeout
        while True:
            try:
                res_fd = os.open(self.response_pipe, os.O_WRONLY | os.O_NONBLOCK)
                break
            except OSError as e:
                if e.errno != errno.ENXIO:
                    raise
            if not self.alive_check():
                raise RuntimeError(f"Worker {self.worker_id}: game exited before connecting")
            if time.monotonic() > deadline:
                raise RuntimeError(f"Worker {self.worker_id}: game did not connect within {timeout}s")
            time.sleep(0.1)
        os.set_blocking(res_fd, True)
        self.response_handle = os.fdopen(res_fd, 'w', encoding='utf-8')
        self.logger.info(f"Worker {self.worker_id}: Balatro connected")

    def wait_for_request(self) -> Optional[Dict[str, Any]]:
        if self._req_fd is None:
            return None
        deadline = time.monotonic() + self.read_timeout
        while b"\n" not in self._buf:
            ready, _, _ = select.select([self._req_fd], [], [], self.POLL_SECS)
            if ready:
                self._buf += os.read(self._req_fd, 65536)
                continue
            if not self.alive_check():
                self.logger.error(f"Worker {self.worker_id}: game process died")
                return None
            if time.monotonic() > deadline:
                self.logger.error(f"Worker {self.worker_id}: no request for {self.read_timeout}s (hung)")
                return None
        line, self._buf = self._buf.split(b"\n", 1)
        try:
            return json.loads(line.decode("utf-8"))
        except json.JSONDecodeError as e:
            self.logger.error(f"Worker {self.worker_id}: invalid JSON in request: {e}")
            return None

    def send_response(self, response_data: Dict[str, Any]) -> bool:
        try:
            return super().send_response(response_data)
        except BrokenPipeError:
            return False

    def cleanup_handles(self):
        super().cleanup_handles()
        if self._req_fd is not None:
            try:
                os.close(self._req_fd)
            except OSError:
                pass
            self._req_fd = None
