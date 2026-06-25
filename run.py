import socket

import uvicorn


HOST = "127.0.0.1"
DEFAULT_PORT = 8000
MAX_PORT = 8010


def is_port_available(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(0.2)
        return sock.connect_ex((HOST, port)) != 0


def find_available_port(start_port: int = DEFAULT_PORT) -> int:
    for port in range(start_port, MAX_PORT + 1):
        if is_port_available(port):
            return port

    raise RuntimeError(f"No available port found from {start_port} to {MAX_PORT}")


if __name__ == "__main__":
    port = find_available_port()
    if port != DEFAULT_PORT:
        print(f"Port {DEFAULT_PORT} is occupied. Using port {port} instead.")

    print(f"Open http://{HOST}:{port}/")
    uvicorn.run("backend.main:app", host=HOST, port=port, reload=True)
