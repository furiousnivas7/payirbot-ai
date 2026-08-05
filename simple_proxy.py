#!/usr/bin/env python3
import socket
import select
from socketserver import ThreadingMixIn, TCPServer, StreamRequestHandler
from urllib.parse import urlsplit

BUF_SIZE = 8192

class ProxyHandler(StreamRequestHandler):
    def handle(self):
        data = self.connection.recv(BUF_SIZE)
        if not data:
            return

        first_line = data.splitlines()[0].decode(errors="ignore")
        if first_line.startswith("CONNECT"):
            self.handle_connect(first_line)
        else:
            self.handle_http(data, first_line)

    def handle_connect(self, first_line):
        try:
            _, address, _ = first_line.split()
            host, port = address.split(":")
            port = int(port)
        except Exception:
            self.send_error(400)
            return

        try:
            remote = socket.create_connection((host, port), timeout=10)
        except Exception:
            self.send_error(502)
            return

        self.connection.sendall(b"HTTP/1.1 200 Connection Established\r\n\r\n")
        self._forward(self.connection, remote)

    def handle_http(self, request_data, first_line):
        parts = first_line.split()
        if len(parts) < 2:
            self.send_error(400)
            return

        method, url = parts[0], parts[1]
        parsed = urlsplit(url)
        host = parsed.hostname
        port = parsed.port
        path = parsed.path or "/"
        if parsed.query:
            path += "?" + parsed.query

        if not host:
            headers = request_data.decode(errors="ignore").splitlines()
            for header in headers:
                if header.lower().startswith("host:"):
                    host_header = header.split(":", 1)[1].strip()
                    if ":" in host_header:
                        host, port = host_header.split(":", 1)
                        port = int(port)
                    else:
                        host = host_header
                        port = 80
                    break

        if not host:
            self.send_error(400)
            return

        if port is None:
            port = 443 if parsed.scheme == "https" else 80

        try:
            remote = socket.create_connection((host, port), timeout=10)
        except Exception:
            self.send_error(502)
            return

        if parsed.scheme:
            first_line = f"{method} {path} HTTP/1.1\r\n".encode()
            rest = request_data.split(b"\r\n", 1)[1]
            remote.sendall(first_line + rest)
        else:
            remote.sendall(request_data)

        self._forward(self.connection, remote)

    def _forward(self, client, remote):
        client.setblocking(False)
        remote.setblocking(False)
        sockets = [client, remote]

        while True:
            r, _, _ = select.select(sockets, [], sockets, 1)
            if not r:
                continue
            if client in r:
                data = client.recv(BUF_SIZE)
                if not data:
                    break
                remote.sendall(data)
            if remote in r:
                data = remote.recv(BUF_SIZE)
                if not data:
                    break
                client.sendall(data)

        remote.close()
        client.close()

    def send_error(self, code):
        self.connection.sendall(f"HTTP/1.1 {code} Error\r\nContent-Length: 0\r\n\r\n".encode())


class ThreadedTCPServer(ThreadingMixIn, TCPServer):
    allow_reuse_address = True
    daemon_threads = True


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Simple local HTTP/HTTPS proxy")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=54008)
    args = parser.parse_args()

    with ThreadedTCPServer((args.host, args.port), ProxyHandler) as server:
        print(f"Proxy listening on http://{args.host}:{args.port}")
        server.serve_forever()
