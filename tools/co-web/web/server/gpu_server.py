#!/usr/bin/env python3
"""
Choreo GPU Execution Server

REST API for compiling and running Choreo .co programs on a GPU.
Designed to be deployed on a machine with NVIDIA GPU + CUDA toolkit.

Endpoints:
  POST /api/compile    - Compile .co source, return generated CUDA/C++
  POST /api/run        - Compile and execute, return stdout/stderr
  GET  /api/status     - Server health and GPU info
"""

import json
import os
import subprocess
import tempfile
import time
from http.server import HTTPServer, BaseHTTPRequestHandler
from pathlib import Path

CHOREO_BIN = os.environ.get('CHOREO_BIN', './choreo')
DEFAULT_TARGET = os.environ.get('CHOREO_TARGET', 'cute')
DEFAULT_ARCH = os.environ.get('CHOREO_ARCH', 'sm_90a')
MAX_SOURCE_SIZE = 1024 * 1024  # 1MB
EXECUTION_TIMEOUT = 30  # seconds
PORT = int(os.environ.get('CHOREO_SERVER_PORT', '8081'))


def get_gpu_info():
    """Get NVIDIA GPU info via nvidia-smi."""
    try:
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=name,memory.total,driver_version',
             '--format=csv,noheader'],
            capture_output=True, text=True, timeout=5
        )
        if result.returncode == 0:
            lines = result.stdout.strip().split('\n')
            gpus = []
            for line in lines:
                parts = [p.strip() for p in line.split(',')]
                if len(parts) >= 3:
                    gpus.append({
                        'name': parts[0],
                        'memory': parts[1],
                        'driver': parts[2]
                    })
            return gpus
    except (subprocess.TimeoutExpired, FileNotFoundError):
        pass
    return []


def compile_source(source, target=None, arch=None, flags=''):
    """Compile .co source, return generated code or errors."""
    target = target or DEFAULT_TARGET
    arch = arch or DEFAULT_ARCH

    with tempfile.NamedTemporaryFile(
        mode='w', suffix='.co', delete=False
    ) as f:
        f.write(source)
        src_path = f.name

    try:
        cmd = [CHOREO_BIN, '-t', target, '-es']
        if arch:
            cmd.append(f'-arch={arch}')
        if flags:
            cmd.extend(flags.split())
        cmd.append(src_path)

        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=EXECUTION_TIMEOUT
        )
        return {
            'success': result.returncode == 0,
            'output': result.stdout,
            'errors': result.stderr,
            'returncode': result.returncode
        }
    except subprocess.TimeoutExpired:
        return {
            'success': False,
            'output': '',
            'errors': 'Compilation timed out.',
            'returncode': -1
        }
    finally:
        os.unlink(src_path)


def compile_and_run(source, target=None, arch=None, flags=''):
    """Compile and execute a .co program, return results."""
    target = target or DEFAULT_TARGET
    arch = arch or DEFAULT_ARCH

    with tempfile.NamedTemporaryFile(
        mode='w', suffix='.co', delete=False
    ) as f:
        f.write(source)
        src_path = f.name

    script_path = src_path + '.sh'

    try:
        # Generate script
        cmd = [CHOREO_BIN, '-gs', '-t', target, f'-arch={arch}']
        if flags:
            cmd.extend(flags.split())
        cmd.extend([src_path, '-o', script_path])

        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=EXECUTION_TIMEOUT
        )
        if result.returncode != 0:
            return {
                'success': False,
                'output': result.stdout,
                'errors': result.stderr,
                'returncode': result.returncode,
                'phase': 'compile'
            }

        # Execute
        result = subprocess.run(
            ['bash', script_path, '--execute'],
            capture_output=True, text=True, timeout=EXECUTION_TIMEOUT,
            env={**os.environ, 'CHOREO_DISABLE_TIMING': '0'}
        )
        return {
            'success': result.returncode == 0,
            'output': result.stdout,
            'errors': result.stderr,
            'returncode': result.returncode,
            'phase': 'execute'
        }
    except subprocess.TimeoutExpired:
        return {
            'success': False,
            'output': '',
            'errors': 'Execution timed out.',
            'returncode': -1,
            'phase': 'timeout'
        }
    finally:
        for p in [src_path, script_path]:
            if os.path.exists(p):
                os.unlink(p)


class ChoreoAPIHandler(BaseHTTPRequestHandler):
    def do_OPTIONS(self):
        self.send_response(204)
        self._cors_headers()
        self.end_headers()

    def do_GET(self):
        if self.path == '/api/status':
            self._handle_status()
        else:
            self._send_error(404, 'Not found')

    def do_POST(self):
        if self.path == '/api/compile':
            self._handle_compile()
        elif self.path == '/api/run':
            self._handle_run()
        else:
            self._send_error(404, 'Not found')

    def _handle_status(self):
        gpus = get_gpu_info()
        choreo_version = 'unknown'
        try:
            result = subprocess.run(
                [CHOREO_BIN, '--help'],
                capture_output=True, text=True, timeout=5
            )
            if 'choreo' in result.stdout.lower():
                choreo_version = 'available'
        except (subprocess.TimeoutExpired, FileNotFoundError):
            choreo_version = 'not found'

        self._send_json({
            'status': 'ok',
            'gpus': gpus,
            'choreo': choreo_version,
            'target': DEFAULT_TARGET,
            'arch': DEFAULT_ARCH
        })

    def _handle_compile(self):
        body = self._read_body()
        if body is None:
            return

        source = body.get('source', '')
        if not source:
            self._send_error(400, 'No source code provided')
            return
        if len(source) > MAX_SOURCE_SIZE:
            self._send_error(400, 'Source code too large')
            return

        result = compile_source(
            source,
            target=body.get('target'),
            arch=body.get('arch'),
            flags=body.get('flags', '')
        )
        self._send_json(result)

    def _handle_run(self):
        body = self._read_body()
        if body is None:
            return

        source = body.get('source', '')
        if not source:
            self._send_error(400, 'No source code provided')
            return
        if len(source) > MAX_SOURCE_SIZE:
            self._send_error(400, 'Source code too large')
            return

        result = compile_and_run(
            source,
            target=body.get('target'),
            arch=body.get('arch'),
            flags=body.get('flags', '')
        )
        self._send_json(result)

    def _read_body(self):
        content_length = int(self.headers.get('Content-Length', 0))
        if content_length == 0:
            self._send_error(400, 'Empty request body')
            return None
        if content_length > MAX_SOURCE_SIZE * 2:
            self._send_error(413, 'Request too large')
            return None

        try:
            raw = self.rfile.read(content_length)
            return json.loads(raw)
        except json.JSONDecodeError:
            self._send_error(400, 'Invalid JSON')
            return None

    def _send_json(self, data):
        body = json.dumps(data).encode()
        self.send_response(200)
        self._cors_headers()
        self.send_header('Content-Type', 'application/json')
        self.send_header('Content-Length', str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_error(self, code, message):
        body = json.dumps({'error': message}).encode()
        self.send_response(code)
        self._cors_headers()
        self.send_header('Content-Type', 'application/json')
        self.send_header('Content-Length', str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _cors_headers(self):
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')

    def log_message(self, format, *args):
        print(f"[{time.strftime('%H:%M:%S')}] {format % args}")


if __name__ == '__main__':
    print(f"Choreo GPU Server starting on port {PORT}")
    print(f"  CHOREO_BIN: {CHOREO_BIN}")
    print(f"  Target: {DEFAULT_TARGET}, Arch: {DEFAULT_ARCH}")

    gpus = get_gpu_info()
    if gpus:
        for i, gpu in enumerate(gpus):
            print(f"  GPU {i}: {gpu['name']} ({gpu['memory']})")
    else:
        print("  WARNING: No GPU detected")

    server = HTTPServer(('', PORT), ChoreoAPIHandler)
    print(f"\nListening on http://0.0.0.0:{PORT}")
    print("Endpoints:")
    print("  GET  /api/status   - Server status and GPU info")
    print("  POST /api/compile  - Compile .co source")
    print("  POST /api/run      - Compile and execute .co source")
    print("\nPress Ctrl+C to stop.")

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down.")
        server.shutdown()
