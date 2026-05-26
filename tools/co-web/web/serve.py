#!/usr/bin/env python3
"""Simple HTTP server for the Choreo web playground with proper MIME types."""
import http.server
import sys

PORT = int(sys.argv[1]) if len(sys.argv) > 1 else 8080

class ChoreoHandler(http.server.SimpleHTTPRequestHandler):
    extensions_map = {
        **http.server.SimpleHTTPRequestHandler.extensions_map,
        '.wasm': 'application/wasm',
        '.js': 'application/javascript',
        '.mjs': 'application/javascript',
    }

    def end_headers(self):
        self.send_header('Cross-Origin-Opener-Policy', 'same-origin')
        self.send_header('Cross-Origin-Embedder-Policy', 'require-corp')
        super().end_headers()

if __name__ == '__main__':
    with http.server.HTTPServer(('', PORT), ChoreoHandler) as httpd:
        print(f'Serving Choreo Playground at http://localhost:{PORT}')
        print('Press Ctrl+C to stop.')
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print('\nShutting down.')
