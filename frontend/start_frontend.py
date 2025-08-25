#!/usr/bin/env python3
"""
Simple HTTP server to serve the frontend files
"""

import http.server
import socketserver
import os
import webbrowser
from pathlib import Path

# Get the directory where this script is located
FRONTEND_DIR = Path(__file__).parent
PORT = 3000

class CustomHTTPRequestHandler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(FRONTEND_DIR), **kwargs)
    
    def end_headers(self):
        # Add CORS headers
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        super().end_headers()

def start_frontend_server():
    """Start the frontend HTTP server."""
    os.chdir(FRONTEND_DIR)
    
    ports_to_try = [PORT, 3001, 3002, 3003, 3004, 3005, 8000, 8001]
    
    for port in ports_to_try:
        try:
            with socketserver.TCPServer(("", port), CustomHTTPRequestHandler) as httpd:
                print(f"🌐 Frontend server started on port {port}")
                print(f"   URL: http://localhost:{port}")
                print(f"   Serving files from: {FRONTEND_DIR}")
                print("   Press Ctrl+C to stop the server")
                
                # Open browser automatically (only on first port)
                if port == PORT:
                    try:
                        webbrowser.open(f'http://localhost:{port}')
                    except:
                        print(f"   Please open your browser and navigate to: http://localhost:{port}")
                
                try:
                    httpd.serve_forever()
                except KeyboardInterrupt:
                    print("\n🛑 Shutting down frontend server...")
                    httpd.shutdown()
                    break
                    
        except OSError as e:
            if "Only one usage of each socket address" in str(e):
                print(f"⚠️  Port {port} is already in use, trying next port...")
                continue
            else:
                print(f"❌ Failed to start server on port {port}: {e}")
                break
        except Exception as e:
            print(f"❌ Unexpected error on port {port}: {e}")
            break
    
    # If we get here, all ports failed
    print(f"❌ Failed to start frontend server on any port from {ports_to_try}")
    print("   Please check if any other services are using these ports")
    print("   You can also try manually starting with: python start_frontend.py")

if __name__ == "__main__":
    start_frontend_server()
