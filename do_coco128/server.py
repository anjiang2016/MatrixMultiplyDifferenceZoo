"""
server.py
启动一个轻量级 HTTP 服务器，提供训练日志数据和静态页面。
"""

import http.server
import json
import os

PORT = 8000

class Handler(http.server.SimpleHTTPRequestHandler):
    def do_GET(self):
        # 处理 /data 请求，返回 JSON 日志
        if self.path == '/data':
            try:
                log_path='./training_log_class_weight.json'
                with open(log_path, 'r') as f:
                    data = json.load(f)
                print(f"尝试读取: {os.path.abspath(log_path)}")
                self.send_response(200)
                self.send_header('Content-type', 'application/json')
                self.send_header('Cache-Control', 'no-cache, no-store, must-revalidate')
                self.send_header('Pragma', 'no-cache')
                self.send_header('Expires', '0')
                self.end_headers()
                self.wfile.write(json.dumps(data).encode())
            except FileNotFoundError:
                self.send_response(404)
                self.end_headers()
                self.wfile.write(b'{"error": "log not found"}')
        else:
            # 否则作为静态文件服务（支持 index.html）
            super().do_GET()

def main():
    # 如果当前目录没有 index.html，自动创建一个示例
    if not os.path.exists('index.html'):
        print("警告: 未找到 index.html，请确保该文件存在。")

    server = http.server.HTTPServer(('', PORT), Handler)
    print(f"服务器运行在 http://localhost:{PORT}")
    print("访问 /data 查看 JSON 数据")
    server.serve_forever()

if __name__ == '__main__':
    main()
