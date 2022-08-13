import os
import yaml

if 'CODE_SERVER_PASSWORD' in os.environ:
    os.makedirs('/root/.config/code-server', exist_ok=True)
    with open('/root/.config/code-server/config.yaml', 'w') as f:
        yaml.safe_dump({
        'bind-addr': '0.0.0.0',
        'auth': 'password',
        'user-data-dir': '/workspace/.code-server',
        'cert': False,
        'password': os.environ['CODE_SERVER_PASSWORD']
    }, f)

os.system('code-server /workspace &')
os.system('python -m http.server 8080 --directory /workspace &')

if os.path.exists('/workspace/start.sh'):
    os.system('bash /workspace/start.sh')
