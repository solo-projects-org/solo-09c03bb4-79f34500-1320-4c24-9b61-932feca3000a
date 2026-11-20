import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))


def pytest_configure(config):
    # 密集大图用例（6001 节点），耗时较长但保证覆盖 256 MiB 内存上限场景
    config.addinivalue_line("markers", "timeout_sensitive: dense large-graph scenarios")
