"""核心业务逻辑：CSV 解析校验、连通图构建与桥接（割点）识别。

本模块不依赖 FastAPI / SQLite，全部为纯函数，便于单独测试。

判定规则（与需求一致）：
- 同一路段、两条【未撤回】上报，平面距离 <= 5 米 且 UTC 时差 <= 10 分钟，则相连；
- 某条上报被移除后，其所在连通组分裂成至少两个非空组，该上报才是疑似桥接
  （即图论中的割点/关节点）；普通边缘点（叶子）移除后只剩一个组，不算桥接。
"""
from __future__ import annotations

import csv
import io
import math
from dataclasses import dataclass
from datetime import datetime, timezone

MAX_DISTANCE_M = 5.0  # 相距不超过 5 米
MAX_TIME_DIFF_S = 600.0  # 时差不超过 10 分钟


class CsvError(Exception):
    """CSV 校验失败。消息中可包含多行错误描述。"""


@dataclass(frozen=True)
class ReportRow:
    report_id: str
    revision: int
    segment_id: str
    x: float
    y: float
    ts_utc: str  # 归一化后的 UTC ISO-8601
    ts_epoch: float  # UTC 秒时间戳，用于时差计算
    withdrawn: bool


_HEADER_ALIASES = {
    "report_id": ("report_id", "上报号"),
    "revision": ("revision", "修订号"),
    "segment_id": ("segment_id", "路段号"),
    "x": ("x", "横坐标"),
    "y": ("y", "纵坐标"),
    "ts_utc": ("ts_utc", "utc时刻", "时刻"),
    "withdrawn": ("withdrawn", "撤回标记"),
}

_TRUE_VALUES = {"1", "true", "t", "yes", "y", "是"}
_FALSE_VALUES = {"0", "false", "f", "no", "n", "", "否"}


def _canon_header(name: str) -> str:
    return name.strip().lstrip("\ufeff").lower()


def _map_header(header: list[str]) -> dict[str, int]:
    alias = {}
    for canon, names in _HEADER_ALIASES.items():
        for n in names:
            alias[n] = canon
    colmap: dict[str, int] = {}
    errors: list[str] = []
    for idx, raw in enumerate(header):
        key = _canon_header(raw)
        if not key:
            continue
        canon = alias.get(key)
        if canon is None:
            errors.append(f"未知列名 {raw!r}")
        elif canon in colmap:
            errors.append(f"列 {canon!r} 在表头重复出现")
        else:
            colmap[canon] = idx
    missing = [c for c in _HEADER_ALIASES if c not in colmap]
    if missing:
        errors.append("缺少必需列: " + ", ".join(missing))
    if errors:
        allowed = " / ".join("/".join(v) for v in _HEADER_ALIASES.values())
        raise CsvError("表头错误：" + "；".join(errors) + f"。允许的列名：{allowed}")
    return colmap


def _parse_bool(value: str) -> bool:
    v = value.strip().lower()
    if v in _TRUE_VALUES:
        return True
    if v in _FALSE_VALUES:
        return False
    raise ValueError(f"撤回标记无法识别: {value!r}（可用 0/1、true/false、yes/no，留空视为未撤回）")


def _parse_ts(value: str) -> tuple[str, float]:
    v = value.strip()
    if not v:
        raise ValueError("UTC 时刻为空")
    try:
        dt = datetime.fromisoformat(v.replace("Z", "+00:00").replace("z", "+00:00"))
    except ValueError:
        raise ValueError(f"UTC 时刻无法解析: {value!r}（需 ISO 8601，如 2026-11-19T08:30:00Z）")
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)  # 无时区按 UTC 处理
    dt = dt.astimezone(timezone.utc)
    return dt.isoformat(), dt.timestamp()


def _parse_row(raw: list[str], colmap: dict[str, int]) -> ReportRow:
    def cell(canon: str) -> str:
        return raw[colmap[canon]].strip()

    report_id = cell("report_id")
    if not report_id:
        raise ValueError("上报号为空")
    segment_id = cell("segment_id")
    if not segment_id:
        raise ValueError("路段号为空")
    rev_raw = cell("revision")
    try:
        revision = int(rev_raw)
    except ValueError:
        raise ValueError(f"修订号不是整数: {rev_raw!r}")
    if revision < 0:
        raise ValueError(f"修订号不能为负: {revision}")
    try:
        x = float(cell("x"))
        y = float(cell("y"))
    except ValueError:
        raise ValueError(f"平面米坐标无法解析: {cell('x')!r}, {cell('y')!r}")
    if not (math.isfinite(x) and math.isfinite(y)):
        raise ValueError("平面米坐标必须是有限数值")
    ts_utc, ts_epoch = _parse_ts(cell("ts_utc"))
    withdrawn = _parse_bool(cell("withdrawn"))
    return ReportRow(report_id, revision, segment_id, x, y, ts_utc, ts_epoch, withdrawn)


def parse_csv(text: str) -> list[ReportRow]:
    """解析并校验整份 CSV。任何一行有问题都抛 CsvError（调用方应整体拒绝，
    不允许部分提交）。"""
    reader = csv.reader(io.StringIO(text))
    header = None
    for raw in reader:
        if raw and any(c.strip() for c in raw):
            header = raw
            break
    if header is None:
        raise CsvError("CSV 为空：缺少表头行")
    colmap = _map_header(header)
    width = len(header)

    rows: list[ReportRow] = []
    errors: list[str] = []
    seen: set[tuple[str, int]] = set()
    for raw in reader:
        if not raw or all(not c.strip() for c in raw):
            continue
        lineno = reader.line_num
        if len(raw) != width:
            errors.append(f"第 {lineno} 行：列数 {len(raw)} 与表头 {width} 不一致")
            continue
        try:
            row = _parse_row(raw, colmap)
        except ValueError as e:
            errors.append(f"第 {lineno} 行：{e}")
            continue
        key = (row.report_id, row.revision)
        if key in seen:
            errors.append(f"第 {lineno} 行：同一文件内重复的上报号+修订号 {key[0]}#{key[1]}")
            continue
        seen.add(key)
        rows.append(row)
    if errors:
        shown = "；".join(errors[:20])
        if len(errors) > 20:
            shown += f"；…共 {len(errors)} 处错误"
        raise CsvError(shown)
    if not rows:
        raise CsvError("CSV 中没有有效数据行")
    return rows


# ---------------------------------------------------------------- 图与桥接
#
# 密集路段（如 3000 节点两两相连的团）会产生 O(V^2) 条边，若把邻接关系物化成
# Python set，仅边对象就可达数 GB，256 MiB 内存上限下必然 OOM。这里改为：
# 1) 5 米空间哈希网格，邻居只可能落在本格及相邻 8 格，按需枚举、绝不存边（O(V) 内存）；
# 2) 单趟迭代式 Tarjan 求割点（无递归深度问题），并在 DFS 树上直接记录
#    low[child] >= disc[ap] 的孩子，分裂组由这些孩子的子树 + 祖先方向剩余节点组成，
#    无需为每个割点重新 BFS。


def connected(a, b) -> bool:
    """两条上报是否相连：距离 <= 5 米 且 时差 <= 10 分钟（含边界）。"""
    d = math.hypot(a["x"] - b["x"], a["y"] - b["y"])
    dt = abs(a["ts_epoch"] - b["ts_epoch"])
    return d <= MAX_DISTANCE_M and dt <= MAX_TIME_DIFF_S


_GRID_CELL = MAX_DISTANCE_M  # 格宽 = 距离阈值；相距 <=5m 的两点格号至多相差 1
_DISTANCE_SQ_LIMIT = MAX_DISTANCE_M * MAX_DISTANCE_M


def _build_grid(reports) -> dict[tuple[int, int], list[int]]:
    """空间哈希网格：格号 -> 节点下标列表。"""
    grid: dict[tuple[int, int], list[int]] = {}
    for i, r in enumerate(reports):
        key = (math.floor(r["x"] / _GRID_CELL), math.floor(r["y"] / _GRID_CELL))
        grid.setdefault(key, []).append(i)
    return grid


def _iter_neighbors(u: int, reports, grid: dict[tuple[int, int], list[int]]):
    """按需枚举节点 u 的全部邻居（生成器，不保存边）。

    距离 <=5m 的节点只能落在本格及周围 8 格；时差与距离在枚举时内联判定，
    避免每条候选边一次函数调用开销。每个 v 只属于一个格子，因此不会重复产出。
    """
    ru = reports[u]
    xu, yu, tu = ru["x"], ru["y"], ru["ts_epoch"]
    cx = math.floor(xu / _GRID_CELL)
    cy = math.floor(yu / _GRID_CELL)
    for gx in (cx - 1, cx, cx + 1):
        for gy in (cy - 1, cy, cy + 1):
            cell = grid.get((gx, gy))
            if cell is None:
                continue
            for v in cell:
                if v == u:
                    continue
                rv = reports[v]
                dt = tu - rv["ts_epoch"]
                if dt < -MAX_TIME_DIFF_S or dt > MAX_TIME_DIFF_S:
                    continue
                dx = xu - rv["x"]
                dy = yu - rv["y"]
                if dx * dx + dy * dy <= _DISTANCE_SQ_LIMIT:
                    yield v


def _collect_subtree(root: int, children: list[list[int]]) -> list[int]:
    """收集 DFS 树上 root 子树的全部节点下标。"""
    out: list[int] = []
    stack = [root]
    while stack:
        u = stack.pop()
        out.append(u)
        stack.extend(children[u])
    return out


def find_bridges(reports) -> list[dict]:
    """找出疑似桥接上报（割点），内存占用 O(V)，可处理密集大图。

    输入：同一路段、未撤回的有效上报，元素为含 report_id/x/y/ts_epoch 的映射。
    输出：[{"report_id": ..., "groups": [[上报号...], ...]}]，按上报号排序；
    仅包含“移除后原连通组分裂成至少两个非空组”的上报，并给出分裂后各组名单
    （分裂组只统计其原连通组，其他不相干连通组不计入，叶子点因此不会被误判）。
    """
    n = len(reports)
    if n < 3:  # 少于 3 个节点，移除任一节点都不可能留下两个非空组
        return []

    grid = _build_grid(reports)
    disc = [-1] * n
    low = [0] * n
    parent = [-1] * n
    children: list[list[int]] = [[] for _ in range(n)]  # 仅 DFS 树边，总计 O(V)
    comp_id = [-1] * n
    comp_nodes: list[list[int]] = []
    timer = 0
    aps: set[int] = set()
    # 割点 -> 使 low[child] >= disc[ap] 成立的树孩子（DFS 根则取其全部树孩子）
    ap_children: dict[int, list[int]] = {}

    for start in range(n):
        if disc[start] != -1:
            continue
        cid = len(comp_nodes)
        members: list[int] = []
        disc[start] = low[start] = timer
        timer += 1
        comp_id[start] = cid
        members.append(start)
        root_children = 0
        stack = [(start, _iter_neighbors(start, reports, grid))]
        while stack:
            u, it = stack[-1]
            descended = False
            for v in it:
                if disc[v] == -1:
                    parent[v] = u
                    children[u].append(v)
                    disc[v] = low[v] = timer
                    timer += 1
                    comp_id[v] = cid
                    members.append(v)
                    if parent[u] == -1:
                        root_children += 1
                    stack.append((v, _iter_neighbors(v, reports, grid)))
                    descended = True
                    break
                if v != parent[u] and disc[v] < low[u]:
                    low[u] = disc[v]  # 回边（无向图忽略指向父亲的边）
            if descended:
                continue
            stack.pop()
            if stack:
                p = stack[-1][0]
                if low[u] < low[p]:
                    low[p] = low[u]
                if parent[p] != -1 and low[u] >= disc[p]:
                    aps.add(p)
                    ap_children.setdefault(p, []).append(u)
        comp_nodes.append(members)
        if root_children > 1:  # DFS 根：树孩子 >1 才是割点，每个孩子子树各成一组
            aps.add(start)
            ap_children[start] = children[start]

    marker = [-1] * n  # 标记已归入某割点分裂组的节点，O(V) 复用，无需每次建集合
    result: list[dict] = []
    for ap in sorted(aps, key=lambda i: reports[i]["report_id"]):
        groups: list[list[str]] = []
        for c in ap_children[ap]:
            subtree = _collect_subtree(c, children)
            for u in subtree:
                marker[u] = ap
            groups.append(sorted(reports[u]["report_id"] for u in subtree))
        if parent[ap] != -1:
            # 非根割点：祖先方向（含经由回边仍与祖先连通的兄弟子树）剩余节点
            rest = [
                reports[u]["report_id"]
                for u in comp_nodes[comp_id[ap]]
                if u != ap and marker[u] != ap
            ]
            if rest:
                groups.append(sorted(rest))
        groups.sort()
        if len(groups) >= 2:  # 至少两个非空组才算桥接；叶子点移除后只剩一组，排除
            result.append({"report_id": reports[ap]["report_id"], "groups": groups})
    return result
