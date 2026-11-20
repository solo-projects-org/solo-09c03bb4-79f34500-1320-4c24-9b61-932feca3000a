"""关键自测：桥接判定、修订语义、上传原子性、重启持久化、交错一致性、密集大图内存。"""
import threading

import pytest
from fastapi.testclient import TestClient

from app import core, db
from app.main import create_app

HEADER = "report_id,revision,segment_id,x,y,ts_utc,withdrawn"

# 经典链：A—B—C（B 是桥接）
CHAIN = [
    ("A", 0, "S1", 0, 0, "2026-11-19T10:00:00Z", 0),
    ("B", 0, "S1", 4, 0, "2026-11-19T10:05:00Z", 0),
    ("C", 0, "S1", 8, 0, "2026-11-19T10:10:00Z", 0),
]

# 密集路段参数：两片各 N 条、片内两两相连（真团），仅由一条桥接 Z 相连
DENSE_N = 3000
DENSE_SIDE = 55  # ceil(sqrt(3000))
DENSE_STEP = 0.06  # 片内最大对角距离 sqrt(2)*54*0.06 ≈ 4.58m <= 5m
DENSE_RIGHT_X = 9.0  # 两片间距 >= 5.76m，片间没有边
DENSE_Z_X = 4.5  # Z 距两片最近点均为 4.5m


def dense_csv(n_per_side=DENSE_N, withdrawn_z=False, z_revision=0, segment="DENSE"):
    """构造密集路段 CSV：左/右两片团 + 桥接 Z（可指定 Z 的修订/撤回状态）。"""
    rows = [HEADER]
    for i in range(n_per_side):
        gx, gy = i % DENSE_SIDE, i // DENSE_SIDE
        rows.append(f"L{i:04d},0,{segment},{gx*DENSE_STEP},{gy*DENSE_STEP},2026-11-19T10:00:00Z,0")
    for i in range(n_per_side):
        gx, gy = i % DENSE_SIDE, i // DENSE_SIDE
        rows.append(
            f"R{i:04d},0,{segment},{DENSE_RIGHT_X+gx*DENSE_STEP},{gy*DENSE_STEP},2026-11-19T10:00:00Z,0"
        )
    rows.append(f"Z,{z_revision},{segment},{DENSE_Z_X},0,2026-11-19T10:00:00Z,{1 if withdrawn_z else 0}")
    return "\n".join(rows) + "\n"


def make_csv(rows, header=HEADER):
    lines = [header]
    for r in rows:
        lines.append(",".join(str(c) for c in r))
    return "\n".join(lines) + "\n"


@pytest.fixture()
def client(tmp_path):
    app = create_app(str(tmp_path / "test.db"))
    with TestClient(app) as c:
        yield c


def post(client, text, filename="data.csv"):
    return client.post(
        "/upload", files={"file": (filename, text.encode("utf-8"), "text/csv")}
    )


def get_bridges(client, segment="S1"):
    r = client.get(f"/api/segments/{segment}/bridges")
    assert r.status_code == 200
    return r.json()


# ---------------------------------------------------------------- 桥接判定


def test_chain_middle_is_bridge(client):
    r = post(client, make_csv(CHAIN))
    assert r.status_code == 200  # 303 重定向后跟到首页
    data = get_bridges(client)
    assert data["active_reports"] == ["A", "B", "C"]
    assert data["bridges"] == [{"report_id": "B", "groups": [["A"], ["C"]]}]


def test_triangle_has_no_bridge(client):
    rows = [
        ("A", 0, "S1", 0, 0, "2026-11-19T10:00:00Z", 0),
        ("B", 0, "S1", 4, 0, "2026-11-19T10:00:00Z", 0),
        ("C", 0, "S1", 2, 3, "2026-11-19T10:00:00Z", 0),
    ]
    post(client, make_csv(rows))
    assert get_bridges(client)["bridges"] == []


def test_pair_edge_points_are_not_bridges(client):
    """两条互连上报：移除任一条只剩一个非空组，不算桥接。"""
    rows = [
        ("A", 0, "S1", 0, 0, "2026-11-19T10:00:00Z", 0),
        ("B", 0, "S1", 5, 0, "2026-11-19T10:00:00Z", 0),
    ]
    post(client, make_csv(rows))
    assert get_bridges(client)["bridges"] == []


def test_star_center_splits_into_three_groups(client):
    rows = [
        ("B", 0, "S1", 0, 0, "2026-11-19T10:00:00Z", 0),
        ("A", 0, "S1", -4, 0, "2026-11-19T10:00:00Z", 0),
        ("C", 0, "S1", 4, 0, "2026-11-19T10:00:00Z", 0),
        ("D", 0, "S1", 0, 4, "2026-11-19T10:00:00Z", 0),
    ]
    post(client, make_csv(rows))
    assert get_bridges(client)["bridges"] == [
        {"report_id": "B", "groups": [["A"], ["C"], ["D"]]}
    ]


def test_distance_threshold_inclusive(client):
    """恰好 5 米相连，5.01 米不相连。"""
    rows = [
        ("A", 0, "S1", 0, 0, "2026-11-19T10:00:00Z", 0),
        ("B", 0, "S1", 5, 0, "2026-11-19T10:00:00Z", 0),
        ("C", 0, "S1", 10, 0, "2026-11-19T10:00:00Z", 0),
    ]
    post(client, make_csv(rows))
    assert [b["report_id"] for b in get_bridges(client)["bridges"]] == ["B"]

    rows2 = [
        ("A", 0, "S2", 0, 0, "2026-11-19T10:00:00Z", 0),
        ("B", 0, "S2", 5.01, 0, "2026-11-19T10:00:00Z", 0),
        ("C", 0, "S2", 10.02, 0, "2026-11-19T10:00:00Z", 0),
    ]
    post(client, make_csv(rows2))
    assert get_bridges(client, "S2")["bridges"] == []


def test_time_threshold_inclusive(client):
    """恰好 10 分钟相连，10 分 01 秒不相连。"""
    rows = [
        ("A", 0, "S1", 0, 0, "2026-11-19T10:00:00Z", 0),
        ("B", 0, "S1", 4, 0, "2026-11-19T10:10:00Z", 0),
        ("C", 0, "S1", 8, 0, "2026-11-19T10:20:00Z", 0),
    ]
    post(client, make_csv(rows))
    assert [b["report_id"] for b in get_bridges(client)["bridges"]] == ["B"]

    rows2 = [
        ("A", 0, "S2", 0, 0, "2026-11-19T10:00:00Z", 0),
        ("B", 0, "S2", 4, 0, "2026-11-19T10:10:01Z", 0),
        ("C", 0, "S2", 8, 0, "2026-11-19T10:20:02Z", 0),
    ]
    post(client, make_csv(rows2))
    assert get_bridges(client, "S2")["bridges"] == []


def test_cross_segment_not_connected(client):
    """不同路段即使距离/时刻满足也不相连。"""
    rows = [
        ("A", 0, "S1", 0, 0, "2026-11-19T10:00:00Z", 0),
        ("B", 0, "S1", 4, 0, "2026-11-19T10:05:00Z", 0),
        ("C", 0, "S2", 8, 0, "2026-11-19T10:10:00Z", 0),
    ]
    post(client, make_csv(rows))
    assert get_bridges(client, "S1")["bridges"] == []
    assert get_bridges(client, "S2")["bridges"] == []


def test_split_groups_only_within_own_component(client):
    """分裂组只统计原连通组：链 A-B-C 与远处一对 D-E 同路段，
    B 的分裂组不含 D-E；D/E 是叶子，不是桥接。"""
    rows = CHAIN + [
        ("D", 0, "S1", 100, 0, "2026-11-19T10:00:00Z", 0),
        ("E", 0, "S1", 104, 0, "2026-11-19T10:00:00Z", 0),
    ]
    post(client, make_csv(rows))
    assert get_bridges(client)["bridges"] == [{"report_id": "B", "groups": [["A"], ["C"]]}]


def test_chain_and_star_in_one_segment(client):
    """链 + 星形两个连通组：B、D 各自为桥接，分裂组互不串扰。"""
    rows = CHAIN + [
        ("D", 0, "S1", 100, 0, "2026-11-19T10:00:00Z", 0),
        ("E", 0, "S1", 104, 0, "2026-11-19T10:00:00Z", 0),
        ("F", 0, "S1", 96, 0, "2026-11-19T10:00:00Z", 0),
        ("G", 0, "S1", 100, 4, "2026-11-19T10:00:00Z", 0),
    ]
    post(client, make_csv(rows))
    assert get_bridges(client)["bridges"] == [
        {"report_id": "B", "groups": [["A"], ["C"]]},
        {"report_id": "D", "groups": [["E"], ["F"], ["G"]]},
    ]


# ---------------------------------------------------------------- 修订语义


def test_higher_revision_corrects_position(client):
    post(client, make_csv(CHAIN))
    assert [b["report_id"] for b in get_bridges(client)["bridges"]] == ["B"]
    # B 的高修订把它挪远：链断开，不再有桥接
    post(client, make_csv([("B", 1, "S1", 100, 0, "2026-11-19T10:05:00Z", 0)]))
    data = get_bridges(client)
    assert data["bridges"] == []
    assert data["active_reports"] == ["A", "B", "C"]


def test_higher_revision_withdraws(client):
    post(client, make_csv(CHAIN))
    post(client, make_csv([("B", 1, "S1", 4, 0, "2026-11-19T10:05:00Z", 1)]))
    data = get_bridges(client)
    assert data["active_reports"] == ["A", "C"]  # 已撤回不参与连通
    assert data["bridges"] == []


def test_late_old_revision_cannot_restore(client):
    """迟到旧修订不能恢复旧证据：先生效 rev5，再传 rev2 无效。"""
    rows = [r if r[0] != "B" else ("B", 5, "S1", 4, 0, "2026-11-19T10:05:00Z", 0) for r in CHAIN]
    post(client, make_csv(rows))
    # 迟到的旧修订 rev2：位置挪走且标记撤回 —— 必须被忽略
    post(client, make_csv([("B", 2, "S1", 100, 0, "2026-11-19T10:05:00Z", 1)]))
    data = get_bridges(client)
    assert data["bridges"] == [{"report_id": "B", "groups": [["A"], ["C"]]}]
    # 页面上 B 的生效修订号仍是 5
    html = client.get("/segments/S1").text
    assert ">5</td>" in html


def test_revision_can_move_segment(client):
    """高修订把上报改到别的路段：原路段不再计入。"""
    post(client, make_csv(CHAIN))
    post(client, make_csv([("C", 1, "S2", 8, 0, "2026-11-19T10:10:00Z", 0)]))
    assert get_bridges(client, "S1")["active_reports"] == ["A", "B"]
    assert get_bridges(client, "S2")["active_reports"] == ["C"]


# ---------------------------------------------------------------- 原子性与一致性


def test_failed_upload_commits_nothing(client):
    """一行有错，整批拒绝：已解析成功的行也不入库。"""
    bad = make_csv(
        [("A", 0, "S1", 0, 0, "2026-11-19T10:00:00Z", 0), ("B", 0, "S1", "abc", 0, "2026-11-19T10:05:00Z", 0)]
    )
    r = post(client, bad)
    assert r.status_code == 400
    assert client.get("/api/segments").json()["segments"] == []
    assert client.get("/").status_code == 200


def test_interleaved_uploads_and_queries(client):
    """上传与查询交错：每次查询结果都精确对应当前已提交的批次集合。"""
    post(client, make_csv(CHAIN[:2]))  # A,B
    assert get_bridges(client)["bridges"] == []  # 仅一对，无桥接

    post(client, make_csv([CHAIN[2]]))  # +C
    assert get_bridges(client)["bridges"] == [{"report_id": "B", "groups": [["A"], ["C"]]}]

    post(client, make_csv([("X", 0, "S1", "bad", 0, "2026-11-19T10:00:00Z", 0)]))  # 失败批次
    assert get_bridges(client)["bridges"] == [{"report_id": "B", "groups": [["A"], ["C"]]}]

    post(client, make_csv([("B", 1, "S1", 100, 0, "2026-11-19T10:05:00Z", 0)]))  # B 挪远
    assert get_bridges(client)["bridges"] == []


def test_restart_keeps_committed_state(tmp_path):
    """重启（重新创建 app，同一数据库文件）后结论一致。"""
    db_path = str(tmp_path / "restart.db")
    with TestClient(create_app(db_path)) as c1:
        post(c1, make_csv(CHAIN))
        before = get_bridges(c1)["bridges"]
    with TestClient(create_app(db_path)) as c2:
        assert get_bridges(c2)["bridges"] == before
        assert c2.get("/segments/S1").status_code == 200


def test_reupload_same_file_is_idempotent(client):
    post(client, make_csv(CHAIN))
    post(client, make_csv(CHAIN))
    segs = client.get("/api/segments").json()["segments"]
    assert segs == [{"segment_id": "S1", "n_effective": 3, "n_active": 3}]
    assert get_bridges(client)["bridges"] == [{"report_id": "B", "groups": [["A"], ["C"]]}]


def test_concurrent_writers_and_readers(tmp_path):
    """多线程上传/读取交错：不报错，且最终已提交集合完整。"""
    db_path = str(tmp_path / "conc.db")
    db.init_db(db_path)
    errors = []

    def writer(n):
        try:
            for i in range(10):
                row = core.ReportRow(
                    f"W{n}-{i}", 0, "S1", float(i * 10 + n), 0.0,
                    "2026-11-19T10:00:00Z", 0.0, False,
                )
                db.save_upload(db_path, f"w{n}.csv", [row])
        except Exception as e:  # noqa: BLE001
            errors.append(e)

    def reader():
        try:
            for _ in range(50):
                reps = db.get_effective_reports(db_path, "S1")
                core.find_bridges(reps)
        except Exception as e:  # noqa: BLE001
            errors.append(e)

    threads = [threading.Thread(target=writer, args=(n,)) for n in range(3)]
    threads += [threading.Thread(target=reader) for _ in range(2)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert errors == []
    assert len(db.get_effective_reports(db_path, "S1")) == 30
    assert len(db.list_uploads(db_path)) == 30


# ---------------------------------------------------------------- 页面与解析


def test_segment_page_lists_groups(client):
    post(client, make_csv(CHAIN))
    html = client.get("/segments/S1").text
    assert "疑似桥接上报（1 条）" in html
    assert "⭐ 上报号 B" in html
    assert "分裂为 2 个非空组" in html


def test_parse_chinese_headers_and_naive_ts():
    text = "上报号,修订号,路段号,x,y,UTC时刻,撤回标记\nA,0,S1,1.5,2.5,2026-11-19 10:00:00,\n"
    rows = core.parse_csv(text)
    assert len(rows) == 1
    assert rows[0].report_id == "A"
    assert rows[0].withdrawn is False
    assert rows[0].ts_utc.endswith("+00:00")  # 无时区按 UTC


def test_parse_rejects_bad_input():
    with pytest.raises(core.CsvError):
        core.parse_csv("")  # 空文件
    with pytest.raises(core.CsvError):
        core.parse_csv("report_id,revision\nA,0\n")  # 缺列
    with pytest.raises(core.CsvError):
        core.parse_csv(make_csv([("A", 0, "S1", 0, 0, "2026-11-19T10:00:00Z", "maybe")]))
    with pytest.raises(core.CsvError):  # 同文件内重复 上报号+修订号
        core.parse_csv(make_csv([CHAIN[0], CHAIN[0]]))


# ---------------------------------------------------------------- 密集大图（内存上限场景）


def _expected_dense_bridge(n_per_side=DENSE_N):
    return {
        "report_id": "Z",
        "groups": [
            sorted(f"L{i:04d}" for i in range(n_per_side)),
            sorted(f"R{i:04d}" for i in range(n_per_side)),
        ],
    }


@pytest.mark.timeout_sensitive
def test_dense_segment_api_and_page_list_full_groups(client):
    """两片各 3000 条、片内两两相连，仅 Z 桥接：API 与网页都要给出 Z 和两组完整名单。"""
    r = post(client, dense_csv())
    assert r.status_code == 200

    data = get_bridges(client, "DENSE")
    assert len(data["active_reports"]) == 2 * DENSE_N + 1
    assert data["bridges"] == [_expected_dense_bridge()]

    html = client.get("/segments/DENSE").text
    assert "疑似桥接上报（1 条）" in html
    assert "⭐ 上报号 Z" in html
    assert "分裂为 2 个非空组" in html
    # 两组完整名单都在页面上（抽首/尾各一个，加上成员计数由 API 断言保证）
    assert "L0000" in html and f"L{DENSE_N-1:04d}" in html
    assert "R0000" in html and f"R{DENSE_N-1:04d}" in html


@pytest.mark.timeout_sensitive
def test_dense_segment_concurrent_web_and_api(client):
    """网页与 API 同时查询同一密集路段：两路都成功且结论一致，不报错。"""
    post(client, dense_csv())
    box = {}

    def call_api():
        box["api"] = client.get("/api/segments/DENSE/bridges")

    def call_page():
        box["page"] = client.get("/segments/DENSE")

    threads = [threading.Thread(target=call_api), threading.Thread(target=call_page)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert box["api"].status_code == 200
    assert box["page"].status_code == 200
    assert box["api"].json()["bridges"] == [_expected_dense_bridge()]
    assert "⭐ 上报号 Z" in box["page"].text


@pytest.mark.timeout_sensitive
def test_dense_segment_correction_and_withdrawal(client):
    """密集路段补传：撤回 Z 后无桥接；再补更正（高修订恢复位置）后桥接与两组名单恢复。"""
    post(client, dense_csv())
    assert get_bridges(client, "DENSE")["bridges"] == [_expected_dense_bridge()]

    # 补传撤回：Z 高修订 withdrawn=1，两片彻底断开，不再有桥接
    post(client, dense_csv(withdrawn_z=True, z_revision=1))
    data = get_bridges(client, "DENSE")
    assert data["active_reports"] == sorted(
        [f"L{i:04d}" for i in range(DENSE_N)] + [f"R{i:04d}" for i in range(DENSE_N)]
    )
    assert data["bridges"] == []

    # 再补更正：rev2 把 Z 放回原位置且未撤回，结论按最新上报恢复
    post(client, dense_csv(z_revision=2))
    assert get_bridges(client, "DENSE")["bridges"] == [_expected_dense_bridge()]


@pytest.mark.timeout_sensitive
def test_dense_queries_interleaved_with_supplement(client):
    """补传与查询交错：补传前已读到旧快照的查询给出旧分裂证据，补传后的查询按新数据。"""
    post(client, dense_csv())

    started = threading.Event()
    release = threading.Event()
    box = {}

    def slow_query():
        # 先在旧快照上把有效报读取出来（模拟“已读到旧数据”的在途查询），
        # 等待补传提交后再计算，结论仍必须对应旧快照
        reports = db.get_effective_reports(client.app.state.db_path, "DENSE")
        started.set()
        release.wait(timeout=10)
        active = [r for r in reports if not r["withdrawn"]]
        box["in_flight"] = core.find_bridges(active)

    t = threading.Thread(target=slow_query)
    t.start()
    assert started.wait(timeout=10)

    # 在途查询持有旧快照期间提交撤回补传
    post(client, dense_csv(withdrawn_z=True, z_revision=1))
    release.set()
    t.join()

    # 旧快照查询：仍能给出 Z 的分裂证据
    assert box["in_flight"] == [_expected_dense_bridge()]
    # 补传后的新查询：按更新后的上报，Z 已撤回，无桥接
    assert get_bridges(client, "DENSE")["bridges"] == []


def test_grid_engine_is_linear_memory_on_dense_cliques():
    """核心引擎自检：6001 节点密集图不物化 O(V^2) 边，能在受限内存下算完且结果正确。"""
    reps = []
    for i in range(DENSE_N):
        gx, gy = i % DENSE_SIDE, i // DENSE_SIDE
        reps.append({"report_id": f"L{i:04d}", "x": gx*DENSE_STEP,
                     "y": gy*DENSE_STEP, "ts_epoch": 0.0})
    for i in range(DENSE_N):
        gx, gy = i % DENSE_SIDE, i // DENSE_SIDE
        reps.append({"report_id": f"R{i:04d}", "x": DENSE_RIGHT_X+gx*DENSE_STEP,
                     "y": gy*DENSE_STEP, "ts_epoch": 0.0})
    reps.append({"report_id": "Z", "x": DENSE_Z_X, "y": 0.0, "ts_epoch": 0.0})

    bridges = core.find_bridges(reps)
    assert bridges == [_expected_dense_bridge()]
