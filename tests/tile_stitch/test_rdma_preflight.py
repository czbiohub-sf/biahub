"""RDMA fabric preflight contracts for Monarch tile-stitch startup."""

from __future__ import annotations

from pathlib import Path

import pytest

from biahub.tile_stitch.monarch.backend import (
    _configure_rdma_transport,
    _inspect_rdma_fabric,
    _is_monarch_global_gid,
)


def _write_port(
    root: Path,
    device_name: str,
    *,
    vendor: str = "0x15b3",
    state: str = "4: ACTIVE",
    gids: tuple[tuple[str, str], ...] = (),
) -> None:
    device = root / device_name
    (device / "device").mkdir(parents=True)
    (device / "device" / "vendor").write_text(vendor)
    port = device / "ports" / "1"
    (port / "gids").mkdir(parents=True)
    (port / "gid_attrs" / "types").mkdir(parents=True)
    (port / "state").write_text(state)
    for index, (gid, gid_type) in enumerate(gids):
        (port / "gids" / str(index)).write_text(gid)
        (port / "gid_attrs" / "types" / str(index)).write_text(gid_type)


@pytest.mark.parametrize(
    ("gid", "expected"),
    [
        ("::", False),
        ("::1", False),
        ("fe80::1", False),
        ("fec0::1", False),
        ("::ffff:169.254.1.1", False),
        ("::ffff:10.30.15.68", True),
        ("fd00::1", True),
        ("2001:db8::1", True),
    ],
)
def test_gid_scope_matches_monarch(gid: str, expected: bool):
    assert _is_monarch_global_gid(gid) is expected


def test_preflight_rejects_active_mellanox_port_without_global_rocev2(tmp_path):
    _write_port(
        tmp_path,
        "ibp0",
        gids=(("fe80::1", "IB/RoCE v1"), ("fe80::2", "RoCE v2")),
    )

    result = _inspect_rdma_fabric(tmp_path)

    assert result.sysfs_readable
    assert result.active_mellanox_ports == ("ibp0/port1",)
    assert result.unusable_mellanox_ports == ("ibp0/port1",)


def test_preflight_accepts_monarch_global_rocev2_and_ignores_other_devices(tmp_path):
    _write_port(tmp_path, "ibp0", gids=(("fd00::1", "RoCE v2"),))
    _write_port(
        tmp_path,
        "ibp1",
        state="1: DOWN",
        gids=(("fe80::1", "IB/RoCE v1"),),
    )
    _write_port(
        tmp_path,
        "efa0",
        vendor="0x1d0f",
        gids=(("fe80::1", "IB/RoCE v1"),),
    )

    result = _inspect_rdma_fabric(tmp_path)

    assert result.active_mellanox_ports == ("ibp0/port1",)
    assert result.unusable_mellanox_ports == ()


def test_transport_forces_tcp_before_actor_startup(tmp_path, monkeypatch):
    import monarch
    import monarch.rdma

    _write_port(tmp_path, "ibp0", gids=(("fe80::1", "IB/RoCE v1"),))
    configured = {}
    monkeypatch.setattr(monarch, "get_global_config", lambda: {"rdma_disable_ibverbs": False})
    monkeypatch.setattr(monarch, "configure", lambda **kwargs: configured.update(kwargs))
    monkeypatch.setattr(monarch.rdma, "is_ibverbs_available", lambda: True)

    assert _configure_rdma_transport(tmp_path) == "tcp"
    assert configured == {
        "rdma_allow_tcp_fallback": True,
        "rdma_disable_ibverbs": True,
    }
