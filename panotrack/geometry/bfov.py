# -*- coding: utf-8 -*-
"""BFoV（球面视场窗口）定义及其与 ERP 边界框的互转。"""
from dataclasses import dataclass

import numpy as np

from .sphere import wrap_lon, lonlat_to_unit, unit_to_lonlat, _tangent_frame, _offset_dirs


@dataclass
class BFoV:
    """球面视场窗口：以 (lon,lat) 为中心、水平/垂直视场角为 fov_h/fov_v（均为度）。"""
    lon: float
    lat: float
    fov_h: float
    fov_v: float
    rotation: float = 0.0


def _px_to_lonlat(u, v, erp_w, erp_h):
    """连续像素坐标转经纬度（私有）。u∈[0,W] 可超界（跨界），v∈[0,H]。"""
    lon = np.asarray(u, dtype=np.float64) / erp_w * 360.0 - 180.0
    lat = 90.0 - np.asarray(v, dtype=np.float64) / erp_h * 180.0
    return lon, lat


def _bbox_boundary_points(x, y, w, h, n):
    """ERP 框边界采样点（私有），每边 n 点含角点，返回连续像素坐标。"""
    t = np.linspace(0.0, 1.0, n)
    xs = np.concatenate([x + t * w, x + t * w, np.full(n, x), np.full(n, x + w)])
    ys = np.concatenate([np.full(n, y), np.full(n, y + h), y + t * h, y + t * h])
    return xs, ys


def bfov_from_erp_bbox(x, y, w, h, erp_w, erp_h):
    """由 ERP 框估算 BFoV：采样框边界点转球面，取中心与角跨度；跨界框先对 x 做模 W 展开。

    参数: x, y, w, h ERP 框（像素，浮点；跨界约定 x+w 可超 erp_w）；
          erp_w, erp_h 全景图尺寸（erp_w = 2*erp_h）。
    返回: BFoV。中心取框中心像素对应经纬度（保证正逆往返一致），
          fov_h/fov_v 为边界采样点在中心切平面上的带符号角跨度。
    """
    # 中心：框中心像素（跨界先模 W），避免极点附近均值方向引入系统偏差
    cu = (x + w / 2.0) % erp_w
    cv = y + h / 2.0
    lon_c, lat_c = _px_to_lonlat(cu, cv, erp_w, erp_h)
    lon_c = wrap_lon(float(lon_c))
    lat_c = float(np.clip(lat_c, -90.0, 90.0))

    # 边界采样（每边 16 点）转球面，在中心切平面上量测带符号角偏移
    xs, ys = _bbox_boundary_points(x, y, w, h, 16)
    lons, lats = _px_to_lonlat(np.mod(xs, erp_w), np.clip(ys, 0.0, erp_h), erp_w, erp_h)
    vx, vy, vz = lonlat_to_unit(lons, lats)
    c, east, north = _tangent_frame(lon_c, lat_c)
    pc = vx * c[0] + vy * c[1] + vz * c[2]
    pe = vx * east[0] + vy * east[1] + vz * east[2]
    pn = vx * north[0] + vy * north[1] + vz * north[2]
    du = np.rad2deg(np.arctan2(pe, pc))
    dv = np.rad2deg(np.arctan2(pn, pc))
    fov_h = max(float(du.max() - du.min()), 1e-3)
    fov_v = max(float(dv.max() - dv.min()), 1e-3)
    return BFoV(lon=lon_c, lat=lat_c, fov_h=fov_h, fov_v=fov_v)


def erp_bbox_from_bfov(bfov, erp_w, erp_h, samples=48):
    """BFoV 边界采样投影回 ERP，取最小面积轴对齐框；跨界时 x+w 可超 W。

    参数: bfov BFoV；erp_w, erp_h 全景图尺寸；samples 每边采样点数。
    返回: (x, y, w, h) 浮点 ERP 框，x∈[0,W)。
    """
    frame = _tangent_frame(bfov.lon, bfov.lat)
    gnomonic = max(bfov.fov_h, bfov.fov_v) <= 90.0
    t = np.linspace(-0.5, 0.5, samples)
    dh = np.deg2rad(bfov.fov_h)
    dvv = np.deg2rad(bfov.fov_v)
    du = np.concatenate([t * dh, t * dh, np.full(samples, -0.5 * dh), np.full(samples, 0.5 * dh)])
    dv = np.concatenate([np.full(samples, -0.5 * dvv), np.full(samples, 0.5 * dvv), t * dvv, t * dvv])
    vx, vy, vz = _offset_dirs(du, dv, frame, gnomonic)
    lon, lat = unit_to_lonlat(vx, vy, vz)
    px = (np.asarray(lon) + 180.0) / 360.0 * erp_w
    py = (90.0 - np.asarray(lat)) / 180.0 * erp_h
    # 跨界展开：相对中位数回绕，使采样点连续
    ref = float(np.median(px))
    px = px + erp_w * np.round((ref - px) / erp_w)
    x0, x1 = float(px.min()), float(px.max())
    y0 = float(np.clip(py.min(), 0.0, erp_h))
    y1 = float(np.clip(py.max(), 0.0, erp_h))
    return x0 % erp_w, y0, x1 - x0, y1 - y0
