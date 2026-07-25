# -*- coding: utf-8 -*-
"""球面基础工具：经度归一化、经纬度与单位向量互转、切平面坐标系（模块内部共享）。"""
import numpy as np


def wrap_lon(lon):
    """经度归一化到 (-180, 180]。

    参数: lon 标量或 np.ndarray，单位度。
    返回: 归一化后的经度，标量输入返回 float，数组输入返回 np.ndarray。
    """
    out = 180.0 - np.mod(180.0 - np.asarray(lon, dtype=np.float64), 360.0)
    return float(out) if np.isscalar(lon) else out


def delta_lon(d):
    """经度差归一化到 (-180, 180]（环绕差分，取最短路径）。

    参数: d 标量或 np.ndarray，单位度。
    返回: 归一化后的经度差。
    """
    out = 180.0 - np.mod(180.0 - np.asarray(d, dtype=np.float64), 360.0)
    return float(out) if np.isscalar(d) else out


def lonlat_to_unit(lon, lat):
    """球面经纬度转单位向量（y 轴指向北极；lon=0,lat=0 对应 +x，lon 增大朝 +z）。

    参数: lon, lat 标量或数组（可广播），单位度。
    返回: (x, y, z)，与输入同形的数组或 float。
    """
    lon_r = np.deg2rad(np.asarray(lon, dtype=np.float64))
    lat_r = np.deg2rad(np.asarray(lat, dtype=np.float64))
    c = np.cos(lat_r)
    x = c * np.cos(lon_r)
    y = np.sin(lat_r)
    z = c * np.sin(lon_r)
    if x.ndim == 0:
        return float(x), float(y), float(z)
    return x, y, z


def unit_to_lonlat(x, y, z):
    """单位向量转经纬度（经度自动 wrap 到 (-180, 180]）。

    参数: x, y, z 标量或数组（可广播），无须严格归一。
    返回: (lon, lat)，单位度，标量输入返回 float。
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    z = np.asarray(z, dtype=np.float64)
    n = np.maximum(np.sqrt(x * x + y * y + z * z), 1e-12)
    lat = np.rad2deg(np.arcsin(np.clip(y / n, -1.0, 1.0)))
    lon = 180.0 - np.mod(180.0 - np.rad2deg(np.arctan2(z, x)), 360.0)
    if lon.ndim == 0:
        return float(lon), float(lat)
    return lon, lat


def _tangent_frame(lon, lat):
    """中心方向的切平面右手坐标系（私有）。

    参数: lon, lat 中心经纬度（度）。
    返回: (center, east, north) 三个 (3,) 单位向量；east 朝经度增大，north 朝纬度增大。
    """
    cx, cy, cz = lonlat_to_unit(lon, lat)
    c = np.array([cx, cy, cz], dtype=np.float64)
    up = np.array([0.0, 1.0, 0.0])
    east = np.cross(c, up)
    n = np.linalg.norm(east)
    if n < 1e-9:  # 中心恰在极点，任取固定东西方向
        east = np.array([0.0, 0.0, 1.0 if cy > 0 else -1.0])
    else:
        east = east / n
    north = np.cross(east, c)
    return c, east, north


def _offset_dirs(du, dv, frame, gnomonic=True):
    """由切平面角偏移生成单位方向向量（私有）。

    参数: du, dv 弧度（可广播数组，east/north 方向带符号角偏移）；
          frame 为 _tangent_frame 输出；
          gnomonic=True 用切平面（gnomonic）投影，False 用 eBFoV 球面均匀角采样。
    返回: (vx, vy, vz)，与 du 广播后同形。
    """
    c, east, north = frame
    du = np.asarray(du, dtype=np.float64)
    dv = np.asarray(dv, dtype=np.float64)
    if gnomonic:
        kx = np.tan(du)
        ky = np.tan(dv)
        vx = c[0] + kx * east[0] + ky * north[0]
        vy = c[1] + kx * east[1] + ky * north[1]
        vz = c[2] + kx * east[2] + ky * north[2]
    else:
        cd, sd = np.cos(du), np.sin(du)
        cv, sv = np.cos(dv), np.sin(dv)
        ax = c[0] * cd + east[0] * sd  # 水平圆（过 center 与 east）
        ay = c[1] * cd + east[1] * sd
        az = c[2] * cd + east[2] * sd
        vx = cv * ax + sv * north[0]
        vy = cv * ay + sv * north[1]
        vz = cv * az + sv * north[2]
    n = np.maximum(np.sqrt(vx * vx + vy * vy + vz * vz), 1e-12)
    return vx / n, vy / n, vz / n
