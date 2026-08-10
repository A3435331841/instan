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


def normalize(v):
    """单位化向量（长度不足时保护，返回与输入同形）。

    参数: v 形状 (..., 3) 数组。
    返回: (..., 3) 单位向量。
    """
    v = np.asarray(v, dtype=np.float64)
    n = np.maximum(np.linalg.norm(v, axis=-1, keepdims=True), 1e-12)
    return v / n


def rodrigues_rotate(v, axis, angle):
    """Rodrigues 旋转：向量 v 绕单位轴 axis 旋转 angle 弧度（S² 状态用）。

    参数: v 形状 (..., 3) 向量；axis 形状 (..., 3) 旋转轴（自动单位化）；
          angle 标量或 (...,) 弧度（可广播）。
    返回: (..., 3) 旋转后的向量（不主动归一，调用方自行决定）。
    公式: v' = v*cosθ + (k×v)*sinθ + k*(k·v)*(1-cosθ)，k = normalize(axis)。
    """
    v = np.asarray(v, dtype=np.float64)
    k = normalize(np.asarray(axis, dtype=np.float64))
    a = np.asarray(angle, dtype=np.float64)
    cos, sin = np.cos(a), np.sin(a)
    kdot = np.sum(k * v, axis=-1, keepdims=True)
    cross = np.cross(k, v)
    return v * cos + cross * sin + k * kdot * (1.0 - cos)


def rotate_with_rotvec(v, rotvec):
    """按旋转向量 rotvec(=axis*angle) 旋转向量 v（S² 状态用）。

    参数: v 形状 (3,) 向量；rotvec 形状 (3,) 旋转向量，模为该旋转角度（弧度）。
    返回: (3,) 旋转后的向量。
    说明: 等价于 rodrigues_rotate(v, normalize(rotvec), |rotvec|)。
    """
    m = float(np.linalg.norm(np.asarray(rotvec, dtype=np.float64)))
    if m < 1e-12:
        return np.array(v, dtype=np.float64)
    return rodrigues_rotate(v, rotvec, m)


def rotation_vec_between(p, q):
    """从单位向量 p 到 q 的最短旋转向量（axis*angle，S² 状态观测速度用）。

    参数: p, q 形状 (3,) 单位向量。
    返回: (3,) 旋转向量 r = angle * axis，其中 axis = normalize(cross(p,q))，
          angle = atan2(|p×q|, p·q) ∈ [0, π]。
    """
    p = normalize(np.asarray(p, dtype=np.float64))
    q = normalize(np.asarray(q, dtype=np.float64))
    c = np.cross(p, q)
    s = float(np.linalg.norm(c))
    d = float(np.dot(p, q))
    angle = float(np.arctan2(s, d))
    if s < 1e-12:  # 对跖点或同向：旋转轴退化，取与 p 正交的固定轴（防奇异）
        axis = np.array([0.0, 1.0, 0.0])
        if abs(float(np.dot(p, axis))) > 0.9:
            axis = np.array([1.0, 0.0, 0.0])
        axis = normalize(axis - float(np.dot(axis, p)) * p)
    else:
        axis = c / s
    return axis * angle


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
