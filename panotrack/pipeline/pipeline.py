# -*- coding: utf-8 -*-
"""PanoTracker：360° ERP 全景视频实时单目标跟踪主类（模块 E）。

内部流程（契约规定）：
  球面状态预测 -> geometry tangent 切图(RemapCache) -> 局部 tracker update
  -> local_bbox_to_erp 逆投影 -> score/psr 判丢 -> 逐级扩大 FoV 重试
  -> 连续丢失则全局重检测，找回后重建局部 tracker 模板并置 status='recovered'。

切图窗口以球面角偏移等间隔采样（gnomonic / eBFoV 一致），因此
ERP 框 <-> 局部框、局部框跨窗口迁移都可在 (du, dv) 角偏移域精确互转。
"""
import numpy as np
from scipy.ndimage import gaussian_filter, uniform_filter1d

from panotrack.geometry.bfov import BFoV, bfov_from_erp_bbox, erp_bbox_from_bfov
from panotrack.geometry.projection import RemapCache, remap_image
from panotrack.geometry.sphere import (
    wrap_lon, delta_lon, lonlat_to_unit, unit_to_lonlat, _tangent_frame, _offset_dirs,
)
from panotrack.geometry.motion_prior import SoftS2MotionPrior
from panotrack.trackers.factory import create_tracker, get_tracker_input_space

from .state import SphericalState
from .redetect import GlobalRedetector
from .redetect_v2 import GlobalRedetectorV2

# PanoTracker(BFoV 切图框架)内置缺省。全帧跟踪器(lightfc_onnx 等)在
# BFoV 下语义不匹配,应经 eval_360vot/CLI 用 configs/default.json 跑全帧路径。
DEFAULT_CONFIG = {
    'tracker': 'ncc',
    'patch_size': 255,
    'sr_ratio': 3.0,
    'sr_min_fov': 20.0,
    'lost_score': 0.45,
    'lost_psr': 7.0,
    'lost_apce': 0.0,
    'redetect_interval': 1,
    'max_lost_frames': 1000000,
    'template_size': 127,
    'search_scale': 2.0,
    'lr': 0.02,
    'hp_sigma': 3.0,
    'refine': True,
    'motion_prior': False,   # GRT-360 Soft S² motion prior（feature-flag）
    'mp_lambda': 1.0,
    'mp_sigma_base': 15.0,
    'mp_sigma_per_speed': 0.5,
}

_QUANT = 2.0          # 与 RemapCache 量化键一致的角度栅格（度）
_FOV_MAX = 160.0      # 切图 FoV 上限
_FOV_EXPAND = (1.0, 1.5, 2.25, 3.0, 4.0)  # 判丢后逐级扩大 FoV 的倍率阶梯
_LAT_LIM = 88.0       # 切图中心纬度上限（留窗口余量）


def _highpass(img, sigma):
    """逐通道高通（原图减高斯模糊），抑制平滑背景斜坡、保留纹理与边缘。

    参数: img (H,W,3) 数组；sigma 高斯核尺度（像素）。
    返回: float32 高通图（零均值附近；NCC 对亮度平移不变，可直接喂 tracker）。
    """
    g = np.asarray(img, dtype=np.float32)
    return g - gaussian_filter(g, sigma=(sigma, sigma, 0.0))


def _segment_extent(prof, center, thr_ratio=0.5):
    """一维能量轮廓上取含参考位置的显著连通段（私有）。

    参数: prof 非负能量轮廓；center 参考下标（跟踪框心在该轴上的位置）；
          thr_ratio 阈值比例（阈值 = 中位背景 + ratio × (峰值 − 中位背景)，
          对平顶纹理能量等价于取 FWHM，尺寸估计稳健）。
    返回: (lo, hi, clean)；clean 表示轮廓是否洁净双峰（中位背景 ≤ 0.35×峰值，
          合成/简单背景为 True，真实杂乱背景多为 False，调用方应降级信任）；
          轮廓过平或无显著段时返回 None。
    """
    n = int(prof.size)
    if n < 6:
        return None
    k = max(3, (n // 14) | 1)                    # 奇数平滑窗，约轮廓长的 1/14
    sm = uniform_filter1d(prof.astype(np.float64), size=k, mode='nearest')
    peak = float(sm.max())
    base = float(np.median(sm))
    if peak <= base * 1.6 + 1e-6:                # 对比不足：无明显目标峰
        return None
    clean = base <= 0.35 * peak
    t = base + thr_ratio * (peak - base)
    c = int(np.clip(round(center), 0, n - 1))
    if sm[c] < t:                                # 框心不在峰上：找最近的峰内 bin
        above = np.flatnonzero(sm >= t)
        if above.size == 0:
            return None
        c = int(above[np.argmin(np.abs(above - c))])
    lo = c
    while lo > 0 and sm[lo - 1] >= t:
        lo -= 1
    hi = c + 1
    while hi < n and sm[hi] >= t:
        hi += 1
    if hi - lo < 3 or hi - lo > 0.9 * n:         # 退化段（过窄/占满搜索窗）
        return None
    return lo, hi, clean


def _refine_box_local(hp, box, thr_ratio=0.5):
    """用高通能量轮廓精修局部跟踪框的位置与尺寸（私有）。

    参数: hp (H,W,3) float 高通切图；box (x,y,w,h) NCC 输出的局部框；
          thr_ratio 连通段阈值比例。
    返回: (x,y,w,h) 精修框；某轴证据不足时保留该轴原值。
    说明: NCC 模板含背景上下文，对逐帧形变（极区拉伸/压扁）的目标会出现
          尺度与位置锚定；目标纹理在高通域的能量显著强于平滑背景，
          在框心 ±1.3 倍框范围的邻域内取 |hp| 行/列轮廓的显著连通段，
          作为目标的真实范围。仅在接受帧调用。
          信任分级：轮廓洁净（clean）时允许激进修正（如极区 3 倍收缩）；
          轮廓杂乱（真实场景常态）时仅接受与跟踪框相容的保守修正
          （段长 ≥ 0.5×框长且段心偏移 ≤ 0.5×框长），防止把框吸到
          背景强纹理上缩成细条（真实数据上曾因此连锁污染模板）。
    """
    H, W = hp.shape[:2]
    mag = np.abs(hp).sum(axis=2)
    x, y, w, h = (float(v) for v in box)
    cx, cy = x + w / 2.0, y + h / 2.0
    x0 = max(0, int(np.floor(cx - 1.3 * w)))
    x1 = min(W, int(np.ceil(cx + 1.3 * w)))
    y0 = max(0, int(np.floor(cy - 1.3 * h)))
    y1 = min(H, int(np.ceil(cy + 1.3 * h)))
    if x1 - x0 < 6 or y1 - y0 < 6:
        return box
    win = mag[y0:y1, x0:x1]
    rx = _segment_extent(win.sum(axis=0), cx - x0, thr_ratio)
    ry = _segment_extent(win.sum(axis=1), cy - y0, thr_ratio)
    if rx is not None:
        lo, hi, clean = rx
        if clean or (hi - lo >= 0.5 * w and abs((lo + hi) / 2.0 - (cx - x0)) <= 0.5 * w):
            x, w = x0 + float(lo), float(hi - lo)
    if ry is not None:
        lo, hi, clean = ry
        if clean or (hi - lo >= 0.5 * h and abs((lo + hi) / 2.0 - (cy - y0)) <= 0.5 * h):
            y, h = y0 + float(lo), float(hi - lo)
    return (x, y, w, h)


# ---------------------------------------------------------------- 角偏移域坐标换算

def _gnomonic(bfov):
    """该窗口是否走 gnomonic 切平面投影（与 tangent_remap 判定一致，私有）。"""
    return max(bfov.fov_h, bfov.fov_v) <= 90.0


def _dir_to_local(vx, vy, vz, bfov, out_w, out_h):
    """单位方向向量 -> 切图连续像素坐标（角点约定，像素 (i,j) 覆盖 [i,i+1)）。

    参数: vx, vy, vz 单位方向分量（可广播数组）；bfov 切图窗口；
          out_w, out_h 切图尺寸。
    返回: (xs, ys) 局部连续坐标；gnomonic 与 eBFoV 两种采样模式分别求逆。
    """
    c, e, n = _tangent_frame(bfov.lon, bfov.lat)
    pc = vx * c[0] + vy * c[1] + vz * c[2]
    pe = vx * e[0] + vy * e[1] + vz * e[2]
    pn = vx * n[0] + vy * n[1] + vz * n[2]
    du = np.rad2deg(np.arctan2(pe, pc))
    if _gnomonic(bfov):
        dv = np.rad2deg(np.arctan2(pn, pc))
    else:  # eBFoV：v = cos(dv)*a + sin(dv)*north => dv = asin(pn)
        dv = np.rad2deg(np.arcsin(np.clip(pn, -1.0, 1.0)))
    xs = (du / bfov.fov_h + 0.5) * out_w
    ys = (0.5 - dv / bfov.fov_v) * out_h
    return xs, ys


def _local_to_dir(xs, ys, bfov, out_w, out_h):
    """切图连续像素坐标 -> 单位方向向量（_dir_to_local 的逆，私有）。"""
    du = np.deg2rad((np.asarray(xs, dtype=np.float64) / out_w - 0.5) * bfov.fov_h)
    dv = np.deg2rad((0.5 - np.asarray(ys, dtype=np.float64) / out_h) * bfov.fov_v)
    return _offset_dirs(du, dv, _tangent_frame(bfov.lon, bfov.lat), _gnomonic(bfov))


def _jacobian_local_per_erp(bfov, out_w, out_h, erp_w, erp_h, ex, ey):
    """ERP 中心点处 d(局部像素)/d(ERP像素) 的数值雅可比（对角近似）。

    参数: bfov 切图窗口；out_w, out_h 切图尺寸；erp_w, erp_h 全景尺寸；
          ex, ey ERP 连续像素坐标（采样点）。
    返回: (dlx, dly) —— 1 个 ERP 像素在 x/y 方向对应的局部像素数。
    """
    lon = np.mod(ex, erp_w) / erp_w * 360.0 - 180.0
    lat = 90.0 - np.clip(ey, 0.0, erp_h) / erp_h * 180.0
    lons = np.array([lon + 360.0 / erp_w, lon - 360.0 / erp_w, lon, lon])
    lats = np.array([lat, lat, lat - 180.0 / erp_h, lat + 180.0 / erp_h])
    vx, vy, vz = lonlat_to_unit(lons, lats)
    lx, ly = _dir_to_local(vx, vy, vz, bfov, out_w, out_h)
    return abs(float(lx[0] - lx[1])) / 2.0, abs(float(ly[2] - ly[3])) / 2.0


def _erp_bbox_to_local(bbox, bfov, out_w, out_h, erp_w, erp_h):
    """ERP 框 -> 指定切图窗口下的局部框（中心映射 + 中心雅可比一阶近似）。

    参数: bbox (x,y,w,h) ERP 框（跨界约定）；bfov 切图窗口；
          out_w, out_h 切图尺寸；erp_w, erp_h 全景尺寸。
    返回: (lx,ly,lw,lh) 局部连续坐标框。
    说明: 边界 min/max 采样会把梯形畸变的最大跨度计入框尺寸，极区显著膨胀
          （理想往返 SR 仅 ~0.56）；中心+雅可比一阶近似理想往返 IoU≈1.0。
    """
    x, y, w, h = (float(v) for v in bbox)
    cx, cy = x + w / 2.0, y + h / 2.0
    lon = np.mod(cx, erp_w) / erp_w * 360.0 - 180.0
    lat = 90.0 - np.clip(cy, 0.0, erp_h) / erp_h * 180.0
    vx, vy, vz = lonlat_to_unit(np.array([lon]), np.array([lat]))
    lx, ly = _dir_to_local(vx, vy, vz, bfov, out_w, out_h)
    dlx, dly = _jacobian_local_per_erp(bfov, out_w, out_h, erp_w, erp_h, cx, cy)
    lw, lh = w * dlx, h * dly
    return (float(lx[0] - lw / 2.0), float(ly[0] - lh / 2.0), lw, lh)


def _jacobian_erp_per_local(bfov, out_w, out_h, erp_w, erp_h, lx0, ly0):
    """局部点处 d(ERP像素)/d(局部像素) 的数值雅可比（对角近似）。

    返回: (dex, dey) —— 1 个局部像素在 x/y 方向对应的 ERP 像素数。
    """
    vx, vy, vz = _local_to_dir(np.array([lx0 + 1.0, lx0 - 1.0, lx0, lx0]),
                               np.array([ly0, ly0, ly0 + 1.0, ly0 - 1.0]),
                               bfov, out_w, out_h)
    lons, lats = unit_to_lonlat(vx, vy, vz)
    dex = abs(float(delta_lon(lons[0] - lons[1]))) / 2.0 / 360.0 * erp_w
    dey = abs(float(lats[2] - lats[3])) / 2.0 / 180.0 * erp_h
    return dex, dey


def _local_bbox_to_erp(lb, bfov, out_w, out_h, erp_w, erp_h):
    """切图局部框 -> ERP 框（中心映射 + 中心雅可比一阶近似，极区不膨胀）。

    参数: lb (x,y,w,h) 局部框；bfov 切图窗口；out_w, out_h 切图尺寸；
          erp_w, erp_h 全景尺寸。
    返回: (x,y,w,h) ERP 框，x∈[0,W)，跨界时 x+w 可超 W。
    """
    x, y, w, h = (float(v) for v in lb)
    cx, cy = x + w / 2.0, y + h / 2.0
    vx, vy, vz = _local_to_dir(np.array([cx]), np.array([cy]), bfov, out_w, out_h)
    lon, lat = unit_to_lonlat(vx[0], vy[0], vz[0])
    ex = (lon + 180.0) / 360.0 * erp_w
    ey = (90.0 - lat) / 180.0 * erp_h
    dex, dey = _jacobian_erp_per_local(bfov, out_w, out_h, erp_w, erp_h, cx, cy)
    w2, h2 = w * dex, h * dey
    return (float((ex - w2 / 2.0) % erp_w), float(ey - h2 / 2.0), w2, h2)


def _transform_local_bbox(bbox, bfov_from, bfov_to, out_w, out_h):
    """局部框在两个切图窗口间迁移：中心经球面方向互转，尺寸按 FoV 比例缩放
    （一阶近似；逐帧调用无膨胀累积）。

    参数: bbox (x,y,w,h) bfov_from 窗口下的局部框；bfov_to 目标窗口；
          out_w, out_h 切图尺寸（两窗口相同）。
    返回: bfov_to 窗口下的局部框。
    """
    x, y, w, h = (float(v) for v in bbox)
    cx, cy = x + w / 2.0, y + h / 2.0
    vx, vy, vz = _local_to_dir(np.array([cx]), np.array([cy]), bfov_from, out_w, out_h)
    nx, ny = _dir_to_local(vx, vy, vz, bfov_to, out_w, out_h)
    w2 = w * bfov_from.fov_h / bfov_to.fov_h
    h2 = h * bfov_from.fov_v / bfov_to.fov_v
    return (float(nx[0] - w2 / 2.0), float(ny[0] - h2 / 2.0), float(w2), float(h2))


def _get_tracker_bbox(tracker):
    """读取局部 tracker 的中心/尺度状态（模块 B 未提供公共迁移接口，集成层内聚于此）。"""
    return (float(tracker._cx) - float(tracker._w) / 2.0,
            float(tracker._cy) - float(tracker._h) / 2.0,
            float(tracker._w), float(tracker._h))


def _set_tracker_bbox(tracker, bbox):
    """把迁移后的局部框写回 tracker 内部状态（与 _get_tracker_bbox 配对）。"""
    x, y, w, h = (float(v) for v in bbox)
    tracker._cx = x + w / 2.0
    tracker._cy = y + h / 2.0
    tracker._w = max(w, 2.0)
    tracker._h = max(h, 2.0)


def _snap(v, lo=None, hi=None):
    """角度对齐到 RemapCache 量化栅格（2°），保证请求窗口与缓存 remap 严格一致。"""
    q = round(float(v) / _QUANT) * _QUANT
    if lo is not None:
        q = max(q, lo)
    if hi is not None:
        q = min(q, hi)
    return q


def _cap_box(box, ref_size, erp_w, erp_h, grow=2.0):
    """输出 ERP 框的安全钳制（私有）。

    参数: box (x,y,w,h) 待输出框；ref_size (w,h) 最后接受框尺寸；
          erp_w, erp_h 全景尺寸；grow 相对参考尺寸的逐帧最大放大倍率。
    返回: 钳制后的 (x,y,w,h)：w/h 不超过参考的 grow 倍且不超过图幅，
          x 归一化到 [0,W)（x+w 仍可跨界超 W），y 夹到 [0, H-h]。
    说明: 丢失外推时状态可能被污染到极区，小角窗经边界采样会退化成
          全图宽巨框（IoU 恒 0 且永不恢复）；接受路径同样加此安全栏。
    """
    x, y, w, h = (float(v) for v in box)
    rw, rh = (max(float(v), 2.0) for v in ref_size)
    w = float(np.clip(w, 2.0, min(grow * rw, float(erp_w))))
    h = float(np.clip(h, 2.0, min(grow * rh, float(erp_h))))
    return (x % erp_w, float(np.clip(y, 0.0, erp_h - h)), w, h)


class PanoTracker:
    """360° ERP 全景视频实时单目标跟踪主类（契约模块 E）。

    init(frame, bbox) 用首帧 ERP 与目标框初始化；
    update(frame) 逐帧输出 {'bbox', 'score', 'status', 'fov'}。
    """

    def __init__(self, config=None):
        """创建 PanoTracker。

        参数: config 配置 dict，缺省值同 configs/default.json（None 全默认）。
        返回: None
        """
        cfg = dict(DEFAULT_CONFIG)
        if config:
            cfg.update(config)
        self._cfg = cfg
        self._cache = RemapCache()
        self._tracker = None
        self._state = None
        self._patch_bfov = None   # 当前切图窗口（tracker 局部坐标所在窗口）
        self._template = None     # 全局重检测模板 (img, (w_erp, h_erp))
        # GRT-360 Soft S² motion prior（feature-flag 'motion_prior'）
        self._motion_prior = None
        if bool(cfg.get('motion_prior', False)):
            self._motion_prior = SoftS2MotionPrior(
                lambda_=float(cfg.get('mp_lambda', 1.0)),
                sigma_base=float(cfg.get('mp_sigma_base', 15.0)),
                sigma_per_speed=float(cfg.get('mp_sigma_per_speed', 0.5)))
        redetector_type = cfg.get('redetector', 'v1')
        if redetector_type == 'v2':
            self._redetector = GlobalRedetectorV2(
                self._get_template, min_score=max(cfg['lost_score'] + 0.15, 0.5))
        else:
            self._redetector = GlobalRedetector(
                self._get_template, min_score=max(cfg['lost_score'] + 0.15, 0.6))
        self._lost_count = 0
        self._erp_w = 0
        self._erp_h = 0
        self._last_fov = (cfg['sr_min_fov'], cfg['sr_min_fov'])
        self._last_good_size = (1.0, 1.0)   # 最后接受框 ERP 尺寸（输出钳制参考）

    # ------------------------------------------------------------ 内部工具

    def _get_template(self):
        """供 GlobalRedetector 取模板（私有回调）。"""
        return self._template

    def _make_cut(self, target_bfov, fov_scale=1.0):
        """由目标 BFoV 生成切图窗口：FoV 随目标角尺度自适应（sr_ratio），
        中心/FoV 对齐 2° 栅格，保证与 RemapCache 缓存 remap 严格一致。"""
        cfg = self._cfg
        ps = getattr(self, '_adaptive_patch_size', int(cfg['patch_size']))
        fh = float(np.clip(target_bfov.fov_h * cfg['sr_ratio'] * fov_scale,
                           cfg['sr_min_fov'], _FOV_MAX))
        fv = float(np.clip(target_bfov.fov_v * cfg['sr_ratio'] * fov_scale,
                           cfg['sr_min_fov'], _FOV_MAX))
        lon = wrap_lon(_snap(wrap_lon(target_bfov.lon)))
        lat = _snap(np.clip(target_bfov.lat, -_LAT_LIM, _LAT_LIM))
        return BFoV(lon=lon, lat=lat,
                    fov_h=_snap(fh, lo=4.0), fov_v=_snap(fv, lo=4.0))

    def _cut(self, frame, bfov):
        """按窗口切 tangent 图（经 RemapCache），返回 (patch, map_x, map_y)。"""
        ps = getattr(self, '_adaptive_patch_size', int(self._cfg['patch_size']))
        mx, my = self._cache.get_remap(bfov, ps, ps, self._erp_w, self._erp_h)
        return remap_image(frame, mx, my), mx, my

    def _migrate_tracker(self, new_bfov):
        """把局部 tracker 状态从旧切图窗口迁移到新窗口（角偏移域精确互转）。"""
        if self._patch_bfov is None:
            return
        ps = getattr(self, '_adaptive_patch_size', int(self._cfg['patch_size']))
        lb = _get_tracker_bbox(self._tracker)
        lb2 = _transform_local_bbox(lb, self._patch_bfov, new_bfov, ps, ps)
        _set_tracker_bbox(self._tracker, lb2)
        self._patch_bfov = new_bfov


    def _crop_template(self, frame, erp_bbox):
        """按 ERP 框裁剪全局重检测模板（水平回绕），退化时保留旧模板。"""
        x, y, w, h = (float(v) for v in erp_bbox)
        W, H = self._erp_w, self._erp_h
        iw, ih = int(round(w)), int(round(h))
        ix, iy = int(round(x)), int(round(y))
        if iw < 4 or ih < 4:
            return self._template
        iy = int(np.clip(iy, 0, max(0, H - ih)))
        cols = np.mod(ix + np.arange(iw), W)
        crop = np.ascontiguousarray(frame[iy:iy + ih][:, cols])
        return crop, (float(iw), float(ih))

    def _confident(self, res):
        """按契约阈值判定局部结果可信度（score/psr/apce 三重门限 + 相对阈值）。"""
        cfg = self._cfg
        score = res['score']
        # 绝对阈值
        abs_ok = (score >= cfg['lost_score']
                  and res['psr'] >= cfg['lost_psr']
                  and res['apce'] >= cfg['lost_apce'])
        if not abs_ok:
            return False
        # 相对阈值：如果历史 score 存在，当前 score 不能比历史均值低太多
        if hasattr(self, '_score_history') and len(self._score_history) > 5:
            hist_avg = sum(self._score_history[-10:]) / len(self._score_history[-10:])
            if hist_avg > 0.3 and score < hist_avg * 0.6:
                return False
        return True

    def _measured_bfov(self, local_bbox, cut):
        """由局部框在角偏移域直接测量目标 BFoV（跳过 ERP 往返，避免极区 cos(lat) 放大）。

        参数: local_bbox 当前切图下的局部框；cut 当前切图窗口。
        返回: BFoV（中心 = 局部框中心对应的球面方向，fov = 局部框占切图比例 × 切图 fov）。
        """
        ps = getattr(self, '_adaptive_patch_size', int(self._cfg['patch_size']))
        lx, ly, lw, lh = (float(v) for v in local_bbox)
        vx, vy, vz = _local_to_dir(np.array([lx + lw / 2.0]), np.array([ly + lh / 2.0]),
                                   cut, ps, ps)
        lon, lat = unit_to_lonlat(vx[0], vy[0], vz[0])
        return BFoV(lon=lon, lat=float(np.clip(lat, -89.9, 89.9)),
                    fov_h=max(lw / ps * cut.fov_h, 1e-3),
                    fov_v=max(lh / ps * cut.fov_v, 1e-3))

    def _local_bbox_dir(self, local_bbox, cut):
        """局部框中心对应的单位球方向 (3,)（供 motion prior 一致性判定）。"""
        ps = getattr(self, '_adaptive_patch_size', int(self._cfg['patch_size']))
        lx, ly, lw, lh = (float(v) for v in local_bbox)
        vx, vy, vz = _local_to_dir(np.array([lx + lw / 2.0]), np.array([ly + lh / 2.0]),
                                   cut, ps, ps)
        return np.array([vx[0], vy[0], vz[0]], dtype=np.float64)

    # ------------------------------------------------------------ 契约接口

    def init(self, frame, bbox):
        """用首帧 ERP 与目标框初始化。

        参数: frame (H,W,3) uint8 ERP 帧；bbox (x,y,w,h) 跨界约定 ERP 框。
        返回: None
        """
        frame = np.asarray(frame)
        assert frame.ndim == 3 and frame.shape[2] == 3 and frame.dtype == np.uint8, \
            'frame 须为 (H,W,3) uint8'
        H, W = frame.shape[:2]
        self._erp_w, self._erp_h = W, H
        cfg = self._cfg
        base_ps = int(cfg['patch_size'])
        # 自适应 patch_size：根据目标角尺度动态调整
        x, y, w, h = (float(v) for v in bbox)
        obj_pixels = max(w, h)
        obj_deg = max(obj_pixels / W * 360.0, obj_pixels / H * 180.0)
        if obj_deg < 3.0:
            ps = 400
        elif obj_deg < 5.0:
            ps = 321
        else:
            ps = base_ps
        self._adaptive_patch_size = ps
        tracker_name = cfg['tracker']
        # GRT-360 Commit 2 input-space 守卫：PanoTracker 只接受局部切图跟踪器，
        # 'erp_full' 跟踪器（LightFC/DirectERP）必须走 eval_360vot/CLI 全帧路径
        if get_tracker_input_space(tracker_name) != 'local_patch':
            raise ValueError(
                f"PanoTracker(BFoV 切图框架)不支持 input_space='erp_full' 的跟踪器 "
                f"{tracker_name!r}；请用 eval_360vot/CLI 全帧 runner 或改用 "
                f"'ncc'/'ncc_v2' 等局部切图跟踪器")
        self._tracker = create_tracker(tracker_name, **{k: v for k, v in cfg.items()
            if k not in ('tracker', 'patch_size', 'sr_ratio', 'sr_min_fov',
                         'lost_score', 'lost_psr', 'lost_apce',
                         'redetect_interval', 'max_lost_frames',
                         'hp_sigma', 'refine',
                         'motion_prior', 'mp_lambda', 'mp_sigma_base',
                         'mp_sigma_per_speed')})

        tb = bfov_from_erp_bbox(*bbox, W, H)
        self._state = SphericalState(
            tb,
            pos_alpha=float(cfg.get('pos_alpha', 0.8)),
            vel_alpha=float(cfg.get('vel_alpha', 0.4)),
            fov_alpha=float(cfg.get('fov_alpha', 0.15)),
            damping=float(cfg.get('state_damping', 0.9))
        )
        cut = self._make_cut(tb)
        patch, _, _ = self._cut(frame, cut)
        lb = _erp_bbox_to_local(bbox, cut, ps, ps, W, H)
        self._tracker.init(_highpass(patch, cfg['hp_sigma']), lb)
        self._patch_bfov = cut
        self._template = self._crop_template(frame, bbox)
        self._lost_count = 0
        self._last_fov = (cut.fov_h, cut.fov_v)
        self._last_good_size = (float(bbox[2]), float(bbox[3]))
        self._score_history = []  # 最近帧的 score 历史（用于自适应阈值）

    def update(self, frame):
        """跟踪新一帧。

        参数: frame (H,W,3) uint8 ERP 帧。
        返回: dict {'bbox': (x,y,w,h) 跨界约定, 'score': float,
                    'status': 'ok'|'lost'|'recovered', 'fov': (fov_h, fov_v)}
        """
        frame = np.asarray(frame)
        H, W = frame.shape[:2]
        assert W == self._erp_w and H == self._erp_h, '帧尺寸与 init 不一致'
        cfg = self._cfg
        ps = getattr(self, '_adaptive_patch_size', int(cfg['patch_size']))
        pred = self._state.predict()
        # Soft S² motion prior：预测单位球方向（供分数调制）
        if self._motion_prior is not None:
            px, py, pz = lonlat_to_unit(float(pred.lon), float(pred.lat))
            pred_vec = np.array([px, py, pz], dtype=np.float64)

        # 局部跟踪 + 逐级扩大 FoV 重试
        accepted = None
        last_res, last_cut = None, self._patch_bfov
        for scale in _FOV_EXPAND:
            cut = self._make_cut(pred, fov_scale=scale)
            patch, _, _ = self._cut(frame, cut)
            self._migrate_tracker(cut)
            hp = _highpass(patch, cfg['hp_sigma'])
            res = self._tracker.update(hp)
            # GRT-360：开启时用 S² 运动先验软调制得分（feature-flag）
            if self._motion_prior is not None:
                obs_vec = self._local_bbox_dir(res['bbox'], cut)
                score, _ = self._motion_prior(float(res['score']), pred_vec,
                                              obs_vec, self._state.angular_speed_deg)
                res = dict(res)          # 浅拷贝，避免污染 tracker 内部返回
                res['score'] = score
            last_res, last_cut = res, cut
            if self._confident(res):
                accepted = (res, cut, hp)
                break

        if accepted is not None:
            res, cut, hp = accepted
            if cfg.get('refine', True):
                rb = _refine_box_local(hp, res['bbox'])
                _set_tracker_bbox(self._tracker, rb)
            else:
                rb = res['bbox']
            erp_bbox = _cap_box(_local_bbox_to_erp(rb, cut, ps, ps, W, H),
                                self._last_good_size, W, H)
            self._state.update(self._measured_bfov(rb, cut))
            self._template = self._crop_template(frame, erp_bbox)
            status = 'recovered' if self._lost_count > 0 else 'ok'
            self._lost_count = 0
            self._last_fov = (cut.fov_h, cut.fov_v)
            self._last_good_size = (erp_bbox[2], erp_bbox[3])
            if hasattr(self, '_score_history'):
                self._score_history.append(float(res['score']))
                if len(self._score_history) > 30:
                    self._score_history.pop(0)
            return {'bbox': erp_bbox, 'score': float(res['score']),
                    'status': status, 'fov': (cut.fov_h, cut.fov_v)}

        # 连续丢失：按间隔做全局重检测（ds=2 细分采样兼顾可分性与速度）
        self._lost_count += 1
        interval = max(1, int(cfg['redetect_interval']))
        if self._lost_count <= int(cfg['max_lost_frames']) \
                and self._lost_count % interval == 0:
            found = self._redetector.search(frame, erp_downscale=2)
            if found is not None:
                rb, rscore = found
                rb = _cap_box(rb, self._last_good_size, W, H)
                tb = bfov_from_erp_bbox(*rb, W, H)
                # 重置状态（速度清零）；重检测框尺寸取模板尺寸有滞后，
                # FoV 沿用丢失前的状态估计，避免尺寸反馈被重播种
                old = self._state.bfov
                self._state = SphericalState(
                    BFoV(tb.lon, tb.lat, old.fov_h, old.fov_v),
                    fov_alpha=float(cfg.get('fov_alpha', 0.15)))
                cut = self._make_cut(tb)
                patch, _, _ = self._cut(frame, cut)
                lb = _erp_bbox_to_local(rb, cut, ps, ps, W, H)
                self._tracker.init(_highpass(patch, cfg['hp_sigma']), lb)  # 重建局部模板
                self._patch_bfov = cut
                self._template = self._crop_template(frame, rb)
                self._lost_count = 0
                self._last_fov = (cut.fov_h, cut.fov_v)
                self._last_good_size = (rb[2], rb[3])
                return {'bbox': rb, 'score': float(rscore),
                        'status': 'recovered', 'fov': (cut.fov_h, cut.fov_v)}

        # 未找回：输出运动外推框（尺寸钳制，防极区退化巨框），保持丢失状态
        out_bbox = _cap_box(erp_bbox_from_bfov(pred, W, H),
                            self._last_good_size, W, H)
        self._last_fov = (last_cut.fov_h, last_cut.fov_v)
        return {'bbox': out_bbox, 'score': float(last_res['score']),
                'status': 'lost', 'fov': (last_cut.fov_h, last_cut.fov_v)}
