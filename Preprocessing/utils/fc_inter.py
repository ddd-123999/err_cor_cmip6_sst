import numpy as np
from scipy.spatial import cKDTree


def idw_with_mask_and_radius(xx_t, yy_t, xx_c, yy_c, sic_c,
                             initial_radius, max_radius,
                             mask, expand_ratio=1.5, p=2):
    t_len = sic_c.shape[0]
    h, w = xx_t.shape
    result = np.full((t_len, h, w), np.nan, dtype=np.float32)

    # 扁平化源坐标
    source_coords = np.stack([xx_c.ravel(), yy_c.ravel()], axis=-1)

    # 构建 KDTree（空间位置是固定的，时间再逐天处理）
    tree = cKDTree(source_coords)

    for t in range(t_len):  # 按天数逐次插值
        sic_flat_t = sic_c[t].ravel()  # 当天的 sic 数据

        idx = 0
        for i in range(h):
            for j in range(w):
                if not mask[i, j]:
                    continue

                x_t, y_t = xx_t[i, j], yy_t[i, j]
                radius = initial_radius

                while radius <= max_radius:
                    neighbor_idx = tree.query_ball_point([x_t, y_t], r=radius)

                    if not neighbor_idx:
                        if radius * expand_ratio <= max_radius:
                            radius *= expand_ratio
                            continue
                        else:
                            idx += 1
                            break
                    elif len(neighbor_idx) < 4:
                        if radius * expand_ratio <= max_radius:
                            radius *= expand_ratio
                            continue

                    neighbor_coords = source_coords[neighbor_idx]
                    dists = np.sqrt((neighbor_coords[:, 0] - x_t) ** 2 +
                                    (neighbor_coords[:, 1] - y_t) ** 2)
                    weights = 1 / (dists ** p + 1e-12)
                    values = sic_flat_t[neighbor_idx]
                    valid = ~np.isnan(values)

                    if np.any(dists < 1e-8):
                        result[t, i, j] = values[dists.argmin()]
                        if np.isnan(result[t, i, j]):
                            print('CMIP数据在该点处为NaN', i, j)
                            idx += 1
                        break

                    if np.sum(valid) < 4:
                        if radius * expand_ratio <= max_radius:
                            radius *= expand_ratio
                            continue
                        # 超过最大范围 → 用现有有效值强制插值
                        if np.any(valid):
                            v = values[valid]
                            w_ = weights[valid]
                            result[t, i, j] = np.sum(w_ * v) / np.sum(w_)
                        else:
                            idx += 1
                        break

                    # 如果有效值 ≥ 4，正常插值
                    v = values[valid]
                    w_ = weights[valid]
                    result[t, i, j] = np.sum(w_ * v) / np.sum(w_)
                    break
        print(f'[{t+1}/{t_len}] 月完成！有：{idx}个NaN')
    return result
