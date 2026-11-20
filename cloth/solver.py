"""布料解算核心。

全局契约：
- 所有位置结果按 1e-6 绝对容差比较，所有速度结果按 1e-6 绝对容差比较。
- 迭代次数一律由调用方显式传入，不使用帧数或隐藏默认值。
- 只使用标准库 math；无随机、无联网、无全局可变状态。
- 所有接口都不得原地修改传入的 positions / velocities / constraints / pinned。
"""

import math


class ClothError(Exception):
    """布料解算参数或状态非法时抛出的异常。"""


def _is_finite_number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _is_finite_vec3(vec):
    return (
        isinstance(vec, (list, tuple))
        and len(vec) == 3
        and all(_is_finite_number(component) for component in vec)
    )


def _checked_positions(positions):
    if not isinstance(positions, (list, tuple)):
        raise ClothError("positions 必须是列表")
    out = []
    for point in positions:
        if not _is_finite_vec3(point):
            raise ClothError("positions 含非法或非有限点: %r" % (point,))
        out.append([float(point[0]), float(point[1]), float(point[2])])
    return out


def _checked_constraints(constraints, count):
    if not isinstance(constraints, (list, tuple)):
        raise ClothError("constraints 必须是列表")
    out = []
    for constraint in constraints:
        if not isinstance(constraint, (list, tuple)) or len(constraint) != 3:
            raise ClothError("约束必须是 [i, j, rest]: %r" % (constraint,))
        i, j, rest = constraint
        if (
            not isinstance(i, int)
            or not isinstance(j, int)
            or isinstance(i, bool)
            or isinstance(j, bool)
            or i < 0
            or j < 0
            or i >= count
            or j >= count
        ):
            raise ClothError("约束索引越界: %r" % (constraint,))
        if not _is_finite_number(rest) or rest <= 0:
            raise ClothError("约束静止长度非法: %r" % (constraint,))
        out.append((i, j, float(rest)))
    return out


def _sub(a, b):
    return [a[0] - b[0], a[1] - b[1], a[2] - b[2]]


def _norm(vec):
    return math.sqrt(vec[0] * vec[0] + vec[1] * vec[1] + vec[2] * vec[2])


def integrate(positions, velocities, dt, gravity, damping):
    """半隐式欧拉积分一帧，阻尼按时间常数衰减。

    契约：
    - 先更新速度：v' = (v + gravity * dt) * exp(-dt / damping)，再更新位置：
      x' = x + v' * dt。damping > 0 时单位为秒；damping == 0 表示无阻尼。
    - 返回 (new_positions, new_velocities)，二者都是新的 list[list[float]]。
    - 位置容差 1e-6，速度容差 1e-6；dt 是调用方显式传入的固定时间步。
    - dt <= 0、damping < 0、维度不一致或输入含非有限值时抛 ClothError。
    """
    if not _is_finite_number(dt) or dt <= 0:
        raise ClothError("dt 必须为正有限数: %r" % (dt,))
    if not _is_finite_number(damping) or damping < 0:
        raise ClothError("damping 必须为非负有限数: %r" % (damping,))
    if not _is_finite_vec3(gravity):
        raise ClothError("gravity 必须是有限三维向量: %r" % (gravity,))
    if not isinstance(velocities, (list, tuple)) or len(positions) != len(velocities):
        raise ClothError("positions 与 velocities 维度不一致")

    pts = _checked_positions(positions)
    vels = _checked_positions(velocities)

    factor = 1.0 if damping == 0 else math.exp(-dt / damping)
    new_positions = []
    new_velocities = []
    for point, velocity in zip(pts, vels):
        new_velocity = [
            (velocity[k] + gravity[k] * dt) * factor for k in range(3)
        ]
        new_point = [point[k] + new_velocity[k] * dt for k in range(3)]
        new_velocities.append(new_velocity)
        new_positions.append(new_point)
    return new_positions, new_velocities


def solve_constraints(positions, constraints, iterations, stiffness):
    """用 PBD 距离约束求解布料，并按 stiffness 加权。

    契约：
    - constraints 中每项为 [i, j, rest]，表示点 i、j 的静止距离为 rest。
    - 执行 iterations 次迭代（显式传入，iterations >= 0）；每次迭代使用与约束顺序
      无关的雅可比式累积修正，单条约束的修正量按 stiffness（0..1）缩放。
    - 求解后静止长度误差应不超过 1e-3；位置容差 1e-6。
    - 返回新的 positions，不修改输入；iterations < 0、stiffness 越界、索引非法或
      rest <= 0 时抛 ClothError。
    """
    if not isinstance(iterations, int) or isinstance(iterations, bool) or iterations < 0:
        raise ClothError("iterations 必须为非负整数: %r" % (iterations,))
    if not _is_finite_number(stiffness) or stiffness < 0 or stiffness > 1:
        raise ClothError("stiffness 必须在 [0, 1]: %r" % (stiffness,))

    pts = _checked_positions(positions)
    cons = _checked_constraints(constraints, len(pts))

    for _ in range(iterations):
        deltas = [[0.0, 0.0, 0.0] for _ in pts]
        counts = [0] * len(pts)
        for i, j, rest in cons:
            direction = _sub(pts[j], pts[i])
            distance = _norm(direction)
            if distance == 0.0:
                # 两点重合时使用确定性方向，避免除零。
                direction = [1.0, 0.0, 0.0]
                distance = 1.0
            correction = 0.5 * stiffness * (distance - rest) / distance
            for k in range(3):
                deltas[i][k] += correction * direction[k]
                deltas[j][k] -= correction * direction[k]
            counts[i] += 1
            counts[j] += 1
        # 雅可比式应用：先累积再统一平均写入，结果与约束顺序无关。
        for index in range(len(pts)):
            if counts[index]:
                inv = 1.0 / counts[index]
                for k in range(3):
                    pts[index][k] += deltas[index][k] * inv
    return pts


def collide(positions, colliders, radius):
    """把布料点推出球体/平面碰撞体，修正到表面外 radius。

    契约：
    - 球体：{"type": "sphere", "center": [x, y, z], "radius": r}；点最终距球心
      至少 r + radius。
    - 平面：{"type": "plane", "normal": [nx, ny, nz], "offset": d}；normal 会归一化，
      点最终满足 normal·x >= d + radius。
    - 只修正位置，返回新的 positions；位置容差 1e-6。
    - 不修改 positions 与 colliders；未知类型、非法半径或非有限参数时抛 ClothError。
    """
    if not _is_finite_number(radius) or radius < 0:
        raise ClothError("radius 必须为非负有限数: %r" % (radius,))
    pts = _checked_positions(positions)
    if not isinstance(colliders, (list, tuple)):
        raise ClothError("colliders 必须是列表")

    checked = []
    for collider in colliders:
        if not isinstance(collider, dict):
            raise ClothError("碰撞体必须是 dict: %r" % (collider,))
        kind = collider.get("type")
        if kind == "sphere":
            center = collider.get("center")
            sphere_radius = collider.get("radius")
            if not _is_finite_vec3(center):
                raise ClothError("球心非法: %r" % (center,))
            if not _is_finite_number(sphere_radius) or sphere_radius < 0:
                raise ClothError("球半径非法: %r" % (sphere_radius,))
            checked.append(("sphere", [float(c) for c in center], float(sphere_radius)))
        elif kind == "plane":
            normal = collider.get("normal")
            offset = collider.get("offset")
            if not _is_finite_vec3(normal):
                raise ClothError("平面法线非法: %r" % (normal,))
            length = _norm(normal)
            if length == 0.0:
                raise ClothError("平面法线不能为零向量")
            if not _is_finite_number(offset):
                raise ClothError("平面偏移非法: %r" % (offset,))
            unit = [float(c) / length for c in normal]
            checked.append(("plane", unit, float(offset)))
        else:
            raise ClothError("未知碰撞体类型: %r" % (kind,))

    out = [list(point) for point in pts]
    for index, point in enumerate(out):
        for entry in checked:
            if entry[0] == "sphere":
                _, center, sphere_radius = entry
                min_distance = sphere_radius + radius
                direction = _sub(point, center)
                distance = _norm(direction)
                if distance < min_distance:
                    if distance == 0.0:
                        direction = [1.0, 0.0, 0.0]
                        distance = 1.0
                    scale = min_distance / distance
                    out[index] = [center[k] + direction[k] * scale for k in range(3)]
                    point = out[index]
            else:
                _, unit, offset = entry
                target = offset + radius
                signed = sum(unit[k] * point[k] for k in range(3))
                if signed < target:
                    push = target - signed
                    out[index] = [point[k] + unit[k] * push for k in range(3)]
                    point = out[index]
    return out


def pin(positions, pinned_ids):
    """把钉住的点写回原始位置。

    契约：
    - pinned_ids 为映射 {点索引: 原始位置 [x, y, z]}，例如 {0: [0.0, 0.0, 0.0]}。
    - 返回新的 positions，其中所有钉点精确恢复为映射中的原始位置；位置容差 1e-6。
    - 不修改 positions 与 pinned_ids；索引越界或位置非法时抛 ClothError。
    """
    pts = _checked_positions(positions)
    if not isinstance(pinned_ids, dict):
        raise ClothError("pinned_ids 必须是映射: %r" % (pinned_ids,))
    out = [list(point) for point in pts]
    for index, target in pinned_ids.items():
        if (
            not isinstance(index, int)
            or isinstance(index, bool)
            or index < 0
            or index >= len(out)
        ):
            raise ClothError("钉点索引越界: %r" % (index,))
        if not _is_finite_vec3(target):
            raise ClothError("钉点位置非法: %r" % (target,))
        out[index] = [float(target[0]), float(target[1]), float(target[2])]
    return out


def stretch_limit(constraints, positions, max_ratio):
    """限制单条距离约束的最大拉伸比。

    契约：
    - 对每条 [i, j, rest]，若当前距离大于 rest * max_ratio，则沿两端连线对称
      收缩到 rest * max_ratio。
    - 返回新的 positions；位置容差 1e-6；不修改 constraints 与 positions。
    - max_ratio < 1、参数非法或索引越界时抛 ClothError。
    """
    if not _is_finite_number(max_ratio) or max_ratio < 1:
        raise ClothError("max_ratio 必须为 >= 1 的有限数: %r" % (max_ratio,))
    pts = _checked_positions(positions)
    cons = _checked_constraints(constraints, len(pts))

    out = [list(point) for point in pts]
    for i, j, rest in cons:
        max_length = rest * max_ratio
        direction = _sub(out[j], out[i])
        distance = _norm(direction)
        if distance > max_length:
            excess = 0.5 * (distance - max_length) / distance
            for k in range(3):
                out[i][k] += direction[k] * excess
                out[j][k] -= direction[k] * excess
    return out


def self_collision(positions, thickness):
    """给同一块布料的点对施加最小距离约束，避免自交。

    契约：
    - 对所有点对，若距离小于 thickness，则沿两端连线对称推开到 thickness；
      两点重合时使用确定性方向，不得使用随机数。
    - 单次调用完成一轮约束（O(n^2)），不做隐藏迭代；位置容差 1e-6。
    - 返回新的 positions，不修改输入；thickness <= 0 或输入非法时抛 ClothError。
    """
    if not _is_finite_number(thickness) or thickness <= 0:
        raise ClothError("thickness 必须为正有限数: %r" % (thickness,))
    pts = _checked_positions(positions)

    out = [list(point) for point in pts]
    count = len(out)
    for i in range(count):
        for j in range(i + 1, count):
            direction = _sub(out[j], out[i])
            distance = _norm(direction)
            if distance < thickness:
                if distance == 0.0:
                    # 重合点使用确定性的索引方向，禁止随机数。
                    axis = (i + j) % 3
                    direction = [0.0, 0.0, 0.0]
                    direction[axis] = 1.0
                    distance = 1.0
                push = 0.5 * (thickness - distance) / distance
                for k in range(3):
                    out[i][k] -= direction[k] * push
                    out[j][k] += direction[k] * push
    return out


def step(positions, velocities, constraints, params):
    """完成一帧：积分 -> 距离约束 -> 碰撞 -> 钉住。

    契约：
    - 严格执行上述顺序，不改变顺序。
    - params 至少包含 dt、gravity、damping、iterations、stiffness、colliders、
      radius、pinned；可选 max_ratio、thickness 分别用于拉伸钳制和自碰撞。
    - 返回 {"positions": [...], "velocities": [...]}，不修改任何入参。
    - 位置容差 1e-6，速度容差 1e-6；迭代次数从 params["iterations"] 显式读取；
      dt <= 0、参数缺失或非法时抛 ClothError。
    """
    required = (
        "dt",
        "gravity",
        "damping",
        "iterations",
        "stiffness",
        "colliders",
        "radius",
        "pinned",
    )
    if not isinstance(params, dict):
        raise ClothError("params 必须是 dict")
    missing = [key for key in required if key not in params]
    if missing:
        raise ClothError("params 缺少参数: %r" % (missing,))

    new_positions, new_velocities = integrate(
        positions, velocities, params["dt"], params["gravity"], params["damping"]
    )
    new_positions = solve_constraints(
        new_positions, constraints, params["iterations"], params["stiffness"]
    )
    max_ratio = params.get("max_ratio")
    if max_ratio is not None:
        new_positions = stretch_limit(constraints, new_positions, max_ratio)
    thickness = params.get("thickness")
    if thickness is not None:
        new_positions = self_collision(new_positions, thickness)
    new_positions = collide(new_positions, params["colliders"], params["radius"])
    new_positions = pin(new_positions, params["pinned"])
    return {"positions": new_positions, "velocities": new_velocities}


def _require_finite(state, label):
    for vec in state:
        if not _is_finite_vec3(vec):
            raise ClothError("%s 出现非有限值: %r" % (label, vec))


def simulate_batch(frames, cloths, params):
    """批量模拟多块布料若干帧，失败时整体回滚。

    契约：
    - frames 为整数帧数（frames >= 0），即批次整体推进的帧数；每一帧按输入顺序
      轮转推进一块布（第 f 帧推进第 f % len(cloths) 块）。cloths 为
      [{"positions": ..., "velocities": ..., "constraints": ...}, ...]。
    - 每块布使用独立深拷贝推进，块与块之间不得共享可变状态；顺序保持不变。
    - 任一帧、任一块布出现 NaN/Inf 时立即抛 ClothError，不返回部分结果；输入不被
      修改，因此整体保持在初始状态（回滚）。
    - 成功时返回 {"cloths": [{"positions": ..., "velocities": ...}, ...],
      "frames": frames}，顺序与输入一致。
    - 位置容差 1e-6，速度容差 1e-6；迭代次数由 params["iterations"] 显式传入。
    """
    if not isinstance(frames, int) or isinstance(frames, bool) or frames < 0:
        raise ClothError("frames 必须为非负整数: %r" % (frames,))
    if not isinstance(cloths, (list, tuple)):
        raise ClothError("cloths 必须是列表")

    # 独立深拷贝：块与块之间、以及与调用方之间都不共享可变状态。
    states = []
    for cloth in cloths:
        if not isinstance(cloth, dict):
            raise ClothError("每块布必须是 dict: %r" % (cloth,))
        for key in ("positions", "velocities", "constraints"):
            if key not in cloth:
                raise ClothError("布料缺少字段: %r" % (key,))
        positions = _checked_positions(cloth["positions"])
        velocities = _checked_positions(cloth["velocities"])
        if len(positions) != len(velocities):
            raise ClothError("positions 与 velocities 维度不一致")
        constraints = _checked_constraints(cloth["constraints"], len(positions))
        states.append(
            {
                "positions": positions,
                "velocities": velocities,
                "constraints": [list(c) for c in constraints],
            }
        )

    for frame in range(frames):
        if not states:
            break
        # 每一帧按顺序轮转推进一块布，各布状态完全隔离。
        index = frame % len(states)
        state = states[index]
        result = step(
            state["positions"],
            state["velocities"],
            state["constraints"],
            params,
        )
        # 任一帧、任一块布出现非有限值立即中止；输入未被修改，整体回滚。
        _require_finite(result["positions"], "positions")
        _require_finite(result["velocities"], "velocities")
        states[index] = {
            "positions": result["positions"],
            "velocities": result["velocities"],
            "constraints": state["constraints"],
        }

    return {
        "cloths": [
            {"positions": state["positions"], "velocities": state["velocities"]}
            for state in states
        ],
        "frames": frames,
    }
