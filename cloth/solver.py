"""布料解算核心。

全局契约：
- 所有位置结果按 1e-6 绝对容差比较，所有速度结果按 1e-6 绝对容差比较。
- 迭代次数一律由调用方显式传入，不使用帧数或隐藏默认值。
- 只使用标准库 math；无随机、无联网、无全局可变状态。
- 所有接口都不得原地修改传入的 positions / velocities / constraints / pinned。
"""


class ClothError(Exception):
    """布料解算参数或状态非法时抛出的异常。"""


def integrate(positions, velocities, dt, gravity, damping):
    """半隐式欧拉积分一帧，阻尼按时间常数衰减。

    契约：
    - 先更新速度：v' = (v + gravity * dt) * exp(-dt / damping)，再更新位置：
      x' = x + v' * dt。damping > 0 时单位为秒；damping == 0 表示无阻尼。
    - 返回 (new_positions, new_velocities)，二者都是新的 list[list[float]]。
    - 位置容差 1e-6，速度容差 1e-6；dt 是调用方显式传入的固定时间步。
    - dt <= 0、damping < 0、维度不一致或输入含非有限值时抛 ClothError。
    """
    raise NotImplementedError


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
    raise NotImplementedError


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
    raise NotImplementedError


def pin(positions, pinned_ids):
    """把钉住的点写回原始位置。

    契约：
    - pinned_ids 为映射 {点索引: 原始位置 [x, y, z]}，例如 {0: [0.0, 0.0, 0.0]}。
    - 返回新的 positions，其中所有钉点精确恢复为映射中的原始位置；位置容差 1e-6。
    - 不修改 positions 与 pinned_ids；索引越界或位置非法时抛 ClothError。
    """
    raise NotImplementedError


def stretch_limit(constraints, positions, max_ratio):
    """限制单条距离约束的最大拉伸比。

    契约：
    - 对每条 [i, j, rest]，若当前距离大于 rest * max_ratio，则沿两端连线对称
      收缩到 rest * max_ratio。
    - 返回新的 positions；位置容差 1e-6；不修改 constraints 与 positions。
    - max_ratio < 1、参数非法或索引越界时抛 ClothError。
    """
    raise NotImplementedError


def self_collision(positions, thickness):
    """给同一块布料的点对施加最小距离约束，避免自交。

    契约：
    - 对所有点对，若距离小于 thickness，则沿两端连线对称推开到 thickness；
      两点重合时使用确定性方向，不得使用随机数。
    - 单次调用完成一轮约束（O(n^2)），不做隐藏迭代；位置容差 1e-6。
    - 返回新的 positions，不修改输入；thickness <= 0 或输入非法时抛 ClothError。
    """
    raise NotImplementedError


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
    raise NotImplementedError


def simulate_batch(frames, cloths, params):
    """批量模拟多块布料若干帧，失败时整体回滚。

    契约：
    - frames 为整数帧数（frames >= 0）；cloths 为
      [{"positions": ..., "velocities": ..., "constraints": ...}, ...]。
    - 每块布使用独立深拷贝推进，块与块之间不得共享可变状态；顺序保持不变。
    - 任一帧、任一块布出现 NaN/Inf 时立即抛 ClothError，不返回部分结果；输入不被
      修改，因此整体保持在初始状态（回滚）。
    - 成功时返回 {"cloths": [{"positions": ..., "velocities": ...}, ...],
      "frames": frames}，顺序与输入一致。
    - 位置容差 1e-6，速度容差 1e-6；迭代次数由 params["iterations"] 显式传入。
    """
    raise NotImplementedError
