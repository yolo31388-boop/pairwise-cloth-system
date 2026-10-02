"""布料物理系统

- Verlet 积分 + 子步, 高刚度/大帧长下稳定
- 距离约束多次迭代求解, 先距离约束后碰撞约束, 含自碰撞
- 碰撞响应同时修正位置与速度, 带摩擦, 支持圆形/胶囊体/网格碰撞体
- 风力含法向与切向分量, 随时间阵风变化, 支持遮挡
- 撕裂会传播并稳定分离的碎片
"""
import math
from dataclasses import dataclass


@dataclass
class Particle:
    pid: str
    position: tuple
    prev_position: tuple
    mass: float = 1.0
    pinned: bool = False


@dataclass
class Constraint:
    p1: str
    p2: str
    rest_length: float


class ClothSystem:
    def __init__(self):
        self.particles: dict[str, Particle] = {}
        self.constraints: list[Constraint] = []
        self.damping: float = 0.98          # 每 1/60s 的速度保留率
        self.substeps: int = 4              # 每帧积分子步数
        self.solver_iterations: int = 8     # 约束求解迭代次数
        self.particle_radius: float = 0.06  # 自碰撞半径
        self.friction: float = 0.3          # 碰撞摩擦系数
        self.restitution: float = 0.0       # 碰撞弹性
        self.max_speed: float = 50.0        # 速度钳制, 防止爆炸
        self.tear_threshold: float = 2.5    # 拉伸比超过该值时撕裂传播
        self.wind_tangent_factor: float = 0.5
        self.wind_gust_amplitude: float = 0.3
        self.wind_gust_frequency: float = 0.7
        self.occluders: list[dict] = []     # 挡风碰撞体
        self.colliders: list[dict] = []     # 每子步参与碰撞的碰撞体
        self.time: float = 0.0

    # ---------- 积分 ----------

    def simulate(self, delta: float, gravity: float = 9.8) -> None:
        """Verlet 积分 + 子步; 每个子步内投影约束与碰撞, 保证数值稳定。"""
        if delta <= 0:
            return
        steps = max(1, self.substeps)
        dt = delta / steps
        retention = self.damping ** (dt * 60.0)
        for _ in range(steps):
            self.time += dt
            accel = gravity * dt * dt
            for p in self.particles.values():
                if p.pinned:
                    p.prev_position = p.position
                    continue
                vx = (p.position[0] - p.prev_position[0]) * retention
                vy = (p.position[1] - p.prev_position[1]) * retention
                speed = math.hypot(vx, vy)
                if speed > self.max_speed * dt:
                    scale = self.max_speed * dt / speed
                    vx *= scale
                    vy *= scale
                p.prev_position = p.position
                p.position = (p.position[0] + vx,
                              p.position[1] + vy + accel)
            # 子步内求解, 保持 position/prev_position 一致, 避免穿透与能量注入
            self.solve_constraints()
            for collider in self.colliders:
                self.resolve_collision(collider)

    # ---------- 约束求解 ----------

    def solve_constraints(self, iterations: int | None = None) -> None:
        """先迭代求解距离约束, 再求解自碰撞约束。"""
        for _ in range(iterations or self.solver_iterations):
            for c in self.constraints:
                self._project_distance(c)
            self._solve_self_collision()

    def _project_distance(self, c: Constraint) -> None:
        p1 = self.particles[c.p1]
        p2 = self.particles[c.p2]
        dx = p2.position[0] - p1.position[0]
        dy = p2.position[1] - p1.position[1]
        dist = math.hypot(dx, dy)
        if dist <= 0:
            return
        diff = (dist - c.rest_length) / dist
        w1 = 0.0 if p1.pinned else 1.0 / p1.mass
        w2 = 0.0 if p2.pinned else 1.0 / p2.mass
        total = w1 + w2
        if total <= 0:
            return
        p1.position = (p1.position[0] + dx * diff * (w1 / total),
                       p1.position[1] + dy * diff * (w1 / total))
        p2.position = (p2.position[0] - dx * diff * (w2 / total),
                       p2.position[1] - dy * diff * (w2 / total))

    def _solve_self_collision(self) -> None:
        connected = {frozenset((c.p1, c.p2)) for c in self.constraints}
        particles = list(self.particles.values())
        min_dist = self.particle_radius * 2.0
        for i in range(len(particles)):
            for j in range(i + 1, len(particles)):
                a, b = particles[i], particles[j]
                if frozenset((a.pid, b.pid)) in connected:
                    continue
                dx = b.position[0] - a.position[0]
                dy = b.position[1] - a.position[1]
                dist = math.hypot(dx, dy)
                if dist >= min_dist:
                    continue
                if dist <= 1e-9:
                    dx, dy, dist = 1.0, 0.0, 1.0
                push = (min_dist - dist) / dist
                w1 = 0.0 if a.pinned else 1.0 / a.mass
                w2 = 0.0 if b.pinned else 1.0 / b.mass
                total = w1 + w2
                if total <= 0:
                    continue
                a.position = (a.position[0] - dx * push * (w1 / total),
                              a.position[1] - dy * push * (w1 / total))
                b.position = (b.position[0] + dx * push * (w2 / total),
                              b.position[1] + dy * push * (w2 / total))

    # ---------- 碰撞响应 ----------

    def resolve_collision(self, obstacle: dict) -> None:
        """位置 + 速度修正, 带摩擦; 支持 circle / capsule / mesh。"""
        for p in self.particles.values():
            if p.pinned:
                continue
            contact = self._contact(p.position, obstacle)
            if contact is not None:
                cx, cy, nx, ny = contact
                vx = p.position[0] - p.prev_position[0]
                vy = p.position[1] - p.prev_position[1]
                p.position = (cx, cy)
                self._correct_velocity(p, vx, vy, nx, ny)

    def _correct_velocity(self, p: Particle, vx: float, vy: float,
                          nx: float, ny: float) -> None:
        vn = vx * nx + vy * ny
        tx, ty = vx - vn * nx, vy - vn * ny
        keep = (1.0 - self.friction)
        if vn < 0:  # 正压向表面: 去掉(或反弹)法向分量
            vn = -self.restitution * vn
        vx = tx * keep + vn * nx
        vy = ty * keep + vn * ny
        p.prev_position = (p.position[0] - vx, p.position[1] - vy)

    def _contact(self, pos: tuple, obstacle: dict):
        """返回 (接触点x, 接触点y, 法线x, 法线y), 未穿透返回 None。"""
        otype = obstacle.get("type")
        if otype == "circle":
            return self._contact_circle(pos, obstacle["x"], obstacle["y"],
                                        obstacle["radius"])
        if otype == "capsule":
            qx, qy = self._closest_on_segment(
                pos, (obstacle["x1"], obstacle["y1"]),
                (obstacle["x2"], obstacle["y2"]))
            return self._contact_circle(pos, qx, qy, obstacle["radius"])
        if otype == "mesh":
            return self._contact_mesh(pos, obstacle)
        return None

    def _contact_circle(self, pos, cx, cy, radius):
        dx = pos[0] - cx
        dy = pos[1] - cy
        dist = math.hypot(dx, dy)
        if dist >= radius:
            return None
        if dist <= 1e-9:
            dx, dy, dist = 0.0, 1.0, 1.0
        nx, ny = dx / dist, dy / dist
        return (cx + nx * radius, cy + ny * radius, nx, ny)

    def _contact_mesh(self, pos, obstacle):
        """网格碰撞体: 由 vertices 构成的多边形, 带厚度。"""
        vertices = obstacle["vertices"]
        thickness = obstacle.get("thickness", self.particle_radius)
        best = None
        best_dist = float("inf")
        for i in range(len(vertices)):
            q = self._closest_on_segment(pos, vertices[i],
                                         vertices[(i + 1) % len(vertices)])
            d = math.hypot(pos[0] - q[0], pos[1] - q[1])
            if d < best_dist:
                best_dist = d
                best = q
        if best is None:
            return None
        inside = self._point_in_polygon(pos, vertices)
        if not inside and best_dist >= thickness:
            return None
        dx, dy = pos[0] - best[0], pos[1] - best[1]
        if best_dist <= 1e-9:
            dx, dy, best_dist = 0.0, 1.0, 1.0
        nx, ny = dx / best_dist, dy / best_dist
        if inside:  # 在多边形内部: 沿最近边推出
            return (best[0], best[1], -nx, -ny)
        return (best[0] + nx * thickness, best[1] + ny * thickness, nx, ny)

    @staticmethod
    def _closest_on_segment(pos, a, b):
        ax, ay = a
        bx, by = b
        abx, aby = bx - ax, by - ay
        length_sq = abx * abx + aby * aby
        if length_sq <= 1e-12:
            return a
        t = ((pos[0] - ax) * abx + (pos[1] - ay) * aby) / length_sq
        t = max(0.0, min(1.0, t))
        return (ax + abx * t, ay + aby * t)

    @staticmethod
    def _point_in_polygon(pos, vertices) -> bool:
        inside = False
        j = len(vertices) - 1
        for i in range(len(vertices)):
            xi, yi = vertices[i]
            xj, yj = vertices[j]
            if (yi > pos[1]) != (yj > pos[1]) and \
                    pos[0] < (xj - xi) * (pos[1] - yi) / (yj - yi) + xi:
                inside = not inside
            j = i
        return inside

    # ---------- 风力 ----------

    def apply_wind(self, wind: tuple, delta: float) -> None:
        """法向 + 切向风力, 阵风随时间变化, 被遮挡的粒子不受风。"""
        gust = 1.0 + self.wind_gust_amplitude * math.sin(
            self.time * self.wind_gust_frequency * 2.0 * math.pi)
        wx, wy = wind[0] * gust, wind[1] * gust
        if wx == 0 and wy == 0:
            return
        normals = self._estimate_normals()
        for p in self.particles.values():
            if p.pinned:
                continue
            if self._is_occluded(p.position, (wx, wy)):
                continue
            normal = normals.get(p.pid)
            if normal is None:
                fx, fy = wx, wy  # 无邻接边: 全部按切向拖曳
            else:
                dot = wx * normal[0] + wy * normal[1]
                fnx, fny = dot * normal[0], dot * normal[1]
                fx = fnx + (wx - fnx) * self.wind_tangent_factor
                fy = fny + (wy - fny) * self.wind_tangent_factor
            p.position = (p.position[0] + fx * delta,
                          p.position[1] + fy * delta)

    def _estimate_normals(self) -> dict:
        """由邻接约束边的平均方向估计每个粒子的布料法线。"""
        accum: dict[str, list[float]] = {}
        for c in self.constraints:
            p1 = self.particles[c.p1]
            p2 = self.particles[c.p2]
            dx = p2.position[0] - p1.position[0]
            dy = p2.position[1] - p1.position[1]
            length = math.hypot(dx, dy)
            if length <= 1e-9:
                continue
            for pid in (c.p1, c.p2):
                slot = accum.setdefault(pid, [0.0, 0.0])
                slot[0] += dx / length
                slot[1] += dy / length
        normals = {}
        for pid, (ex, ey) in accum.items():
            length = math.hypot(ex, ey)
            if length > 1e-9:
                normals[pid] = (-ey / length, ex / length)
        return normals

    def _is_occluded(self, pos: tuple, wind: tuple) -> bool:
        """从粒子逆风投射, 若被 occluder 挡住则不受风。"""
        if not self.occluders:
            return False
        speed = math.hypot(wind[0], wind[1])
        if speed <= 1e-9:
            return False
        dirx, diry = -wind[0] / speed, -wind[1] / speed
        reach = max(2.0, speed)
        for obstacle in self.occluders:
            for step in range(1, 9):
                sample = (pos[0] + dirx * reach * step / 8,
                          pos[1] + diry * reach * step / 8)
                if self._inside_shape(sample, obstacle):
                    return True
        return False

    def _inside_shape(self, pos: tuple, obstacle: dict) -> bool:
        otype = obstacle.get("type")
        if otype == "circle":
            return math.hypot(pos[0] - obstacle["x"],
                              pos[1] - obstacle["y"]) < obstacle["radius"]
        if otype == "capsule":
            qx, qy = self._closest_on_segment(
                pos, (obstacle["x1"], obstacle["y1"]),
                (obstacle["x2"], obstacle["y2"]))
            return math.hypot(pos[0] - qx, pos[1] - qy) < obstacle["radius"]
        if otype == "mesh":
            return self._point_in_polygon(pos, obstacle["vertices"])
        return False

    # ---------- 撕裂 ----------

    def tear_cloth(self, constraint_idx: int) -> None:
        """删除约束, 撕裂向过度拉伸的相邻约束传播, 并稳定脱落的碎片。"""
        if not (0 <= constraint_idx < len(self.constraints)):
            return
        torn = [self.constraints.pop(constraint_idx)]
        while torn:  # 撕裂传播: 相邻约束超载则一并断裂
            current = torn.pop()
            for pid in (current.p1, current.p2):
                for c in list(self.constraints):
                    if pid not in (c.p1, c.p2):
                        continue
                    p1 = self.particles[c.p1]
                    p2 = self.particles[c.p2]
                    stretch = math.hypot(p2.position[0] - p1.position[0],
                                         p2.position[1] - p1.position[1])
                    if c.rest_length > 0 and \
                            stretch / c.rest_length > self.tear_threshold:
                        self.constraints.remove(c)
                        torn.append(c)
        self._stabilize_detached_fragments()

    def _stabilize_detached_fragments(self) -> None:
        """撕裂后与固定点失去连接的碎片: 大幅衰减速度, 防止乱飞。"""
        adjacency: dict[str, set[str]] = {}
        for c in self.constraints:
            adjacency.setdefault(c.p1, set()).add(c.p2)
            adjacency.setdefault(c.p2, set()).add(c.p1)
        anchored = {pid for pid, p in self.particles.items() if p.pinned}
        visited = set(anchored)
        stack = list(anchored)
        while stack:
            pid = stack.pop()
            for other in adjacency.get(pid, ()):
                if other not in visited:
                    visited.add(other)
                    stack.append(other)
        for pid, p in self.particles.items():
            if pid not in visited and not p.pinned:
                vx = (p.position[0] - p.prev_position[0]) * 0.1
                vy = (p.position[1] - p.prev_position[1]) * 0.1
                p.prev_position = (p.position[0] - vx, p.position[1] - vy)
