"""布料系统 - 修复版"""
import math
from dataclasses import dataclass, field


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
        # 阻尼系数: 每子步速度保留比例, 小于1保证布料最终静止
        self.damping: float = 0.98
        # 速度低于该阈值直接清零, 防止布料永远抖动停不下来
        self.sleep_epsilon: float = 1e-4
        # 单个子步的最大步长, 超过则拆分子步防止穿透
        self.max_substep: float = 0.02
        # 自碰撞最小间距
        self.self_collision_radius: float = 0.05
        # 撕裂传播阈值: 拉伸比超过该值的相邻约束一并断裂
        self.tear_threshold: float = 2.0
        # 风力参数
        self.wind_drag: float = 1.0        # 切向拖拽系数
        self.wind_normal_coeff: float = 0.5  # 法向压力系数
        self.wind_gust_amplitude: float = 0.5
        self.wind_gust_frequency: float = 2.0
        self.time: float = 0.0

    # ---------- 模拟: Verlet 积分 + 子步 ----------
    def simulate(self, delta: float, gravity: float = 9.8, substeps: int = None) -> None:
        if delta <= 0:
            return
        if substeps is None:
            substeps = max(1, math.ceil(delta / self.max_substep))
        dt = delta / substeps
        for _ in range(substeps):
            self._integrate(dt, gravity)
            self.solve_constraints()
        self.time += delta

    def _integrate(self, dt: float, gravity: float) -> None:
        accel = gravity * dt * dt
        for p in self.particles.values():
            if p.pinned:
                p.prev_position = p.position
                continue
            vx = (p.position[0] - p.prev_position[0]) * self.damping
            vy = (p.position[1] - p.prev_position[1]) * self.damping
            if abs(vx) < self.sleep_epsilon:
                vx = 0.0
            if abs(vy) < self.sleep_epsilon:
                vy = 0.0
            new_pos = (p.position[0] + vx, p.position[1] + vy + accel)
            p.prev_position = p.position
            p.position = new_pos

    # ---------- 约束求解: 多次迭代, 先距离约束后碰撞约束, 含自碰撞 ----------
    def solve_constraints(self, iterations: int = 5) -> None:
        for _ in range(max(1, iterations)):
            self._solve_distance_constraints()
            self._solve_self_collision()

    def _solve_distance_constraints(self) -> None:
        for c in self.constraints:
            p1 = self.particles[c.p1]
            p2 = self.particles[c.p2]
            dx = p2.position[0] - p1.position[0]
            dy = p2.position[1] - p1.position[1]
            dist = math.hypot(dx, dy)
            if dist <= 0:
                continue
            diff = (dist - c.rest_length) / dist
            w1 = 0.0 if p1.pinned else 0.5
            w2 = 0.0 if p2.pinned else 0.5
            if p1.pinned and not p2.pinned:
                w2 = 1.0
            elif p2.pinned and not p1.pinned:
                w1 = 1.0
            p1.position = (p1.position[0] + dx * w1 * diff,
                           p1.position[1] + dy * w1 * diff)
            p2.position = (p2.position[0] - dx * w2 * diff,
                           p2.position[1] - dy * w2 * diff)

    def _solve_self_collision(self) -> None:
        connected = set()
        for c in self.constraints:
            connected.add((c.p1, c.p2))
            connected.add((c.p2, c.p1))
        ids = list(self.particles.keys())
        min_dist = self.self_collision_radius
        for i in range(len(ids)):
            for j in range(i + 1, len(ids)):
                a, b = ids[i], ids[j]
                if (a, b) in connected:
                    continue
                pa = self.particles[a]
                pb = self.particles[b]
                dx = pb.position[0] - pa.position[0]
                dy = pb.position[1] - pa.position[1]
                dist = math.hypot(dx, dy)
                if dist >= min_dist:
                    continue
                if dist <= 1e-12:
                    dx, dy, dist = 1.0, 0.0, 1e-12
                push = (min_dist - dist) / dist
                wa = 0.0 if pa.pinned else 0.5
                wb = 0.0 if pb.pinned else 0.5
                if pa.pinned and not pb.pinned:
                    wb = 1.0
                elif pb.pinned and not pa.pinned:
                    wa = 1.0
                pa.position = (pa.position[0] - dx * wa * push,
                               pa.position[1] - dy * wa * push)
                pb.position = (pb.position[0] + dx * wb * push,
                               pb.position[1] + dy * wb * push)

    # ---------- 碰撞响应: 修正位置+速度, 含摩擦, 支持圆/胶囊体/网格 ----------
    def resolve_collision(self, obstacle: dict, friction: float = None,
                          restitution: float = 0.0) -> None:
        if friction is None:
            friction = obstacle.get("friction", 0.3)
        for p in self.particles.values():
            if p.pinned:
                continue
            hit = self._push_out(p.position, obstacle)
            if hit is None:
                continue
            new_pos, normal = hit
            vx = p.position[0] - p.prev_position[0]
            vy = p.position[1] - p.prev_position[1]
            vn = vx * normal[0] + vy * normal[1]
            tx, ty = vx - vn * normal[0], vy - vn * normal[1]
            # 摩擦: 衰减切向速度
            tx *= (1.0 - friction)
            ty *= (1.0 - friction)
            # 法向速度: 撞入表面的分量按弹性系数反弹(默认0即吸收)
            vn_out = -vn * restitution if vn < 0 else vn
            nvx = tx + vn_out * normal[0]
            nvy = ty + vn_out * normal[1]
            p.position = new_pos
            p.prev_position = (new_pos[0] - nvx, new_pos[1] - nvy)

    def _push_out(self, pos: tuple, obstacle: dict):
        otype = obstacle["type"]
        if otype == "circle":
            dx = pos[0] - obstacle["x"]
            dy = pos[1] - obstacle["y"]
            dist = math.hypot(dx, dy)
            r = obstacle["radius"]
            if dist >= r:
                return None
            if dist <= 1e-12:
                dx, dy, dist = 0.0, 1.0, 1e-12
            nx, ny = dx / dist, dy / dist
            return ((obstacle["x"] + nx * r, obstacle["y"] + ny * r), (nx, ny))
        if otype == "capsule":
            cx, cy = self._closest_point_on_segment(
                pos, (obstacle["x1"], obstacle["y1"]),
                (obstacle["x2"], obstacle["y2"]))
            dx = pos[0] - cx
            dy = pos[1] - cy
            dist = math.hypot(dx, dy)
            r = obstacle["radius"]
            if dist >= r:
                return None
            if dist <= 1e-12:
                dx, dy, dist = 0.0, 1.0, 1e-12
            nx, ny = dx / dist, dy / dist
            return ((cx + nx * r, cy + ny * r), (nx, ny))
        if otype == "mesh":
            return self._push_out_polygon(pos, obstacle["points"])
        raise ValueError(f"未知碰撞体类型: {otype}")

    @staticmethod
    def _closest_point_on_segment(pos, a, b):
        ax, ay = a
        bx, by = b
        abx, aby = bx - ax, by - ay
        length_sq = abx * abx + aby * aby
        if length_sq <= 1e-12:
            return a
        t = ((pos[0] - ax) * abx + (pos[1] - ay) * aby) / length_sq
        t = max(0.0, min(1.0, t))
        return (ax + abx * t, ay + aby * t)

    def _push_out_polygon(self, pos, points):
        # 网格碰撞体: 顶点列表构成的凸/凹多边形
        inside = False
        n = len(points)
        for i in range(n):
            x1, y1 = points[i]
            x2, y2 = points[(i + 1) % n]
            if (y1 > pos[1]) != (y2 > pos[1]):
                xin = (x2 - x1) * (pos[1] - y1) / (y2 - y1) + x1
                if pos[0] < xin:
                    inside = not inside
        if not inside:
            return None
        best = None
        for i in range(n):
            a = points[i]
            b = points[(i + 1) % n]
            c = self._closest_point_on_segment(pos, a, b)
            d = math.hypot(pos[0] - c[0], pos[1] - c[1])
            if best is None or d < best[0]:
                best = (d, c, a, b)
        _, c, a, b = best
        ex, ey = b[0] - a[0], b[1] - a[1]
        elen = math.hypot(ex, ey)
        if elen <= 1e-12:
            return None
        # 多边形顶点按逆时针时外法线为 (ey, -ex)
        nx, ny = ey / elen, -ex / elen
        to_center = (sum(pt[0] for pt in points) / n - c[0],
                     sum(pt[1] for pt in points) / n - c[1])
        if nx * to_center[0] + ny * to_center[1] > 0:
            nx, ny = -nx, -ny
        eps = 1e-6
        return ((c[0] + nx * eps, c[1] + ny * eps), (nx, ny))

    # ---------- 风力: 法向+切向, 阵风变化, 遮挡 ----------
    def apply_wind(self, wind: tuple, delta: float,
                   occluders: list = None, time: float = None) -> None:
        t = self.time if time is None else time
        gust = 1.0 + self.wind_gust_amplitude * math.sin(
            t * self.wind_gust_frequency)
        wx, wy = wind[0] * gust, wind[1] * gust
        wind_len = math.hypot(wx, wy)
        if wind_len <= 1e-12:
            return
        wind_dir = (wx / wind_len, wy / wind_len)
        for p in self.particles.values():
            if p.pinned:
                continue
            scale = 1.0
            if occluders and self._is_occluded(p.position, wind_dir, occluders):
                scale = 0.2  # 被身体等遮挡物挡住的部分只受少量风
            normal = self._estimate_normal(p.pid)
            # 切向力: 风沿布料表面的拖拽
            tx, ty = wx * self.wind_drag, wy * self.wind_drag
            # 法向力: 风在布料法向上的压力
            dot = wx * normal[0] + wy * normal[1]
            nx = dot * normal[0] * self.wind_normal_coeff
            ny = dot * normal[1] * self.wind_normal_coeff
            fx = (tx + nx) * scale
            fy = (ty + ny) * scale
            p.position = (p.position[0] + fx * delta,
                          p.position[1] + fy * delta)

    def _estimate_normal(self, pid: str) -> tuple:
        # 由相邻约束边估算布料法线(2D: 边的垂直方向)
        neighbors = []
        for c in self.constraints:
            if c.p1 == pid:
                neighbors.append(c.p2)
            elif c.p2 == pid:
                neighbors.append(c.p1)
        if not neighbors:
            return (0.0, 1.0)
        p = self.particles[pid]
        ex = ey = 0.0
        for nid in neighbors:
            q = self.particles[nid]
            ex += q.position[0] - p.position[0]
            ey += q.position[1] - p.position[1]
        elen = math.hypot(ex, ey)
        if elen <= 1e-12:
            return (0.0, 1.0)
        return (-ey / elen, ex / elen)

    @staticmethod
    def _is_occluded(pos: tuple, wind_dir: tuple, occluders: list) -> bool:
        # 从粒子逆风向发射线, 命中遮挡物则认为处于风影中
        ox, oy = -wind_dir[0], -wind_dir[1]
        for oc in occluders:
            cx, cy, r = oc["x"], oc["y"], oc["radius"]
            fx, fy = pos[0] - cx, pos[1] - cy
            b = 2.0 * (fx * ox + fy * oy)
            c = fx * fx + fy * fy - r * r
            disc = b * b - 4.0 * c
            if disc < 0:
                continue
            sq = math.sqrt(disc)
            t1 = (-b - sq) / 2.0
            t2 = (-b + sq) / 2.0
            if t1 > 1e-9 or t2 > 1e-9:
                return True
        return False

    # ---------- 撕裂: 传播 + 结构稳定性检查 ----------
    def tear_cloth(self, constraint_idx: int) -> None:
        if not (0 <= constraint_idx < len(self.constraints)):
            return
        self.constraints.pop(constraint_idx)
        self._propagate_tear()
        self._stabilize_detached_fragments()

    def _propagate_tear(self) -> None:
        # 撕裂传播: 拉伸比超过阈值的相邻约束随之断裂
        changed = True
        while changed:
            changed = False
            for i in range(len(self.constraints) - 1, -1, -1):
                c = self.constraints[i]
                p1 = self.particles[c.p1]
                p2 = self.particles[c.p2]
                dist = math.hypot(p2.position[0] - p1.position[0],
                                  p2.position[1] - p1.position[1])
                if c.rest_length > 0 and dist / c.rest_length > self.tear_threshold:
                    self.constraints.pop(i)
                    changed = True

    def _stabilize_detached_fragments(self) -> None:
        # 结构稳定性: 无固定点的碎片清零速度, 防止撕裂后乱飞
        adjacency: dict[str, list] = {pid: [] for pid in self.particles}
        for c in self.constraints:
            adjacency[c.p1].append(c.p2)
            adjacency[c.p2].append(c.p1)
        visited = set()
        for pid in self.particles:
            if pid in visited:
                continue
            stack = [pid]
            component = []
            has_pin = False
            while stack:
                cur = stack.pop()
                if cur in visited:
                    continue
                visited.add(cur)
                component.append(cur)
                if self.particles[cur].pinned:
                    has_pin = True
                stack.extend(adjacency[cur])
            if not has_pin:
                for cid in component:
                    p = self.particles[cid]
                    p.prev_position = p.position
