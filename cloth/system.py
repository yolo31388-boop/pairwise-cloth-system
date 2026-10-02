"""布料系统 - 含5个bug"""
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
        self.damping: float = 0.99

    def simulate(self, delta: float, gravity: float = 9.8) -> None:
        # bug1: 显式欧拉，无子步
        for p in self.particles.values():
            if p.pinned:
                continue
            vel = ((p.position[0] - p.prev_position[0]) * self.damping,
                   (p.position[1] - p.prev_position[1]) * self.damping)
            new_pos = (p.position[0] + vel[0],
                       p.position[1] + vel[1] + gravity * delta)
            p.prev_position = p.position
            p.position = new_pos

    def solve_constraints(self, iterations: int = 1) -> None:
        # bug2: 只迭代一次，无自碰撞
        for _ in range(iterations):
            for c in self.constraints:
                p1 = self.particles[c.p1]
                p2 = self.particles[c.p2]
                dx = p2.position[0] - p1.position[0]
                dy = p2.position[1] - p1.position[1]
                dist = (dx*dx + dy*dy) ** 0.5
                if dist > 0:
                    diff = (dist - c.rest_length) / dist
                    if not p1.pinned:
                        p1.position = (p1.position[0] + dx * 0.5 * diff,
                                      p1.position[1] + dy * 0.5 * diff)
                    if not p2.pinned:
                        p2.position = (p2.position[0] - dx * 0.5 * diff,
                                      p2.position[1] - dy * 0.5 * diff)

    def resolve_collision(self, obstacle: dict) -> None:
        # bug3: 只修正位置，无摩擦
        for p in self.particles.values():
            if p.pinned:
                continue
            if obstacle["type"] == "circle":
                dx = p.position[0] - obstacle["x"]
                dy = p.position[1] - obstacle["y"]
                dist = (dx*dx + dy*dy) ** 0.5
                if dist < obstacle["radius"]:
                    p.position = (obstacle["x"] + dx/dist * obstacle["radius"],
                                 obstacle["y"] + dy/dist * obstacle["radius"])

    def apply_wind(self, wind: tuple, delta: float) -> None:
        # bug4: 只法向力，恒风
        for p in self.particles.values():
            if p.pinned:
                continue
            p.position = (p.position[0] + wind[0] * delta,
                         p.position[1] + wind[1] * delta)

    def tear_cloth(self, constraint_idx: int) -> None:
        # bug5: 只删约束，不检查结构
        if 0 <= constraint_idx < len(self.constraints):
            self.constraints.pop(constraint_idx)
