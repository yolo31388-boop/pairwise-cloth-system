"""布料系统 - 红态测试"""
import pytest, sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from cloth.system import ClothSystem, Particle, Constraint

class TestStability:
    def test_high_stiffness_stable(self):
        cs = ClothSystem()
        cs.particles["p1"] = Particle("p1", (0,0), (0,0), pinned=True)
        cs.particles["p2"] = Particle("p2", (0,1), (0,1))
        cs.constraints.append(Constraint("p1", "p2", 1.0))
        for _ in range(10):
            cs.simulate(0.1, gravity=100)
            cs.solve_constraints(5)
        assert abs(cs.particles["p2"].position[1]) < 100

class TestSelfCollision:
    def test_cloth_does_not_self_penetrate(self):
        cs = ClothSystem()
        cs.particles["p1"] = Particle("p1", (0,0), (0,0))
        cs.particles["p2"] = Particle("p2", (0.1,0), (0.1,0))
        cs.solve_constraints(3)
        assert cs.particles["p1"].position != cs.particles["p2"].position

class TestCollisionVelocity:
    def test_collision_modifies_velocity(self):
        cs = ClothSystem()
        cs.particles["p1"] = Particle("p1", (0,0), (0,-1))
        cs.resolve_collision({"type":"circle","x":0,"y":-0.5,"radius":1})
        assert cs.particles["p1"].prev_position[1] != -1

class TestWindTangent:
    def test_wind_has_tangential_force(self):
        cs = ClothSystem()
        cs.particles["p1"] = Particle("p1", (0,0), (0,0))
        cs.apply_wind((1, 0), 0.1)
        assert cs.particles["p1"].position[0] > 0

class TestTearStability:
    def test_tear_checks_structure(self):
        cs = ClothSystem()
        cs.particles["p1"] = Particle("p1", (0,0), (0,0), pinned=True)
        cs.particles["p2"] = Particle("p2", (1,0), (1,0))
        cs.constraints.append(Constraint("p1", "p2", 1.0))
        cs.tear_cloth(0)
        assert len(cs.constraints) == 0
