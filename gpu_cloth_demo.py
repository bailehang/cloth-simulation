"""GPU cloth simulation demo powered by Taichi/CUDA.

The expensive particle integration, Jacobi constraint projection, and Bunny
triangle collision queries run in Taichi kernels.  Python only builds topology,
loads the PLY mesh, and drives the GGUI controls.
"""

import argparse
import json
import math
import os
import sys
import time
import traceback
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import taichi as ti


ROOT = Path(__file__).resolve().parent
MAX_PARTICLES = 8192
MAX_CONSTRAINTS = 40000
MAX_RENDER_INDICES = 120000
# Render/collision safety envelope.  It is intentionally larger than the
# visual surface epsilon so constraint projection cannot reveal z-fighting.
CLOTH_THICKNESS = 0.022
MAX_PARTICLE_STEP = 0.032
MAX_WIND = 100.0

ALGORITHMS = {"pbd": 0, "xpbd": 1, "havok": 2}
CLOTH_TYPES = {"cape", "scarf", "hair"}


@dataclass
class MeshData:
    vertices: np.ndarray
    faces: np.ndarray
    normals: np.ndarray
    bounds_min: np.ndarray
    bounds_max: np.ndarray


@dataclass
class ClothData:
    positions: np.ndarray
    inv_mass: np.ndarray
    constraints: np.ndarray
    rest_lengths: np.ndarray
    constraint_types: np.ndarray
    render_indices: np.ndarray
    render_mode: str
    color: tuple[float, float, float]


def load_ascii_ply(path: Path, target_height: float = 1.6) -> MeshData:
    """Load the bundled ASCII PLY without a third-party mesh dependency."""
    with path.open("r", encoding="ascii") as stream:
        if stream.readline().strip() != "ply":
            raise ValueError(f"Not a PLY file: {path}")
        vertex_count = face_count = 0
        while True:
            line = stream.readline()
            if not line:
                raise ValueError("Unexpected end of PLY header")
            tokens = line.split()
            if tokens[:2] == ["element", "vertex"]:
                vertex_count = int(tokens[2])
            elif tokens[:2] == ["element", "face"]:
                face_count = int(tokens[2])
            elif tokens[0] == "end_header":
                break
        vertices = np.asarray(
            [[float(value) for value in stream.readline().split()[:3]] for _ in range(vertex_count)],
            dtype=np.float32,
        )
        faces = np.empty((face_count, 3), dtype=np.int32)
        for face_index in range(face_count):
            values = [int(value) for value in stream.readline().split()]
            if values[0] != 3:
                raise ValueError("Only triangular Bunny faces are supported")
            faces[face_index] = values[1:4]

    source_min = vertices.min(axis=0)
    source_max = vertices.max(axis=0)
    source_center = (source_min + source_max) * 0.5
    scale = target_height / (source_max[1] - source_min[1])
    vertices[:, 0] = (vertices[:, 0] - source_center[0]) * scale
    vertices[:, 1] = (vertices[:, 1] - source_min[1]) * scale
    vertices[:, 2] = (vertices[:, 2] - source_center[2]) * scale

    normals = np.zeros_like(vertices)
    a, b, c = vertices[faces[:, 0]], vertices[faces[:, 1]], vertices[faces[:, 2]]
    face_normals = np.cross(b - a, c - a)
    lengths = np.linalg.norm(face_normals, axis=1, keepdims=True)
    face_normals /= np.maximum(lengths, 1e-12)
    for corner in range(3):
        np.add.at(normals, faces[:, corner], face_normals)
    normals /= np.maximum(np.linalg.norm(normals, axis=1, keepdims=True), 1e-12)
    return MeshData(
        vertices=vertices.astype(np.float32),
        faces=faces,
        normals=normals.astype(np.float32),
        bounds_min=vertices.min(axis=0),
        bounds_max=vertices.max(axis=0),
    )


def add_constraint(
    constraints: list[tuple[int, int]],
    types: list[int],
    first: int,
    second: int,
    constraint_type: int = 0,
) -> None:
    constraints.append((first, second))
    types.append(constraint_type)


def finalize_cloth(
    positions: list[tuple[float, float, float]],
    pinned: set[int],
    constraints: list[tuple[int, int]],
    constraint_types: list[int],
    render_indices: list[int],
    render_mode: str,
    color: tuple[float, float, float],
) -> ClothData:
    position_array = np.asarray(positions, dtype=np.float32)
    constraint_array = np.asarray(constraints, dtype=np.int32)
    inv_mass = np.ones(len(position_array), dtype=np.float32)
    if pinned:
        inv_mass[np.asarray(sorted(pinned), dtype=np.int32)] = 0.0
    rest = np.linalg.norm(
        position_array[constraint_array[:, 1]] - position_array[constraint_array[:, 0]], axis=1
    ).astype(np.float32)
    return ClothData(
        positions=position_array,
        inv_mass=inv_mass,
        constraints=constraint_array,
        rest_lengths=rest,
        constraint_types=np.asarray(constraint_types, dtype=np.int32),
        render_indices=np.asarray(render_indices, dtype=np.int32),
        render_mode=render_mode,
        color=color,
    )


def build_cape(mesh: MeshData) -> ClothData:
    cols, rows, spacing = 18, 19, 0.045
    size = mesh.bounds_max - mesh.bounds_min
    head_y = size[1] * 0.72
    front_z = size[2] * 0.35
    body_radius = size[0] * 0.4
    start_x = -((cols - 1) * spacing) * 0.5
    # Python demo cape hangs in front of the Bunny. The small positive
    # clearance gives the collision solver room to settle the four anchors.
    start_z = front_z + body_radius * 0.08
    positions = [
        (start_x + column * spacing, head_y + 0.02 - row * spacing, start_z)
        for row in range(rows)
        for column in range(cols)
    ]
    pinned = {6, 8, 9, 11}
    constraints: list[tuple[int, int]] = []
    types: list[int] = []
    triangles: list[int] = []
    index = lambda row, column: row * cols + column
    for row in range(rows):
        for column in range(cols):
            current = index(row, column)
            if column + 1 < cols:
                add_constraint(constraints, types, current, index(row, column + 1))
            if row + 1 < rows:
                add_constraint(constraints, types, current, index(row + 1, column))
            if row + 1 < rows and column + 1 < cols:
                add_constraint(constraints, types, current, index(row + 1, column + 1))
                add_constraint(constraints, types, index(row, column + 1), index(row + 1, column))
                a, b = current, index(row, column + 1)
                c, d = index(row + 1, column), index(row + 1, column + 1)
                triangles.extend((a, c, b, b, c, d))
            if column + 2 < cols:
                add_constraint(constraints, types, current, index(row, column + 2), 1)
            if row + 2 < rows:
                add_constraint(constraints, types, current, index(row + 2, column), 1)
    return finalize_cloth(positions, pinned, constraints, types, triangles, "mesh", (0.82, 0.12, 0.32))


def build_scarf(mesh: MeshData) -> ClothData:
    segments, rings, ring_height = 20, 21, 0.07
    size = mesh.bounds_max - mesh.bounds_min
    neck_y = size[1] * 0.55
    neck_radius = size[0] * 0.4 * 0.8
    hem_radius = size[0] * 0.4 * 1.1
    positions: list[tuple[float, float, float]] = []
    for ring in range(rings):
        blend = ring / (rings - 1)
        radius = neck_radius + (hem_radius - neck_radius) * blend
        for segment in range(segments):
            angle = segment / segments * math.tau
            positions.append((math.cos(angle) * radius, neck_y - ring * ring_height, math.sin(angle) * radius))
    pinned = set(range(segments))
    constraints: list[tuple[int, int]] = []
    types: list[int] = []
    triangles: list[int] = []
    index = lambda ring, segment: ring * segments + segment % segments
    for ring in range(rings):
        for segment in range(segments):
            current = index(ring, segment)
            add_constraint(constraints, types, current, index(ring, segment + 1))
            if ring + 1 < rings:
                add_constraint(constraints, types, current, index(ring + 1, segment))
                a, b = current, index(ring, segment + 1)
                c, d = index(ring + 1, segment), index(ring + 1, segment + 1)
                triangles.extend((a, c, b, b, c, d))
            if ring + 2 < rings:
                add_constraint(constraints, types, current, index(ring + 2, segment), 1)
    return finalize_cloth(positions, pinned, constraints, types, triangles, "mesh", (0.18, 0.38, 0.92))


def build_hair(mesh: MeshData) -> ClothData:
    segments, strands, spacing = 14, 8, 0.055
    size = mesh.bounds_max - mesh.bounds_min
    ear_y = size[1] * 0.92
    positions: list[tuple[float, float, float]] = []
    for strand in range(strands):
        side = -0.08 if strand < strands // 2 else 0.08
        local = strand % (strands // 2)
        angle = local / (strands // 2) * math.tau
        base = (side + math.cos(angle) * 0.03, ear_y, 0.04 + math.sin(angle) * 0.03)
        positions.extend((base[0], base[1] - segment * spacing, base[2]) for segment in range(segments))
    pinned = {strand * segments for strand in range(strands)}
    constraints: list[tuple[int, int]] = []
    types: list[int] = []
    lines: list[int] = []
    for strand in range(strands):
        base = strand * segments
        for segment in range(segments - 1):
            add_constraint(constraints, types, base + segment, base + segment + 1)
            lines.extend((base + segment, base + segment + 1))
        for segment in range(segments - 2):
            add_constraint(constraints, types, base + segment, base + segment + 2, 1)
    return finalize_cloth(positions, pinned, constraints, types, lines, "lines", (0.88, 0.26, 0.58))


class CollisionGrid:
    """CPU-built triangle cell list consumed directly by CUDA kernels."""

    def __init__(self, mesh: MeshData, cell_size: float = 0.08) -> None:
        self.cell_size = float(cell_size)
        self.minimum = (mesh.bounds_min - cell_size).astype(np.float32)
        maximum = mesh.bounds_max + cell_size
        self.dims = np.maximum(np.ceil((maximum - self.minimum) / cell_size).astype(np.int32), 1)
        cell_count = int(np.prod(self.dims))
        buckets: list[list[int]] = [[] for _ in range(cell_count)]
        triangles = mesh.vertices[mesh.faces]
        for triangle_index, triangle in enumerate(triangles):
            low = np.floor((triangle.min(axis=0) - self.minimum - CLOTH_THICKNESS) / cell_size).astype(int)
            high = np.floor((triangle.max(axis=0) - self.minimum + CLOTH_THICKNESS) / cell_size).astype(int)
            low = np.clip(low, 0, self.dims - 1)
            high = np.clip(high, 0, self.dims - 1)
            for x in range(low[0], high[0] + 1):
                for y in range(low[1], high[1] + 1):
                    for z in range(low[2], high[2] + 1):
                        flat = (x * self.dims[1] + y) * self.dims[2] + z
                        buckets[flat].append(triangle_index)
        offsets = np.zeros(cell_count + 1, dtype=np.int32)
        offsets[1:] = np.cumsum([len(bucket) for bucket in buckets], dtype=np.int32)
        self.offsets = offsets
        self.triangle_ids = np.asarray([value for bucket in buckets for value in bucket], dtype=np.int32)


@ti.data_oriented
class GpuClothSimulation:
    def __init__(self, mesh: MeshData, collision_grid: CollisionGrid) -> None:
        self.mesh = mesh
        self.grid = collision_grid
        self.particle_count = 0
        self.constraint_count = 0
        self.render_count = 0
        self.render_mode = "mesh"
        self.cloth_color = (0.82, 0.12, 0.32)
        self.algorithm = ALGORITHMS["pbd"]
        self.cloth_type = "cape"
        self.gravity = 9.8
        self.wind = 3.0
        self.stiffness = 0.5
        self.iterations = 6
        self.damping = 0.02
        self.collision_enabled = True
        self.wind_enabled = True
        self.paused = False
        self.wireframe = False
        self.bunny_offset_value = 0.0

        self.x = ti.Vector.field(3, ti.f32, shape=MAX_PARTICLES)
        self.previous = ti.Vector.field(3, ti.f32, shape=MAX_PARTICLES)
        self.velocity = ti.Vector.field(3, ti.f32, shape=MAX_PARTICLES)
        self.pin_position = ti.Vector.field(3, ti.f32, shape=MAX_PARTICLES)
        self.inv_mass = ti.field(ti.f32, shape=MAX_PARTICLES)
        self.correction = ti.Vector.field(3, ti.f32, shape=MAX_PARTICLES)
        self.correction_count = ti.field(ti.i32, shape=MAX_PARTICLES)
        self.constraint_pair = ti.Vector.field(2, ti.i32, shape=MAX_CONSTRAINTS)
        self.rest_length = ti.field(ti.f32, shape=MAX_CONSTRAINTS)
        self.constraint_type = ti.field(ti.i32, shape=MAX_CONSTRAINTS)
        self.lagrange = ti.field(ti.f32, shape=MAX_CONSTRAINTS)
        self.render_indices = ti.field(ti.i32, shape=MAX_RENDER_INDICES)
        self.bunny_offset = ti.Vector.field(3, ti.f32, shape=())
        self.contact_count = ti.field(ti.i32, shape=())

        self.bunny_vertices = ti.Vector.field(3, ti.f32, shape=len(mesh.vertices))
        self.bunny_render_vertices = ti.Vector.field(3, ti.f32, shape=len(mesh.vertices))
        self.bunny_normals = ti.Vector.field(3, ti.f32, shape=len(mesh.vertices))
        self.bunny_faces = ti.field(ti.i32, shape=mesh.faces.size)
        self.triangle_vertices = ti.Vector.field(3, ti.f32, shape=(len(mesh.faces), 3))
        self.triangle_normals = ti.Vector.field(3, ti.f32, shape=len(mesh.faces))
        self.grid_offsets = ti.field(ti.i32, shape=len(collision_grid.offsets))
        self.grid_triangle_ids = ti.field(ti.i32, shape=max(1, len(collision_grid.triangle_ids)))

        self.bunny_vertices.from_numpy(mesh.vertices)
        self.bunny_render_vertices.from_numpy(mesh.vertices)
        self.bunny_normals.from_numpy(mesh.normals)
        self.bunny_faces.from_numpy(mesh.faces.reshape(-1))
        triangle_vertices = mesh.vertices[mesh.faces].astype(np.float32)
        self.triangle_vertices.from_numpy(triangle_vertices)
        triangle_normals = np.cross(triangle_vertices[:, 1] - triangle_vertices[:, 0], triangle_vertices[:, 2] - triangle_vertices[:, 0])
        triangle_normals /= np.maximum(np.linalg.norm(triangle_normals, axis=1, keepdims=True), 1e-12)
        self.triangle_normals.from_numpy(triangle_normals.astype(np.float32))
        self.grid_offsets.from_numpy(collision_grid.offsets)
        if len(collision_grid.triangle_ids):
            self.grid_triangle_ids.from_numpy(collision_grid.triangle_ids)
        else:
            self.grid_triangle_ids.from_numpy(np.zeros(1, dtype=np.int32))
        self.bunny_offset[None] = (0.0, 0.0, 0.0)
        self.set_cloth("cape")

    def set_cloth(self, cloth_type: str) -> None:
        builders = {"cape": build_cape, "scarf": build_scarf, "hair": build_hair}
        data = builders[cloth_type](self.mesh)
        if len(data.positions) > MAX_PARTICLES or len(data.constraints) > MAX_CONSTRAINTS:
            raise ValueError("Cloth topology exceeds allocated GPU capacity")
        if len(data.render_indices) > MAX_RENDER_INDICES:
            raise ValueError("Cloth render topology exceeds allocated GPU capacity")
        self.cloth_type = cloth_type
        self.particle_count = len(data.positions)
        self.constraint_count = len(data.constraints)
        self.render_count = len(data.render_indices)
        self.render_mode = data.render_mode
        self.cloth_color = data.color

        def padded(values: np.ndarray, shape: tuple[int, ...], dtype: np.dtype) -> np.ndarray:
            output = np.zeros(shape, dtype=dtype)
            output[: len(values)] = values
            return output

        positions = padded(data.positions, (MAX_PARTICLES, 3), np.float32)
        inv_mass = padded(data.inv_mass, (MAX_PARTICLES,), np.float32)
        pairs = padded(data.constraints, (MAX_CONSTRAINTS, 2), np.int32)
        rest = padded(data.rest_lengths, (MAX_CONSTRAINTS,), np.float32)
        types = padded(data.constraint_types, (MAX_CONSTRAINTS,), np.int32)
        indices = padded(data.render_indices, (MAX_RENDER_INDICES,), np.int32)
        self.x.from_numpy(positions)
        self.previous.from_numpy(positions)
        self.pin_position.from_numpy(positions)
        self.velocity.fill(0.0)
        self.inv_mass.from_numpy(inv_mass)
        self.constraint_pair.from_numpy(pairs)
        self.rest_length.from_numpy(rest)
        self.constraint_type.from_numpy(types)
        self.render_indices.from_numpy(indices)
        self.lagrange.fill(0.0)

    @ti.func
    def closest_on_segment(self, point, start, end):
        direction = end - start
        denominator = direction.dot(direction)
        parameter = 0.0
        if denominator > 1e-12:
            parameter = ti.max(0.0, ti.min(1.0, (point - start).dot(direction) / denominator))
        return start + direction * parameter

    @ti.func
    def closest_on_triangle(self, point, a, b, c):
        ab, ac = b - a, c - a
        normal = ab.cross(ac)
        normal_length_sq = normal.dot(normal)
        projected = a
        if normal_length_sq > 1e-12:
            projected = point - normal * ((point - a).dot(normal) / normal_length_sq)
        d00, d01, d11 = ab.dot(ab), ab.dot(ac), ac.dot(ac)
        relative = projected - a
        d20, d21 = relative.dot(ab), relative.dot(ac)
        denominator = d00 * d11 - d01 * d01
        inside = 0
        if ti.abs(denominator) > 1e-12:
            v = (d11 * d20 - d01 * d21) / denominator
            w = (d00 * d21 - d01 * d20) / denominator
            u = 1.0 - v - w
            inside = ti.cast(u >= 0.0 and v >= 0.0 and w >= 0.0, ti.i32)
        result = projected
        if inside == 0:
            first = self.closest_on_segment(point, a, b)
            second = self.closest_on_segment(point, b, c)
            third = self.closest_on_segment(point, c, a)
            result = first
            best = (point - first).dot(point - first)
            second_distance = (point - second).dot(point - second)
            third_distance = (point - third).dot(point - third)
            if second_distance < best:
                result, best = second, second_distance
            if third_distance < best:
                result = third
        return result

    @ti.kernel
    def update_bunny_render_vertices(self):
        for index in self.bunny_render_vertices:
            self.bunny_render_vertices[index] = self.bunny_vertices[index] + self.bunny_offset[None]

    @ti.kernel
    def update_pins(self, count: ti.i32):
        for index in range(count):
            if self.inv_mass[index] == 0.0:
                target = self.pin_position[index] + self.bunny_offset[None]
                self.velocity[index] = target - self.x[index]
                self.x[index] = target
                self.previous[index] = target

    @ti.kernel
    def integrate(self, count: ti.i32, dt: ti.f32, gravity: ti.f32, wind: ti.f32, damping: ti.f32, elapsed: ti.f32):
        damping_factor = ti.exp(-damping * dt * 60.0)
        wind_direction = ti.Vector([0.45 + ti.sin(elapsed * 0.7) * 0.35, 0.05, 0.65])
        wind_direction = wind_direction.normalized()
        for index in range(count):
            if self.inv_mass[index] > 0.0:
                self.previous[index] = self.x[index]
                self.velocity[index].y -= gravity * dt
                self.velocity[index] += wind_direction * wind * dt * 0.5
                self.velocity[index] *= damping_factor
                displacement = self.velocity[index] * dt
                displacement_length = displacement.norm()
                if displacement_length > MAX_PARTICLE_STEP:
                    displacement *= MAX_PARTICLE_STEP / displacement_length
                    self.velocity[index] = displacement / dt
                self.x[index] += displacement

    @ti.kernel
    def reset_lagrange(self, count: ti.i32):
        for index in range(count):
            self.lagrange[index] = 0.0

    @ti.kernel
    def clear_corrections(self, count: ti.i32):
        for index in range(count):
            self.correction[index] = ti.Vector([0.0, 0.0, 0.0])
            self.correction_count[index] = 0

    @ti.kernel
    def solve_constraints(
        self,
        constraint_count: ti.i32,
        algorithm: ti.i32,
        stiffness: ti.f32,
        dt: ti.f32,
    ):
        for constraint_index in range(constraint_count):
            pair = self.constraint_pair[constraint_index]
            first, second = pair[0], pair[1]
            delta = self.x[second] - self.x[first]
            distance = delta.norm()
            if distance > 1e-8:
                first_weight, second_weight = self.inv_mass[first], self.inv_mass[second]
                weight_sum = first_weight + second_weight
                if weight_sum > 1e-8:
                    bend_scale = 0.28 if self.constraint_type[constraint_index] == 1 else 1.0
                    first_delta = ti.Vector([0.0, 0.0, 0.0])
                    second_delta = ti.Vector([0.0, 0.0, 0.0])
                    if algorithm == 1:
                        compliance = (1e-5 / ti.max(stiffness, 0.01)) * (3.0 if self.constraint_type[constraint_index] == 1 else 1.0)
                        alpha = compliance / (dt * dt)
                        constraint_value = distance - self.rest_length[constraint_index]
                        delta_lambda = (-constraint_value - alpha * self.lagrange[constraint_index]) / (weight_sum + alpha)
                        self.lagrange[constraint_index] += delta_lambda
                        scale = delta_lambda / distance
                        first_delta = delta * (-first_weight * scale)
                        second_delta = delta * (second_weight * scale)
                    else:
                        correction_scale = stiffness * bend_scale * (distance - self.rest_length[constraint_index]) / distance
                        first_delta = delta * (correction_scale * first_weight / weight_sum)
                        second_delta = delta * (-correction_scale * second_weight / weight_sum)
                    if first_weight > 0.0:
                        for component in ti.static(range(3)):
                            ti.atomic_add(self.correction[first][component], first_delta[component])
                        ti.atomic_add(self.correction_count[first], 1)
                    if second_weight > 0.0:
                        for component in ti.static(range(3)):
                            ti.atomic_add(self.correction[second][component], second_delta[component])
                        ti.atomic_add(self.correction_count[second], 1)

    @ti.kernel
    def apply_corrections(self, count: ti.i32):
        for index in range(count):
            if self.inv_mass[index] > 0.0 and self.correction_count[index] > 0:
                self.x[index] += self.correction[index] / ti.cast(self.correction_count[index], ti.f32)

    @ti.kernel
    def collide_with_mesh(
        self,
        count: ti.i32,
        grid_min_x: ti.f32,
        grid_min_y: ti.f32,
        grid_min_z: ti.f32,
        cell_size: ti.f32,
        dim_x: ti.i32,
        dim_y: ti.i32,
        dim_z: ti.i32,
    ):
        for index in range(count):
            local = self.x[index] - self.bunny_offset[None]
            # Include both dynamic particles and zero-inverse-mass anchors.
            if self.inv_mass[index] >= 0.0:
                coordinate = ti.cast(
                    ti.floor((local - ti.Vector([grid_min_x, grid_min_y, grid_min_z])) / cell_size), ti.i32
                )
                best_distance_sq = (cell_size * 2.25) ** 2
                best_point = local
                best_normal = ti.Vector([0.0, 1.0, 0.0])
                found = 0
                for offset_x, offset_y, offset_z in ti.ndrange((-1, 2), (-1, 2), (-1, 2)):
                    x = coordinate.x + offset_x
                    y = coordinate.y + offset_y
                    z = coordinate.z + offset_z
                    if 0 <= x < dim_x and 0 <= y < dim_y and 0 <= z < dim_z:
                        cell = (x * dim_y + y) * dim_z + z
                        for grid_index in range(self.grid_offsets[cell], self.grid_offsets[cell + 1]):
                            triangle_index = self.grid_triangle_ids[grid_index]
                            closest = self.closest_on_triangle(
                                local,
                                self.triangle_vertices[triangle_index, 0],
                                self.triangle_vertices[triangle_index, 1],
                                self.triangle_vertices[triangle_index, 2],
                            )
                            distance_sq = (local - closest).dot(local - closest)
                            if distance_sq < best_distance_sq:
                                best_distance_sq = distance_sq
                                best_point = closest
                                best_normal = self.triangle_normals[triangle_index]
                                found = 1
                if found == 1:
                    signed_distance = (local - best_point).dot(best_normal)
                    if signed_distance < CLOTH_THICKNESS:
                        resolved = best_point + best_normal * CLOTH_THICKNESS + self.bunny_offset[None]
                        self.x[index] = resolved
                        if self.inv_mass[index] == 0.0:
                            # Move the persistent anchor itself outside the mesh;
                            # otherwise update_pins() would reinsert it every step.
                            self.pin_position[index] = resolved - self.bunny_offset[None]
                            self.previous[index] = resolved
                            self.velocity[index] = ti.Vector([0.0, 0.0, 0.0])
                        else:
                            normal_speed = self.velocity[index].dot(best_normal)
                            if normal_speed < 0.0:
                                self.velocity[index] -= best_normal * normal_speed
                            self.velocity[index] *= 0.997
                        ti.atomic_add(self.contact_count[None], 1)
                if self.inv_mass[index] > 0.0 and self.x[index].y < 0.015:
                    self.x[index].y = 0.015
                    if self.velocity[index].y < 0.0:
                        self.velocity[index].y *= -0.15

    @ti.kernel
    def collide_ground_only(self, count: ti.i32):
        for index in range(count):
            if self.inv_mass[index] > 0.0 and self.x[index].y < 0.015:
                self.x[index].y = 0.015
                if self.velocity[index].y < 0.0:
                    self.velocity[index].y *= -0.15

    @ti.kernel
    def update_velocity(self, count: ti.i32, dt: ti.f32):
        for index in range(count):
            if self.inv_mass[index] > 0.0:
                self.velocity[index] = (self.x[index] - self.previous[index]) / dt

    def collide(self) -> None:
        if self.collision_enabled:
            minimum = self.grid.minimum
            dims = self.grid.dims
            self.collide_with_mesh(
                self.particle_count,
                float(minimum[0]),
                float(minimum[1]),
                float(minimum[2]),
                self.grid.cell_size,
                int(dims[0]),
                int(dims[1]),
                int(dims[2]),
            )
        else:
            self.collide_ground_only(self.particle_count)

    def step(self, dt: float, elapsed: float) -> None:
        if self.paused:
            return
        self.contact_count[None] = 0
        # At least two physics substeps keep the swept displacement below the
        # local collision-cell scale. Havok-style uses one additional substep.
        substeps = 3 if self.algorithm == ALGORITHMS["havok"] else 2
        sub_dt = dt / substeps
        for _ in range(substeps):
            self.update_pins(self.particle_count)
            self.integrate(
                self.particle_count,
                sub_dt,
                self.gravity,
                self.wind if self.wind_enabled else 0.0,
                self.damping,
                elapsed,
            )
            if self.algorithm == ALGORITHMS["xpbd"]:
                self.reset_lagrange(self.constraint_count)
            iteration_count = max(2, int(self.iterations * 0.65)) if self.algorithm == ALGORITHMS["havok"] else self.iterations
            for _ in range(iteration_count):
                self.clear_corrections(self.particle_count)
                self.solve_constraints(
                    self.constraint_count, self.algorithm, self.stiffness, sub_dt
                )
                self.apply_corrections(self.particle_count)
                self.collide()
            self.update_velocity(self.particle_count, sub_dt)
            # Velocity reconstruction can reintroduce an inward component.
            # A final collision pass removes it after all constraints settle.
            self.collide()
        self.update_bunny_render_vertices()

    def stats(self) -> dict[str, object]:
        ti.sync()
        positions = self.x.to_numpy()[: self.particle_count]
        return {
            "particles": self.particle_count,
            "constraints": self.constraint_count,
            "contacts": int(self.contact_count[None]),
            "finite": bool(np.isfinite(positions).all()),
            "bounds_min": positions.min(axis=0).round(4).tolist(),
            "bounds_max": positions.max(axis=0).round(4).tolist(),
        }


def choose_arch(name: str):
    return {"cuda": ti.cuda, "vulkan": ti.vulkan, "cpu": ti.cpu}[name]


def run_headless(simulation: GpuClothSimulation, steps: int, dt: float) -> None:
    start = time.perf_counter()
    for frame in range(steps):
        simulation.step(dt, frame * dt)
    ti.sync()
    elapsed = time.perf_counter() - start
    output = simulation.stats()
    output.update(
        {
            "arch": str(ti.lang.impl.current_cfg().arch),
            "steps": steps,
            "elapsed_seconds": round(elapsed, 4),
            "simulated_fps": round(steps / max(elapsed, 1e-9), 2),
        }
    )
    print(json.dumps(output, ensure_ascii=False, indent=2))
    if not output["finite"]:
        raise SystemExit("Simulation produced non-finite particle positions")


def run_window(simulation: GpuClothSimulation, run_seconds: float = 0.0) -> None:
    window = ti.ui.Window("GPU Cloth Simulation Lab - Taichi CUDA", (1280, 760), vsync=True)
    canvas = window.get_canvas()
    canvas.set_background_color((0.035, 0.045, 0.065))
    scene = window.get_scene()
    camera = ti.ui.Camera()
    camera.position(0.0, 1.25, 4.1)
    camera.lookat(0.0, 0.75, 0.0)
    camera.fov(45)
    gui = window.get_gui()
    start = time.perf_counter()
    previous = start
    orbit_yaw = 0.0
    orbit_pitch = 0.12
    orbit_radius = 4.1
    orbit_speed = 0.35
    auto_orbit = False
    last_orbit_cursor: tuple[float, float] | None = None

    while window.running:
        now = time.perf_counter()
        frame_dt = min(now - previous, 1.0 / 30.0)
        previous = now

        cursor = window.get_cursor_pos()
        if window.is_pressed(ti.ui.RMB):
            if last_orbit_cursor is not None:
                delta_x = cursor[0] - last_orbit_cursor[0]
                delta_y = cursor[1] - last_orbit_cursor[1]
                orbit_yaw -= delta_x * math.tau * 1.2
                orbit_pitch = max(-1.25, min(1.25, orbit_pitch + delta_y * math.pi * 1.5))
                auto_orbit = False
            last_orbit_cursor = cursor
        else:
            last_orbit_cursor = None

        with gui.sub_window("GPU Cloth Controls", 0.015, 0.02, 0.27, 0.80):
            gui.text(f"Backend: {ti.lang.impl.current_cfg().arch} | particles: {simulation.particle_count}")
            gui.text(f"Constraints: {simulation.constraint_count} | contacts: {int(simulation.contact_count[None])}")
            if gui.button("PBD"):
                simulation.algorithm = ALGORITHMS["pbd"]
            if gui.button("XPBD"):
                simulation.algorithm = ALGORITHMS["xpbd"]
            if gui.button("Havok-style"):
                simulation.algorithm = ALGORITHMS["havok"]
            if gui.button("Cape"):
                simulation.set_cloth("cape")
            if gui.button("Long scarf"):
                simulation.set_cloth("scarf")
            if gui.button("Ear ornaments"):
                simulation.set_cloth("hair")
            simulation.gravity = gui.slider_float("Gravity", simulation.gravity, 0.0, 30.0)
            simulation.wind = gui.slider_float("Wind (0-100)", simulation.wind, 0.0, MAX_WIND)
            if gui.button("Strong wind (35)"):
                simulation.wind = 35.0
                simulation.wind_enabled = True
            if gui.button("Storm (70)"):
                simulation.wind = 70.0
                simulation.wind_enabled = True
            if gui.button("Extreme wind (100)"):
                simulation.wind = MAX_WIND
                simulation.wind_enabled = True
            simulation.stiffness = gui.slider_float("Stiffness", simulation.stiffness, 0.02, 1.0)
            simulation.iterations = gui.slider_int("Iterations", simulation.iterations, 1, 20)
            simulation.damping = gui.slider_float("Damping", simulation.damping, 0.0, 0.1)
            simulation.wind_enabled = gui.checkbox("Wind enabled", simulation.wind_enabled)
            simulation.collision_enabled = gui.checkbox("Bunny mesh collision", simulation.collision_enabled)
            simulation.wireframe = gui.checkbox("Wireframe", simulation.wireframe)
            simulation.paused = gui.checkbox("Paused", simulation.paused)
            if gui.button("Reset cloth"):
                simulation.set_cloth(simulation.cloth_type)
            if gui.button("Bunny hop"):
                simulation.bunny_offset_value = 0.18

        with gui.sub_window("Camera Controls", 0.72, 0.02, 0.265, 0.42):
            gui.text("RMB drag: orbit around Bunny")
            gui.text("Use Distance to zoom in / out")
            auto_orbit = gui.checkbox("Auto orbit", auto_orbit)
            orbit_speed = gui.slider_float("Orbit speed", orbit_speed, 0.05, 1.5)
            orbit_radius = gui.slider_float("Distance", orbit_radius, 2.0, 8.0)
            if gui.button("Front view"):
                orbit_yaw, orbit_pitch = 0.0, 0.12
                auto_orbit = False
            if gui.button("Back view"):
                orbit_yaw, orbit_pitch = math.pi, 0.12
                auto_orbit = False
            if gui.button("Left view"):
                orbit_yaw, orbit_pitch = -math.pi * 0.5, 0.12
                auto_orbit = False
            if gui.button("Right view"):
                orbit_yaw, orbit_pitch = math.pi * 0.5, 0.12
                auto_orbit = False
            if gui.button("Top view"):
                orbit_yaw, orbit_pitch = 0.0, 1.18
                auto_orbit = False

        if auto_orbit:
            orbit_yaw += frame_dt * orbit_speed
        target_y = 0.78
        horizontal_radius = orbit_radius * math.cos(orbit_pitch)
        camera.position(
            math.sin(orbit_yaw) * horizontal_radius,
            target_y + math.sin(orbit_pitch) * orbit_radius,
            math.cos(orbit_yaw) * horizontal_radius,
        )
        camera.lookat(0.0, target_y, 0.0)
        camera.up(0.0, 1.0, 0.0)

        elapsed = now - start
        if simulation.bunny_offset_value > 0.0:
            simulation.bunny_offset_value = max(0.0, simulation.bunny_offset_value - frame_dt * 0.7)
        hop = math.sin(simulation.bunny_offset_value / 0.18 * math.pi) * 0.14 if simulation.bunny_offset_value else 0.0
        simulation.bunny_offset[None] = (0.0, hop, 0.0)
        simulation.step(1.0 / 60.0, elapsed)

        scene.set_camera(camera)
        scene.ambient_light((0.48, 0.50, 0.56))
        scene.point_light(pos=(3.5, 5.5, 4.5), color=(1.0, 0.92, 0.82))
        scene.point_light(pos=(-3.0, 2.5, -2.0), color=(0.32, 0.42, 0.62))
        scene.mesh(
            simulation.bunny_render_vertices,
            simulation.bunny_faces,
            simulation.bunny_normals,
            color=(0.72, 0.50, 0.30),
        )
        if simulation.render_mode == "mesh":
            scene.mesh(
                simulation.x,
                simulation.render_indices,
                color=simulation.cloth_color,
                index_count=simulation.render_count,
                show_wireframe=simulation.wireframe,
            )
        else:
            scene.lines(
                simulation.x,
                width=4.0,
                indices=simulation.render_indices,
                color=simulation.cloth_color,
                index_count=simulation.render_count,
            )
        scene.particles(simulation.x, radius=0.006, color=(0.12, 0.92, 0.86), index_count=simulation.particle_count)
        canvas.scene(scene)
        window.show()
        if run_seconds > 0.0 and time.perf_counter() - start >= run_seconds:
            break


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Taichi CUDA cloth simulation demo")
    parser.add_argument("--arch", choices=("cuda", "vulkan", "cpu"), default="cuda")
    parser.add_argument("--cloth", choices=sorted(CLOTH_TYPES), default="cape")
    parser.add_argument("--algorithm", choices=sorted(ALGORITHMS), default="pbd")
    parser.add_argument("--headless", action="store_true", help="Run validation without opening a window")
    parser.add_argument("--no-collision", action="store_true", help="Disable Bunny mesh collision for diagnostics")
    parser.add_argument("--steps", type=int, default=120)
    parser.add_argument("--dt", type=float, default=1.0 / 60.0)
    parser.add_argument("--wind", type=float, default=3.0, help="Wind strength from 0 to 100")
    parser.add_argument("--run-seconds", type=float, default=0.0, help="Automatically close the window after N seconds")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ti.init(
        arch=choose_arch(args.arch),
        default_fp=ti.f32,
        device_memory_GB=1.0,
        kernel_profiler=False,
        offline_cache=True,
    )
    mesh = load_ascii_ply(ROOT / "bunny.ply")
    grid = CollisionGrid(mesh)
    simulation = GpuClothSimulation(mesh, grid)
    simulation.set_cloth(args.cloth)
    simulation.algorithm = ALGORITHMS[args.algorithm]
    simulation.wind = max(0.0, min(MAX_WIND, args.wind))
    simulation.collision_enabled = not args.no_collision
    if args.headless:
        run_headless(simulation, args.steps, args.dt)
    else:
        run_window(simulation, args.run_seconds)


if __name__ == "__main__":
    # Taichi 1.7.4's Windows/Python 3.12 runtime can fault during interpreter
    # teardown after all kernels have completed.  Bypass only that native
    # destructor path; runtime exceptions still print and return exit code 1.
    exit_code = 0
    try:
        main()
    except BaseException:  # noqa: BLE001 - command-line entry point boundary
        traceback.print_exc()
        exit_code = 1
    finally:
        sys.stdout.flush()
        sys.stderr.flush()
    os._exit(exit_code)
