/**
 * Cloth Simulation Lab - 布料模拟算法实验室
 *
 * 三大布料求解算法演示:
 *   1. PBD   - Position Based Dynamics (Muller 2007)
 *   2. XPBD  - Extended PBD with compliance (Macklin 2016)
 *   3. Havok-style - stabilized constraint projection + adaptive substep
 *
 * 模型: Stanford Bunny (1994, Greg Turk & Marc Levoy, 69,451 triangles)
 *       加载真实 .ply 扫描数据, 含围巾/耳饰/披风布料模拟
 */

import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
import { PLYLoader } from "three/addons/loaders/PLYLoader.js";

// ============================================================================
//  全局状态
// ============================================================================
const canvas = document.querySelector("#scene");

const state = {
  algorithm: "pbd",
  clothType: "cape",
  running: true,
  walking: false,
  walkTime: 0,

  // 物理参数
  gravity: 9.8,
  windStrength: 3.0,
  stiffness: 50,       // 1-100, 映射到各算法的刚度参数
  iterations: 6,
  damping: 0.02,

  // 开关
  windEnabled: true,
  collideEnabled: true,
  wireframe: false,
  autoRotate: true,

  // 统计
  fps: 0,
  frameMs: 16,
  particleCount: 0,
  constraintCount: 0,
};

// 接近地面的长款披风；横向分辨率保持适中，避免演示页约束数暴涨。
const CAPE_CONFIG = Object.freeze({ cols: 18, rows: 19, spacing: 0.045 });
const SCARF_CONFIG = Object.freeze({ segments: 20, rings: 21, ringHeight: 0.07 });
// 与 Python/Taichi GPU 版本保持一致的碰撞安全层和单子步位移上限。
const CLOTH_THICKNESS = 0.022;
const MAX_PARTICLE_STEP = 0.032;

// ============================================================================
//  Three.js 场景初始化
// ============================================================================
const renderer = new THREE.WebGLRenderer({ canvas, antialias: true, alpha: false });
renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
renderer.setSize(window.innerWidth, window.innerHeight);
renderer.shadowMap.enabled = true;
renderer.shadowMap.type = THREE.PCFSoftShadowMap;
renderer.toneMapping = THREE.ACESFilmicToneMapping;
renderer.toneMappingExposure = 1.5;

const scene = new THREE.Scene();
scene.background = new THREE.Color(0x1c2028);
scene.fog = new THREE.Fog(0x1c2028, 20, 55);

const camera = new THREE.PerspectiveCamera(45, window.innerWidth / window.innerHeight, 0.1, 100);
camera.position.set(0, 1.5, 4.5);

const controls = new OrbitControls(camera, canvas);
controls.enableDamping = true;
controls.dampingFactor = 0.08;
controls.minDistance = 3;
controls.maxDistance = 20;
controls.maxPolarAngle = Math.PI * 0.52;
controls.target.set(0, 0.8, 0);
controls.autoRotate = true;
controls.autoRotateSpeed = 0.6;

// 灯光 (调亮)
const ambient = new THREE.AmbientLight(0x90a0b0, 1.2);
scene.add(ambient);

const keyLight = new THREE.DirectionalLight(0xfff5e8, 2.5);
keyLight.position.set(5, 10, 6);
keyLight.castShadow = true;
keyLight.shadow.mapSize.set(2048, 2048);
keyLight.shadow.camera.left = -5;
keyLight.shadow.camera.right = 5;
keyLight.shadow.camera.top = 8;
keyLight.shadow.camera.bottom = -2;
keyLight.shadow.bias = -0.0005;
scene.add(keyLight);

const fillLight = new THREE.DirectionalLight(0x80a0ff, 1.0);
fillLight.position.set(-6, 4, -3);
scene.add(fillLight);

const rimLight = new THREE.DirectionalLight(0xffaa88, 0.9);
rimLight.position.set(0, 5, -8);
scene.add(rimLight);

// 地面
const groundGeo = new THREE.CircleGeometry(12, 64);
const groundMat = new THREE.MeshStandardMaterial({
  color: 0x2e333c, roughness: 0.85, metalness: 0.1,
});
const ground = new THREE.Mesh(groundGeo, groundMat);
ground.rotation.x = -Math.PI / 2;
ground.receiveShadow = true;
scene.add(ground);

// 地面网格线
const gridHelper = new THREE.GridHelper(20, 20, 0x3a3f4a, 0x2a2e36);
gridHelper.position.y = 0.01;
scene.add(gridHelper);

// ============================================================================
//  角色构建 - Stanford Bunny (真实扫描模型)
// ============================================================================
const characterGroup = new THREE.Group();
scene.add(characterGroup);

// 兔子材质 (陶土色, 还原原始雕像)
const bunnyMat = new THREE.MeshStandardMaterial({
  color: 0xc8a070, roughness: 0.75, metalness: 0.05, flatShading: false,
});

// Stanford Bunny 模型数据 (用于碰撞体和布料挂载点的参考)
const bunnyData = {
  loaded: false,
  mesh: null,
  collider: null,
  // 缩放后的大致身体参数 (用于布料挂载)
  scale: 8.0,
  centerY: 0,
  bodyY: 0.6,
  headY: 1.1,
  earTipY: 1.45,
  neckY: 0.85,
  backZ: -0.35,
  bodyRadius: 0.35,
};

/**
 * 静态三角网格 BVH。用于布料粒子到 Bunny 真实三角面的最近点和连续线段碰撞。
 * 模型只构建一次；查询复杂度由遍历全部三角面降为近似 O(log n)。
 */
class TriangleMeshCollider {
  constructor(geometry, leafSize = 12) {
    const position = geometry.getAttribute("position");
    const index = geometry.index;
    const triangleCount = index ? index.count / 3 : position.count / 3;
    this.triangles = new Array(triangleCount);
    this._candidate = new THREE.Vector3();
    this._segmentDirection = new THREE.Vector3();
    this._edgeCross = new THREE.Vector3();
    this._originDelta = new THREE.Vector3();
    this._rayCross = new THREE.Vector3();
    this._rayQ = new THREE.Vector3();

    const readVertex = (vertexIndex) => new THREE.Vector3().fromBufferAttribute(position, vertexIndex);
    for (let triangleIndex = 0; triangleIndex < triangleCount; triangleIndex++) {
      const offset = triangleIndex * 3;
      const ia = index ? index.getX(offset) : offset;
      const ib = index ? index.getX(offset + 1) : offset + 1;
      const ic = index ? index.getX(offset + 2) : offset + 2;
      const shape = new THREE.Triangle(readVertex(ia), readVertex(ib), readVertex(ic));
      const normal = shape.getNormal(new THREE.Vector3());
      const bounds = new THREE.Box3().setFromPoints([shape.a, shape.b, shape.c]);
      const centroid = shape.getMidpoint(new THREE.Vector3());
      this.triangles[triangleIndex] = { shape, normal, bounds, centroid };
    }

    this.root = this._buildNode(Array.from({ length: triangleCount }, (_, indexValue) => indexValue), leafSize);
  }

  _buildNode(indices, leafSize) {
    const bounds = new THREE.Box3();
    const centroidBounds = new THREE.Box3();
    for (const triangleIndex of indices) {
      const triangle = this.triangles[triangleIndex];
      bounds.union(triangle.bounds);
      centroidBounds.expandByPoint(triangle.centroid);
    }
    if (indices.length <= leafSize) return { bounds, indices };

    const extent = centroidBounds.getSize(new THREE.Vector3());
    const axis = extent.x >= extent.y && extent.x >= extent.z ? "x" : extent.y >= extent.z ? "y" : "z";
    indices.sort((left, right) => this.triangles[left].centroid[axis] - this.triangles[right].centroid[axis]);
    const middle = Math.floor(indices.length / 2);
    return {
      bounds,
      left: this._buildNode(indices.slice(0, middle), leafSize),
      right: this._buildNode(indices.slice(middle), leafSize),
    };
  }

  closestPoint(point, maxDistance = Infinity) {
    let bestDistanceSq = maxDistance * maxDistance;
    let bestTriangle = null;
    const bestPoint = new THREE.Vector3();
    const visit = (node) => {
      if (node.bounds.distanceToPoint(point) ** 2 > bestDistanceSq) return;
      if (node.indices) {
        for (const triangleIndex of node.indices) {
          const triangle = this.triangles[triangleIndex];
          triangle.shape.closestPointToPoint(point, this._candidate);
          const distanceSq = this._candidate.distanceToSquared(point);
          if (distanceSq < bestDistanceSq) {
            bestDistanceSq = distanceSq;
            bestTriangle = triangle;
            bestPoint.copy(this._candidate);
          }
        }
        return;
      }
      const leftDistance = node.left.bounds.distanceToPoint(point);
      const rightDistance = node.right.bounds.distanceToPoint(point);
      if (leftDistance <= rightDistance) {
        visit(node.left);
        visit(node.right);
      } else {
        visit(node.right);
        visit(node.left);
      }
    };
    visit(this.root);
    return bestTriangle ? { point: bestPoint, normal: bestTriangle.normal, distance: Math.sqrt(bestDistanceSq) } : null;
  }

  segmentCast(start, end) {
    const direction = this._segmentDirection.subVectors(end, start);
    let bestT = 1 + 1e-8;
    let bestTriangle = null;
    const bestPoint = new THREE.Vector3();
    const intersectsBounds = (box) => {
      let minT = 0;
      let maxT = bestT;
      for (const axis of ["x", "y", "z"]) {
        const component = direction[axis];
        if (Math.abs(component) < 1e-10) {
          if (start[axis] < box.min[axis] || start[axis] > box.max[axis]) return false;
          continue;
        }
        const inverse = 1 / component;
        let near = (box.min[axis] - start[axis]) * inverse;
        let far = (box.max[axis] - start[axis]) * inverse;
        if (near > far) [near, far] = [far, near];
        minT = Math.max(minT, near);
        maxT = Math.min(maxT, far);
        if (minT > maxT) return false;
      }
      return maxT >= 0 && minT <= bestT;
    };
    const visit = (node) => {
      if (!intersectsBounds(node.bounds)) return;
      if (node.indices) {
        for (const triangleIndex of node.indices) {
          const triangle = this.triangles[triangleIndex];
          // 只阻挡从模型外侧进入表面的运动，允许已在内部的粒子退出。
          if (direction.dot(triangle.normal) >= -1e-9) continue;
          const edge1 = this._candidate.subVectors(triangle.shape.b, triangle.shape.a);
          const edge2 = this._edgeCross.subVectors(triangle.shape.c, triangle.shape.a);
          const h = this._rayCross.crossVectors(direction, edge2);
          const determinant = edge1.dot(h);
          if (Math.abs(determinant) < 1e-10) continue;
          const inverseDeterminant = 1 / determinant;
          const s = this._originDelta.subVectors(start, triangle.shape.a);
          const u = inverseDeterminant * s.dot(h);
          if (u < 0 || u > 1) continue;
          const q = this._rayQ.crossVectors(s, edge1);
          const v = inverseDeterminant * direction.dot(q);
          if (v < 0 || u + v > 1) continue;
          const t = inverseDeterminant * edge2.dot(q);
          if (t >= 0 && t < bestT && t <= 1) {
            bestT = t;
            bestTriangle = triangle;
            bestPoint.copy(direction).multiplyScalar(t).add(start);
          }
        }
        return;
      }
      visit(node.left);
      visit(node.right);
    };
    visit(this.root);
    return bestTriangle ? { point: bestPoint, normal: bestTriangle.normal, t: bestT } : null;
  }
}

function buildCharacter() {
  const loader = new PLYLoader();
  loader.load("./bunny.ply", (geometry) => {
    // 计算包围盒以确定缩放和居中
    geometry.computeBoundingBox();
    const box = geometry.boundingBox;
    const size = new THREE.Vector3();
    box.getSize(size);
    const center = new THREE.Vector3();
    box.getCenter(center);

    // 缩放到目标高度 (~1.6 单位)
    const targetHeight = 1.6;
    bunnyData.scale = targetHeight / size.y;

    // 居中并放到地面上
    geometry.translate(-center.x, -box.min.y, -center.z);
    geometry.scale(bunnyData.scale, bunnyData.scale, bunnyData.scale);
    geometry.computeVertexNormals();

    // 更新参考参数
    const scaledSize = size.clone().multiplyScalar(bunnyData.scale);
    bunnyData.centerY = scaledSize.y / 2;
    bunnyData.bodyY = scaledSize.y * 0.38;
    bunnyData.headY = scaledSize.y * 0.72;
    bunnyData.earTipY = scaledSize.y * 0.92;
    bunnyData.neckY = scaledSize.y * 0.55;
    bunnyData.backZ = -scaledSize.z * 0.35;
    bunnyData.bodyRadius = scaledSize.x * 0.4;

    const mesh = new THREE.Mesh(geometry, bunnyMat);
    mesh.castShadow = true;
    mesh.receiveShadow = true;
    characterGroup.add(mesh);
    bunnyData.mesh = mesh;
    bunnyData.collider = new TriangleMeshCollider(geometry);
    bunnyData.loaded = true;

    // 隐藏 loading 提示
    const loadingEl = document.getElementById("bunnyLoading");
    if (loadingEl) loadingEl.style.display = "none";

    // 重建布料以匹配新身体参数
    rebuildCloth();
  }, undefined, (err) => {
    console.error("Failed to load bunny.ply:", err);
    const loadingEl = document.getElementById("bunnyLoading");
    if (loadingEl) loadingEl.textContent = "模型加载失败, 请确保 bunny.ply 存在";
  });
}

buildCharacter();

// ============================================================================
//  布料系统核心
// ============================================================================

/**
 * ClothParticle - 布料粒子
 */
class ClothParticle {
  constructor(x, y, z, mass = 1.0, pinned = false) {
    this.position = new THREE.Vector3(x, y, z);
    this.previous = new THREE.Vector3(x, y, z);
    this.velocity = new THREE.Vector3(0, 0, 0);
    this.acceleration = new THREE.Vector3(0, 0, 0);
    this.force = new THREE.Vector3(0, 0, 0);
    this.mass = mass;
    this.invMass = pinned ? 0 : 1 / mass;
    this.pinned = pinned;
    this.pinLocalPosition = pinned ? this.position.clone() : null;
  }
}

/**
 * ClothConstraint - 距离约束 (PBD/XPBD 通用)
 */
class DistanceConstraint {
  constructor(p1, p2, restLength, type = "distance") {
    this.p1 = p1;
    this.p2 = p2;
    this.restLength = restLength;
    this.type = type; // "distance" | "bending"
    // XPBD 状态
    this.lambda = 0; // 累积 Lagrange 乘子
  }
}

/**
 * ClothMesh - 布料网格, 管理粒子和约束, 提供三种求解器
 */
class ClothMesh {
  constructor() {
    this.particles = [];
    this.constraints = [];
    this.geometry = null;
    this.mesh = null;
    this.material = null;
    this.collisionSpheres = []; // 碰撞体球列表
    // 高频求解路径复用临时向量，避免每帧创建数千个对象。
    this._delta = new THREE.Vector3();
    this._windDirection = new THREE.Vector3();
    this._collisionDelta = new THREE.Vector3();
    this._positionDelta = new THREE.Vector3();
    this._relativeVelocity = new THREE.Vector3();
    this._localPosition = new THREE.Vector3();
    this._localPrevious = new THREE.Vector3();
    this._localVelocity = new THREE.Vector3();
    this._worldNormal = new THREE.Vector3();
    this._worldToLocal = new THREE.Matrix4();
    this._normalMatrix = new THREE.Matrix3();
  }

  /**
   * 创建围巾布料 (兔子脖颈处)
   */
  createScarf() {
    this.clear();
    const neckY = bunnyData.neckY || 0.85;
    const neckRadius = (bunnyData.bodyRadius || 0.35) * 0.8;
    const hemRadius = (bunnyData.bodyRadius || 0.35) * 1.1;
    const { segments, rings, ringHeight } = SCARF_CONFIG;

    for (let r = 0; r < rings; r++) {
      const t = r / (rings - 1);
      const radius = neckRadius + (hemRadius - neckRadius) * t;
      const y = neckY - r * ringHeight;
      for (let s = 0; s < segments; s++) {
        const angle = (s / segments) * Math.PI * 2;
        const x = Math.cos(angle) * radius;
        const z = Math.sin(angle) * radius;
        const pinned = r === 0;
        this.particles.push(new ClothParticle(x, y, z, 1.0, pinned));
      }
    }

    this.conformInitialParticlesToMesh((bunnyData.bodyRadius || 0.35) * 0.35);

    this.buildConstraints(segments, rings);
    this.buildGeometry(segments, rings, 0x6688cc);
    this.setupCollision();
  }

  /**
   * 创建耳饰布条 (从兔子耳朵垂下)
   */
  createHair() {
    this.clear();
    const earTipY = bunnyData.earTipY || 1.45;
    const segments = 14;
    const strands = 8;
    const strandLength = 0.055;

    for (let strand = 0; strand < strands; strand++) {
      const earOffset = 0.08;
      const earSide = strand < strands / 2 ? -earOffset : earOffset;
      const localStrand = strand < strands / 2 ? strand : strand - strands / 2;
      const localAngle = (localStrand / (strands / 2)) * Math.PI * 2;
      const baseRadius = 0.03;
      const baseX = earSide + Math.cos(localAngle) * baseRadius;
      const baseZ = 0.04 + Math.sin(localAngle) * baseRadius;
      const baseY = earTipY;

      for (let s = 0; s < segments; s++) {
        const x = baseX;
        const y = baseY - s * strandLength;
        const z = baseZ;
        const pinned = s === 0;
        this.particles.push(new ClothParticle(x, y, z, 0.5, pinned));
      }
    }

    this.buildHairConstraints(segments, strands);
    this.buildHairGeometry(segments, strands);
    this.setupCollision();
  }

  /**
   * 创建披风布料 (兔子背部)
   */
  createCape() {
    this.clear();
    const { cols, rows, spacing } = CAPE_CONFIG;
    // 披风从头部后方中央扣合，顶部两侧自由下垂。
    const pinColumns = new Set([6, 8, 9, 11]);
    const startY = (bunnyData.headY || 1.1) + 0.02;
    const startX = -((cols - 1) * spacing) / 2;
    const startZ = (bunnyData.backZ || -0.35) - (bunnyData.bodyRadius || 0.35) * 0.08;

    for (let r = 0; r < rows; r++) {
      for (let c = 0; c < cols; c++) {
        const x = startX + c * spacing;
        const y = startY - r * spacing;
        const z = startZ;
        const pinned = r === 0 && pinColumns.has(c);
        this.particles.push(new ClothParticle(x, y, z, 1.0, pinned));
      }
    }

    this.conformInitialParticlesToMesh(
      (bunnyData.bodyRadius || 0.35) * 0.45,
      normal => normal.z < -0.1
    );

    this.buildGridConstraints(cols, rows);
    this.buildGridGeometry(cols, rows, 0xcc4466);
    this.setupCollision();
  }

  clear() {
    this.particles = [];
    this.constraints = [];
    this.characteristicLength = null;
    if (this.mesh) {
      scene.remove(this.mesh);
      this.geometry?.dispose();
      this.material?.dispose();
    }
    if (this._hairTubeGroup) {
      scene.remove(this._hairTubeGroup);
      for (const child of this._hairTubeGroup.children) child.geometry?.dispose();
      this._hairTubeMat?.dispose();
      this._hairTubeGroup = null;
      this._hairTubeMat = null;
    }
    this.geometry = null;
    this.mesh = null;
    this.material = null;
    this.collisionSpheres = [];
  }

  /**
   * 构建距离约束 + 弯曲约束 (环形布料: 裙子)
   */
  buildConstraints(segments, rings) {
    const idx = (r, s) => r * segments + s;
    for (let r = 0; r < rings; r++) {
      for (let s = 0; s < segments; s++) {
        const sNext = (s + 1) % segments;
        // 环向距离约束
        if (s < segments) {
          const p1 = this.particles[idx(r, s)];
          const p2 = this.particles[idx(r, sNext)];
          this.constraints.push(new DistanceConstraint(p1, p2, p1.position.distanceTo(p2.position), "distance"));
        }
        // 径向距离约束
        if (r < rings - 1) {
          const p1 = this.particles[idx(r, s)];
          const p2 = this.particles[idx(r + 1, s)];
          this.constraints.push(new DistanceConstraint(p1, p2, p1.position.distanceTo(p2.position), "distance"));
        }
        // 弯曲约束 (隔一行)
        if (r < rings - 2) {
          const p1 = this.particles[idx(r, s)];
          const p2 = this.particles[idx(r + 2, s)];
          this.constraints.push(new DistanceConstraint(p1, p2, p1.position.distanceTo(p2.position), "bending"));
        }
      }
    }
  }

  /**
   * 构建头发约束 (链式)
   */
  buildHairConstraints(segments, strands) {
    for (let strand = 0; strand < strands; strand++) {
      const base = strand * segments;
      for (let s = 0; s < segments - 1; s++) {
        const p1 = this.particles[base + s];
        const p2 = this.particles[base + s + 1];
        this.constraints.push(new DistanceConstraint(p1, p2, p1.position.distanceTo(p2.position), "distance"));
      }
      // 弯曲约束
      for (let s = 0; s < segments - 2; s++) {
        const p1 = this.particles[base + s];
        const p2 = this.particles[base + s + 2];
        this.constraints.push(new DistanceConstraint(p1, p2, p1.position.distanceTo(p2.position), "bending"));
      }
    }
  }

  /**
   * 构建网格约束 (披风)
   */
  buildGridConstraints(cols, rows) {
    const idx = (r, c) => r * cols + c;
    for (let r = 0; r < rows; r++) {
      for (let c = 0; c < cols; c++) {
        if (c < cols - 1) {
          const p1 = this.particles[idx(r, c)];
          const p2 = this.particles[idx(r, c + 1)];
          this.constraints.push(new DistanceConstraint(p1, p2, p1.position.distanceTo(p2.position), "distance"));
        }
        if (r < rows - 1) {
          const p1 = this.particles[idx(r, c)];
          const p2 = this.particles[idx(r + 1, c)];
          this.constraints.push(new DistanceConstraint(p1, p2, p1.position.distanceTo(p2.position), "distance"));
        }
        // 对角线剪切约束
        if (r < rows - 1 && c < cols - 1) {
          const p1 = this.particles[idx(r, c)];
          const p2 = this.particles[idx(r + 1, c + 1)];
          this.constraints.push(new DistanceConstraint(p1, p2, p1.position.distanceTo(p2.position), "distance"));
        }
        // 双向剪切约束，避免网格只沿一个对角方向产生偏斜。
        if (r < rows - 1 && c > 0) {
          const p1 = this.particles[idx(r, c)];
          const p2 = this.particles[idx(r + 1, c - 1)];
          this.constraints.push(new DistanceConstraint(p1, p2, p1.position.distanceTo(p2.position), "distance"));
        }
        // 弯曲约束
        if (c < cols - 2) {
          const p1 = this.particles[idx(r, c)];
          const p2 = this.particles[idx(r, c + 2)];
          this.constraints.push(new DistanceConstraint(p1, p2, p1.position.distanceTo(p2.position), "bending"));
        }
        if (r < rows - 2) {
          const p1 = this.particles[idx(r, c)];
          const p2 = this.particles[idx(r + 2, c)];
          this.constraints.push(new DistanceConstraint(p1, p2, p1.position.distanceTo(p2.position), "bending"));
        }
      }
    }
  }

  /**
   * 构建渲染几何体 (环形)
   */
  buildGeometry(segments, rings, color) {
    const vertexCount = this.particles.length;
    const positions = new Float32Array(vertexCount * 3);
    const normals = new Float32Array(vertexCount * 3);
    const uvs = new Float32Array(vertexCount * 2);

    for (let i = 0; i < vertexCount; i++) {
      positions[i * 3] = this.particles[i].position.x;
      positions[i * 3 + 1] = this.particles[i].position.y;
      positions[i * 3 + 2] = this.particles[i].position.z;
      uvs[i * 2] = (i % segments) / segments;
      uvs[i * 2 + 1] = Math.floor(i / segments) / rings;
    }

    const indices = [];
    for (let r = 0; r < rings - 1; r++) {
      for (let s = 0; s < segments; s++) {
        const sNext = (s + 1) % segments;
        const a = r * segments + s;
        const b = r * segments + sNext;
        const c = (r + 1) * segments + s;
        const d = (r + 1) * segments + sNext;
        indices.push(a, c, b);
        indices.push(b, c, d);
      }
    }

    this.geometry = new THREE.BufferGeometry();
    this.geometry.setAttribute("position", new THREE.BufferAttribute(positions, 3));
    this.geometry.setAttribute("normal", new THREE.BufferAttribute(normals, 3));
    this.geometry.setAttribute("uv", new THREE.BufferAttribute(uvs, 2));
    this.geometry.setIndex(indices);
    this.geometry.computeVertexNormals();

    this.material = new THREE.MeshStandardMaterial({
      color: color,
      roughness: 0.65,
      metalness: 0.05,
      side: THREE.DoubleSide,
      wireframe: state.wireframe,
    });

    this.mesh = new THREE.Mesh(this.geometry, this.material);
    this.mesh.castShadow = true;
    this.mesh.receiveShadow = true;
    scene.add(this.mesh);
  }

  /**
   * 构建头发渲染几何体
   */
  buildHairGeometry(segments, strands) {
    const vertexCount = this.particles.length;
    const positions = new Float32Array(vertexCount * 3);
    const normals = new Float32Array(vertexCount * 3);
    const uvs = new Float32Array(vertexCount * 2);

    for (let i = 0; i < vertexCount; i++) {
      positions[i * 3] = this.particles[i].position.x;
      positions[i * 3 + 1] = this.particles[i].position.y;
      positions[i * 3 + 2] = this.particles[i].position.z;
    }

    // 用 LineSegments 渲染头发
    const indices = [];
    for (let strand = 0; strand < strands; strand++) {
      const base = strand * segments;
      for (let s = 0; s < segments - 1; s++) {
        indices.push(base + s, base + s + 1);
      }
    }

    this.geometry = new THREE.BufferGeometry();
    this.geometry.setAttribute("position", new THREE.BufferAttribute(positions, 3));

    this.material = new THREE.LineBasicMaterial({
      color: 0xcc6688,
      linewidth: 2,
    });

    this.mesh = new THREE.LineSegments(this.geometry, this.material);
    scene.add(this.mesh);

    // 同时创建一个稍粗的管状网格作为头发视觉
    this._buildHairTubeMesh(segments, strands);
  }

  _buildHairTubeMesh(segments, strands) {
    // 为每束头发创建带粗度的 mesh, 同时设置 userData 索引
    const tubeGroup = new THREE.Group();
    const tubeMat = new THREE.MeshStandardMaterial({
      color: 0xcc6688, roughness: 0.75, metalness: 0.1, side: THREE.DoubleSide,
    });

    for (let strand = 0; strand < strands; strand++) {
      const base = strand * segments;
      for (let s = 0; s < segments - 1; s++) {
        const p1 = this.particles[base + s].position;
        const p2 = this.particles[base + s + 1].position;
        const dir = this._delta.subVectors(p2, p1);
        const len = Math.max(0.001, dir.length());

        const cylGeo = new THREE.CylinderGeometry(0.012, 0.008, len, 5);
        const cyl = new THREE.Mesh(cylGeo, tubeMat);
        cyl.position.copy(p1).add(p2).multiplyScalar(0.5);
        cyl.lookAt(p2);
        cyl.rotateX(Math.PI / 2);
        cyl.castShadow = true;
        // 直接在创建时设置粒子索引, 避免 rebuildHairTubeUserData 对不上
        cyl.userData.s0 = base + s;
        cyl.userData.s1 = base + s + 1;
        cyl.userData.origLen = len;
        tubeGroup.add(cyl);
      }
    }

    tubeGroup.name = "hairTubes";
    scene.add(tubeGroup);
    this._hairTubeGroup = tubeGroup;
    this._hairTubeMat = tubeMat;
  }

  /**
   * 构建网格渲染几何体 (披风)
   */
  buildGridGeometry(cols, rows, color) {
    const vertexCount = this.particles.length;
    const positions = new Float32Array(vertexCount * 3);
    const normals = new Float32Array(vertexCount * 3);
    const uvs = new Float32Array(vertexCount * 2);

    for (let i = 0; i < vertexCount; i++) {
      positions[i * 3] = this.particles[i].position.x;
      positions[i * 3 + 1] = this.particles[i].position.y;
      positions[i * 3 + 2] = this.particles[i].position.z;
      uvs[i * 2] = (i % cols) / (cols - 1);
      uvs[i * 2 + 1] = Math.floor(i / cols) / (rows - 1);
    }

    const indices = [];
    for (let r = 0; r < rows - 1; r++) {
      for (let c = 0; c < cols - 1; c++) {
        const a = r * cols + c;
        const b = r * cols + c + 1;
        const cI = (r + 1) * cols + c;
        const d = (r + 1) * cols + c + 1;
        indices.push(a, cI, b);
        indices.push(b, cI, d);
      }
    }

    this.geometry = new THREE.BufferGeometry();
    this.geometry.setAttribute("position", new THREE.BufferAttribute(positions, 3));
    this.geometry.setAttribute("normal", new THREE.BufferAttribute(normals, 3));
    this.geometry.setAttribute("uv", new THREE.BufferAttribute(uvs, 2));
    this.geometry.setIndex(indices);
    this.geometry.computeVertexNormals();

    this.material = new THREE.MeshStandardMaterial({
      color: color,
      roughness: 0.6,
      metalness: 0.08,
      side: THREE.DoubleSide,
      wireframe: state.wireframe,
    });

    this.mesh = new THREE.Mesh(this.geometry, this.material);
    this.mesh.castShadow = true;
    this.mesh.receiveShadow = true;
    scene.add(this.mesh);
  }

  /**
   * 设置碰撞体 (兔子身体球体近似)
   */
  setupCollision() {
    const by = bunnyData.bodyY || 0.6;
    const hy = bunnyData.headY || 1.1;
    const br = bunnyData.bodyRadius || 0.35;
    const bz = bunnyData.backZ || -0.35;

    const structuralLengths = this.constraints
      .filter(constraint => constraint.type === "distance")
      .map(constraint => constraint.restLength);
    this.characteristicLength = structuralLengths.length ? Math.min(...structuralLengths) : 0.04;

    // 身体前部
    this._addCollisionSphere(0, by, br * 0.5, br);
    // 身体后部
    this._addCollisionSphere(0, by, bz * 0.5, br * 1.05);
    // 头部
    this._addCollisionSphere(0, hy, br * 0.3, br * 0.7);
    // 后腿/臀 (左)
    this._addCollisionSphere(-br * 0.6, by * 0.5, bz * 0.7, br * 0.55);
    // 后腿/臀 (右)
    this._addCollisionSphere(br * 0.6, by * 0.5, bz * 0.7, br * 0.55);
  }

  _addCollisionSphere(x, y, z, radius) {
    const localCenter = new THREE.Vector3(x, y, z);
    this.collisionSpheres.push({ center: localCenter.clone(), localCenter, radius });
  }

  conformInitialParticlesToMesh(maxDistance, normalFilter = () => true) {
    if (!bunnyData.collider) return;
    for (const particle of this.particles) {
      const surface = bunnyData.collider.closestPoint(particle.position, maxDistance);
      if (!surface || !normalFilter(surface.normal)) continue;
      particle.position.copy(surface.point).addScaledVector(surface.normal, CLOTH_THICKNESS);
      particle.previous.copy(particle.position);
      if (particle.pinned) particle.pinLocalPosition.copy(particle.position);
    }
  }

  updatePinnedPositions(dt) {
    characterGroup.updateMatrixWorld(true);
    const safeDt = Math.max(dt, 1e-6);
    for (const p of this.particles) {
      if (!p.pinned || !p.pinLocalPosition) continue;
      p.previous.copy(p.position);
      p.position.copy(p.pinLocalPosition).applyMatrix4(characterGroup.matrixWorld);
      p.velocity.subVectors(p.position, p.previous).divideScalar(safeDt);
    }
  }

  advanceParticle(particle, dt) {
    const speed = particle.velocity.length();
    const maxSpeed = MAX_PARTICLE_STEP / Math.max(dt, 1e-6);
    if (speed > maxSpeed) particle.velocity.multiplyScalar(maxSpeed / speed);
    particle.position.addScaledVector(particle.velocity, dt);
  }

  /**
   * 同步渲染几何体
   */
  syncGeometry() {
    if (!this.geometry) return;
    const positions = this.geometry.attributes.position.array;
    for (let i = 0; i < this.particles.length; i++) {
      positions[i * 3] = this.particles[i].position.x;
      positions[i * 3 + 1] = this.particles[i].position.y;
      positions[i * 3 + 2] = this.particles[i].position.z;
    }
    this.geometry.attributes.position.needsUpdate = true;
    this.geometry.computeVertexNormals();

    // 头发管状网格同步
    if (this._hairTubeGroup) {
      // 简单方式: 重建 (粒子少, 性能可接受)
      const strands = this._hairTubeGroup.children.length;
      // 直接更新位置
      let tubeIdx = 0;
      const segments = this.particles.length;
      // 通过粒子位置更新管状网格
      for (let i = 0; i < this._hairTubeGroup.children.length; i++) {
        const cyl = this._hairTubeGroup.children[i];
        // 从 userData 获取粒子索引
        const s0 = cyl.userData.s0;
        const s1 = cyl.userData.s1;
        if (s0 !== undefined && s1 !== undefined && s0 < this.particles.length && s1 < this.particles.length) {
          const p1 = this.particles[s0].position;
          const p2 = this.particles[s1].position;
          cyl.position.copy(p1).add(p2).multiplyScalar(0.5);
          cyl.lookAt(p2);
          cyl.rotateX(Math.PI / 2);
          const dir = this._delta.subVectors(p2, p1);
          const len = Math.max(0.001, dir.length());
          cyl.scale.set(1, len / cyl.userData.origLen, 1);
        }
      }
    }
  }

  /**
   * 重建头发管状网格索引
   */
  rebuildHairTubeUserData(segments, strands) {
    if (!this._hairTubeGroup) return;
    let tubeIdx = 0;
    for (let strand = 0; strand < strands; strand++) {
      const base = strand * segments;
      for (let s = 0; s < segments - 1; s++) {
        if (tubeIdx < this._hairTubeGroup.children.length) {
          const cyl = this._hairTubeGroup.children[tubeIdx];
          cyl.userData.s0 = base + s;
          cyl.userData.s1 = base + s + 1;
          cyl.userData.origLen = cyl.geometry.parameters.height;
          tubeIdx++;
        }
      }
    }
  }

  // ========================================================================
  //  求解器 1: PBD (Position Based Dynamics)
  //  Muller, Heidelberger, Hennix, Ratcliff (2007)
  // ========================================================================
  solvePBD(dt) {
    const gravity = state.gravity;
    const windStr = state.windEnabled ? state.windStrength : 0;
    const damping = state.damping;
    const stiffness = state.stiffness / 100; // 0-1
    const iterations = state.iterations;
    const dampingFactor = Math.exp(-damping * dt * 60);

    // Step 1: 施加外力 + 预测位置
    const elapsed = performance.now() * 0.001;
    const windDir = this._windDirection.set(
      0.45 + Math.sin(elapsed * 0.7) * 0.35,
      0.05,
      0.65
    ).normalize();

    for (const p of this.particles) {
      if (p.pinned) continue;
      // 保存上一帧位置
      p.previous.copy(p.position);
      // 施加重力
      p.velocity.y -= gravity * dt;
      // 施加风力
      if (windStr > 0) {
        p.velocity.addScaledVector(windDir, windStr * dt * 0.5);
      }
      // 阻尼
      p.velocity.multiplyScalar(dampingFactor);
      // 半隐式欧拉预测位置: p' = p + v*dt
      this.advanceParticle(p, dt);
    }

    // Step 2: 迭代约束投影
    for (let iter = 0; iter < iterations; iter++) {
      for (const c of this.constraints) {
        const delta = this._delta.subVectors(c.p2.position, c.p1.position);
        const dist = delta.length();
        if (dist < 1e-8) continue;

        const diff = (dist - c.restLength) / dist;

        // 弯曲约束刚度降低
        const k = c.type === "bending" ? stiffness * 0.3 : stiffness;

        const w1 = c.p1.invMass;
        const w2 = c.p2.invMass;
        const wSum = w1 + w2;
        if (wSum < 1e-8) continue;

        // PBD 位置修正: p1 沿 delta 方向移动, p2 反向移动
        const corr1 = (k * diff * w1) / wSum;
        const corr2 = (k * diff * w2) / wSum;
        if (!c.p1.pinned) c.p1.position.addScaledVector(delta, corr1);
        if (!c.p2.pinned) c.p2.position.addScaledVector(delta, -corr2);
      }

      // 碰撞处理 (每次迭代后)
      if (state.collideEnabled) {
        this.resolveCollisions();
      }
    }

    // Step 3: 从位置变化更新速度
    for (const p of this.particles) {
      if (p.pinned) continue;
      p.velocity.subVectors(p.position, p.previous).divideScalar(dt);
    }

    // 速度重建后再做一次碰撞，避免约束末次投影重新压入模型。
    if (state.collideEnabled) this.resolveCollisions();

    // 地面碰撞
    for (const p of this.particles) {
      if (p.position.y < 0.02) {
        p.position.y = 0.02;
        p.velocity.y *= -0.3;
      }
    }
  }

  // ========================================================================
  //  求解器 2: XPBD (Extended Position Based Dynamics)
  //  Macklin, Muller, Chentanez (2016)
  // ========================================================================
  solveXPBD(dt) {
    const gravity = state.gravity;
    const windStr = state.windEnabled ? state.windStrength : 0;
    const damping = state.damping;
    const iterations = state.iterations;
    const dampingFactor = Math.exp(-damping * dt * 60);

    // compliance (柔量): 越大越软, 与迭代次数解耦
    // stiffness 1-100 -> compliance 约 1e-5 ~ 5e-4
    const complianceBase = 1 / (state.stiffness / 100 + 0.01);
    const alphaDistance = complianceBase * 1e-5;
    const alphaBending = alphaDistance * 3;

    const elapsed = performance.now() * 0.001;
    const windDir = this._windDirection.set(
      0.45 + Math.sin(elapsed * 0.7) * 0.35,
      0.05,
      0.65
    ).normalize();

    // Step 1: 预测位置 + 重置 lambda
    for (const p of this.particles) {
      if (p.pinned) continue;
      p.previous.copy(p.position);
      p.velocity.y -= gravity * dt;
      if (windStr > 0) {
        p.velocity.addScaledVector(windDir, windStr * dt * 0.5);
      }
      p.velocity.multiplyScalar(dampingFactor);
      this.advanceParticle(p, dt);
    }
    for (const c of this.constraints) {
      c.lambda = 0;
    }

    // Step 2: XPBD 迭代求解
    const invDt = 1 / dt;
    const invDt2 = invDt * invDt;

    for (let iter = 0; iter < iterations; iter++) {
      for (const c of this.constraints) {
        const delta = this._delta.subVectors(c.p2.position, c.p1.position);
        const dist = delta.length();
        if (dist < 1e-8) continue;

        const w1 = c.p1.invMass;
        const w2 = c.p2.invMass;
        const wSum = w1 + w2;
        if (wSum < 1e-8) continue;

        // XPBD 核心公式:
        //   dlambda = (-C - alpha * lambda * inv_dt^2) / (wSum + alpha * inv_dt^2)
        //   C = dist - restLength
        const C = dist - c.restLength;
        const alpha = c.type === "bending" ? alphaBending : alphaDistance;
        const alphaTilde = alpha * invDt2;

        const dlambda = (-C - alphaTilde * c.lambda) / (wSum + alphaTilde);

        // XPBD 位置修正: 沿约束梯度方向 (delta/dist)
        const correctionScale = dlambda / dist;
        if (!c.p1.pinned) {
          c.p1.position.addScaledVector(delta, -w1 * correctionScale);
        }
        if (!c.p2.pinned) {
          c.p2.position.addScaledVector(delta, w2 * correctionScale);
        }

        // 累积 Lagrange 乘子 (XPBD 关键: 跨迭代保持)
        c.lambda += dlambda;
      }

      if (state.collideEnabled) {
        this.resolveCollisions();
      }
    }

    // Step 3: 更新速度
    for (const p of this.particles) {
      if (p.pinned) continue;
      p.velocity.subVectors(p.position, p.previous).divideScalar(dt);
    }

    if (state.collideEnabled) this.resolveCollisions();

    // 地面
    for (const p of this.particles) {
      if (p.position.y < 0.02) {
        p.position.y = 0.02;
        p.velocity.y *= -0.3;
      }
    }
  }

  // ========================================================================
  //  求解器 3: Havok-style (半隐式积分 + 约束投影 + 三子步)
  //  教学近似，并非 Havok Cloth 内核复现
  // ========================================================================
  solveHavok(dt) {
    const gravity = state.gravity;
    const windStr = state.windEnabled ? state.windStrength : 0;
    const damping = state.damping;
    const stiffness = state.stiffness / 100;

    // 与当前 Python/Taichi Havok-style 演示一致：固定三个物理子步。
    const subSteps = 3;

    const subDt = dt / subSteps;
    const elapsed = performance.now() * 0.001;
    const windDir = this._windDirection.set(
      0.45 + Math.sin(elapsed * 0.7) * 0.35,
      0.05,
      0.65
    ).normalize();

    // 半隐式欧拉更新速度后再更新位置；阻尼使用有界的隐式形式。
    // 这是演示中的稳定化策略，不代表 Havok SDK 内部积分器实现。

    const iterations = Math.max(2, Math.floor(state.iterations * 0.6)); // Havok 每子步迭代少, 但子步多

    for (let sub = 0; sub < subSteps; sub++) {
      // Step 1: 半隐式欧拉 + 有界阻尼
      for (const p of this.particles) {
        if (p.pinned) continue;
        p.previous.copy(p.position);

        // 隐式阻尼: v = v / (1 + dt * c)  -- 无条件稳定
        const implicitDamping = 1 + subDt * (damping * 50);
        p.velocity.divideScalar(implicitDamping);

        // 重力 (半隐式)
        p.velocity.y -= gravity * subDt;

        // 风力
        if (windStr > 0) {
          p.velocity.addScaledVector(windDir, windStr * subDt * 0.5);
        }

        // 位置更新
        this.advanceParticle(p, subDt);
      }

      // Step 2: 约束投影 (约束集管理)
      // Havok 使用统一的约束集, 按类型分组投影
      // 距离约束先解, 弯曲约束后解 (优先级排序)
      for (let iter = 0; iter < iterations; iter++) {
        // 距离约束
        for (const c of this.constraints) {
          if (c.type !== "distance") continue;
          this._projectConstraintHavok(c, stiffness, subDt);
        }
        // 弯曲约束 (刚度降低)
        for (const c of this.constraints) {
          if (c.type !== "bending") continue;
          this._projectConstraintHavok(c, stiffness * 0.3, subDt);
        }

        if (state.collideEnabled) {
          this.resolveCollisions();
        }
      }

      // Step 3: 速度从位置差更新 (含速度修正)
      for (const p of this.particles) {
        if (p.pinned) continue;
        const posDelta = this._positionDelta.subVectors(p.position, p.previous);
        const newVel = posDelta.divideScalar(subDt);
        // 混合: 保留部分旧速度 (增加稳定性, 减少抖动)
        p.velocity.lerp(newVel, 0.85);
      }
      if (state.collideEnabled) this.resolveCollisions();
    }

    // 地面
    for (const p of this.particles) {
      if (p.position.y < 0.02) {
        p.position.y = 0.02;
        p.velocity.y *= -0.3;
      }
    }

    // 存储 subSteps 用于调试显示
    state._havokSubSteps = subSteps;
  }

  /**
   * Havok 约束投影 (带速度级修正)
   */
  _projectConstraintHavok(c, stiffness, dt) {
    const delta = this._delta.subVectors(c.p2.position, c.p1.position);
    const dist = delta.length();
    if (dist < 1e-8) return;

    const w1 = c.p1.invMass;
    const w2 = c.p2.invMass;
    const wSum = w1 + w2;
    if (wSum < 1e-8) return;

    const C = dist - c.restLength;
    // Havok 风格: 位置修正 + 速度修正同时进行
    const correctionMag = (stiffness * C * w1) / wSum;
    const dirX = delta.x / dist;
    const dirY = delta.y / dist;
    const dirZ = delta.z / dist;

    if (!c.p1.pinned) {
      c.p1.position.x += correctionMag * dirX;
      c.p1.position.y += correctionMag * dirY;
      c.p1.position.z += correctionMag * dirZ;
    }
    if (!c.p2.pinned) {
      c.p2.position.x -= (stiffness * C * w2) / wSum * dirX;
      c.p2.position.y -= (stiffness * C * w2) / wSum * dirY;
      c.p2.position.z -= (stiffness * C * w2) / wSum * dirZ;
    }

    // 速度级碰撞冲量 (Havok 独有: 位置+速度双投影)
    const relVel = this._relativeVelocity.subVectors(c.p2.velocity, c.p1.velocity);
    const velAlongConstraint = relVel.x * dirX + relVel.y * dirY + relVel.z * dirZ;
    if (Math.abs(velAlongConstraint) > 0.01) {
      const velCorrection = velAlongConstraint * stiffness * 0.3;
      if (!c.p1.pinned) {
        c.p1.velocity.x += velCorrection * dirX * w1 / wSum;
        c.p1.velocity.y += velCorrection * dirY * w1 / wSum;
        c.p1.velocity.z += velCorrection * dirZ * w1 / wSum;
      }
      if (!c.p2.pinned) {
        c.p2.velocity.x -= velCorrection * dirX * w2 / wSum;
        c.p2.velocity.y -= velCorrection * dirY * w2 / wSum;
        c.p2.velocity.z -= velCorrection * dirZ * w2 / wSum;
      }
    }
  }

  /**
   * 碰撞解算 (球体碰撞)
   */
  resolveCollisions() {
    if (bunnyData.collider && bunnyData.mesh) {
      this.resolveMeshCollisions();
      return;
    }

    // 模型碰撞器不可用时退回球体近似。
    for (const p of this.particles) {
      if (p.pinned) continue;
      for (const sphere of this.collisionSpheres) {
        const diff = this._collisionDelta.subVectors(p.position, sphere.center);
        const dist = diff.length();
        if (dist < sphere.radius && dist > 1e-6) {
          // 推出碰撞体
          const pushOut = diff.multiplyScalar(sphere.radius / dist);
          p.position.copy(sphere.center).add(pushOut);
          // 速度反射 (法向分量反转)
          const normal = diff.normalize();
          const velAlongNormal = p.velocity.dot(normal);
          if (velAlongNormal < 0) {
            p.velocity.addScaledVector(normal, -velAlongNormal * 1.2);
          }
        }
      }
    }
  }

  resolveMeshCollisions() {
    const collider = bunnyData.collider;
    characterGroup.updateMatrixWorld(true);
    const worldToLocal = this._worldToLocal.copy(characterGroup.matrixWorld).invert();
    this._normalMatrix.getNormalMatrix(characterGroup.matrixWorld);
    let contactCount = 0;

    for (const p of this.particles) {
      if (p.pinned) continue;
      const localPosition = this._localPosition.copy(p.position).applyMatrix4(worldToLocal);
      const localPrevious = this._localPrevious.copy(p.previous).applyMatrix4(worldToLocal);
      const hit = collider.segmentCast(localPrevious, localPosition);
      // 无扫掠命中时仍查询全局最近三角面。只查 proximity 会漏掉已经
      // 深入模型的粒子，使强风下的围巾无法被重新推出。
      const contact = hit || collider.closestPoint(localPosition);
      if (!contact) continue;

      if (!hit) {
        this._localVelocity.subVectors(localPosition, contact.point);
        // 最近点在粒子内侧时法线点积为负；外侧但位于厚度内时也推出。
        const signedDistance = this._localVelocity.dot(contact.normal);
        if (signedDistance >= CLOTH_THICKNESS) continue;
      }

      const resolvedLocal = this._localPosition.copy(contact.point).addScaledVector(contact.normal, CLOTH_THICKNESS);
      p.position.copy(resolvedLocal).applyMatrix4(characterGroup.matrixWorld);
      // 固定点也要记住推出后的表面位置，否则下一帧会再次插回模型。
      if (p.pinned && p.pinLocalPosition) p.pinLocalPosition.copy(resolvedLocal);
      contactCount++;
      const worldNormal = this._worldNormal.copy(contact.normal).applyMatrix3(this._normalMatrix).normalize();
      const inwardVelocity = p.velocity.dot(worldNormal);
      if (inwardVelocity < 0) p.velocity.addScaledVector(worldNormal, -inwardVelocity);
      // 少量切向摩擦抑制贴面抖动，同时保留披风滑动。
      const outwardVelocity = Math.max(0, p.velocity.dot(worldNormal));
      p.velocity.addScaledVector(worldNormal, -outwardVelocity).multiplyScalar(0.995)
        .addScaledVector(worldNormal, outwardVelocity);
    }
    state._meshContacts = (state._meshContacts || 0) + contactCount;
  }

  /**
   * 更新碰撞体位置 (跟随兔子动画)
   */
  updateCollisionPositions() {
    characterGroup.updateMatrixWorld(true);
    for (const sphere of this.collisionSpheres) {
      sphere.center.copy(sphere.localCenter).applyMatrix4(characterGroup.matrixWorld);
    }
  }
}

// ============================================================================
//  创建布料实例
// ============================================================================
const cloth = new ClothMesh();

function rebuildCloth() {
  // 如果兔子还没加载完成, 稍后再重建 (buildCharacter 完成后会调 rebuildCloth)
  if (!bunnyData.loaded) return;

  switch (state.clothType) {
    case "scarf": cloth.createScarf(); break;
    case "hair": cloth.createHair(); break;
    case "cape": cloth.createCape(); break;
  }

  // 头发管状网格索引
  if (state.clothType === "hair") {
    const segments = 14;
    const strands = 8;
    cloth.rebuildHairTubeUserData(segments, strands);
  }

  state.particleCount = cloth.particles.length;
  state.constraintCount = cloth.constraints.length;
  updateMetrics();
}

rebuildCloth();

// ============================================================================
//  兔子蹦跳动画
// ============================================================================
function animateCharacter(dt) {
  if (!bunnyData.loaded) return;

  if (!state.walking) {
    characterGroup.position.y = THREE.MathUtils.lerp(characterGroup.position.y, 0, 0.1);
    characterGroup.rotation.z = THREE.MathUtils.lerp(characterGroup.rotation.z, 0, 0.1);
    return;
  }

  state.walkTime += dt;
  const t = state.walkTime * 3;
  const hopPhase = Math.sin(t * 2);
  const hopAbs = Math.abs(hopPhase);

  // 蹦跳: 整体上下移动
  characterGroup.position.y = hopAbs * 0.15;

  // 轻微左右摇摆
  characterGroup.rotation.z = Math.sin(t * 2) * 0.03;
}

// ============================================================================
//  主循环
// ============================================================================
const clock = new THREE.Clock();
let frameCount = 0;
let fpsTimer = 0;

function animate() {
  requestAnimationFrame(animate);

  const dt = Math.min(clock.getDelta(), 1 / 30); // 限制最大步长
  frameCount++;
  fpsTimer += dt;
  if (fpsTimer >= 0.5) {
    state.fps = Math.round(frameCount / fpsTimer);
    frameCount = 0;
    fpsTimer = 0;
    updateMetrics();
  }

  if (state.running) {
    animateCharacter(dt);
    state._meshContacts = 0;

    // 固定点和碰撞体都随角色的完整世界变换移动。
    cloth.updatePinnedPositions(dt);
    cloth.updateCollisionPositions();

    // Python 版本中 PBD/XPBD 每帧至少两个物理子步；Havok-style
    // 在求解器内部执行三个子步。
    const frameSubsteps = state.algorithm === "havok" ? 1 : 2;
    const subDt = dt / frameSubsteps;
    for (let substep = 0; substep < frameSubsteps; substep++) {
      switch (state.algorithm) {
        case "pbd":   cloth.solvePBD(subDt);   break;
        case "xpbd":  cloth.solveXPBD(subDt);  break;
        case "havok": cloth.solveHavok(subDt); break;
      }
    }
    canvas.dataset.meshContacts = String(state._meshContacts);

    // 同步渲染
    cloth.syncGeometry();
  }

  controls.update();
  renderer.render(scene, camera);
}

animate();

// ============================================================================
//  窗口适配
// ============================================================================
window.addEventListener("resize", () => {
  camera.aspect = window.innerWidth / window.innerHeight;
  camera.updateProjectionMatrix();
  renderer.setSize(window.innerWidth, window.innerHeight);
});

// ============================================================================
//  UI 绑定
// ============================================================================
function updateMetrics() {
  document.getElementById("fpsValue").textContent = state.fps || "--";
  document.getElementById("particleValue").textContent = state.particleCount;
  document.getElementById("constraintValue").textContent = state.constraintCount;
  document.getElementById("contactValue").textContent = state._meshContacts || 0;
}

// 算法切换
document.querySelectorAll(".algorithm-button").forEach(btn => {
  btn.addEventListener("click", () => {
    document.querySelectorAll(".algorithm-button").forEach(b => b.classList.remove("active"));
    btn.classList.add("active");
    state.algorithm = btn.dataset.algo;
    updateInfoPanel();
  });
});

// 布料类型切换
document.querySelectorAll("#clothTypeButtons button").forEach(btn => {
  btn.addEventListener("click", () => {
    document.querySelectorAll("#clothTypeButtons button").forEach(b => b.classList.remove("active"));
    btn.classList.add("active");
    state.clothType = btn.dataset.cloth;
    rebuildCloth();
  });
});

// 暂停
document.getElementById("pauseButton").addEventListener("click", () => {
  state.running = !state.running;
  document.getElementById("pauseIcon").textContent = state.running ? "||" : ">";
  document.getElementById("infoStatus").textContent = state.running ? "运行中" : "已暂停";
});

// 滑块
const sliders = [
  ["gravitySlider", "gravityValue", "gravity", v => parseFloat(v), v => v.toFixed(1)],
  ["windSlider", "windValue", "windStrength", v => parseFloat(v), v => v.toFixed(1)],
  ["stiffnessSlider", "stiffnessValue", "stiffness", v => parseInt(v), v => v],
  ["iterationSlider", "iterationValue", "iterations", v => parseInt(v), v => v],
  ["dampingSlider", "dampingValue", "damping", v => parseFloat(v), v => v.toFixed(3)],
];

for (const [sliderId, valueId, stateKey, parseFn, formatFn] of sliders) {
  const slider = document.getElementById(sliderId);
  const valueLabel = document.getElementById(valueId);
  slider.addEventListener("input", () => {
    state[stateKey] = parseFn(slider.value);
    valueLabel.textContent = formatFn(state[stateKey]);
    if (stateKey === "windStrength") {
      document.querySelectorAll(".wind-presets button").forEach(button => button.classList.remove("active"));
    }
  });
}

document.querySelectorAll(".wind-presets button").forEach(button => {
  button.addEventListener("click", () => {
    const value = Number(button.dataset.wind);
    state.windStrength = value;
    state.windEnabled = true;
    document.getElementById("windSlider").value = String(value);
    document.getElementById("windValue").textContent = value.toFixed(1);
    document.getElementById("windToggle").checked = true;
    document.querySelectorAll(".wind-presets button").forEach(item => item.classList.toggle("active", item === button));
  });
});

// 开关
document.getElementById("windToggle").addEventListener("change", e => state.windEnabled = e.target.checked);
document.getElementById("collideToggle").addEventListener("change", e => state.collideEnabled = e.target.checked);
document.getElementById("wireToggle").addEventListener("change", e => {
  state.wireframe = e.target.checked;
  if (cloth.material) cloth.material.wireframe = e.target.checked;
});
document.getElementById("autoRotateToggle").addEventListener("change", e => controls.autoRotate = e.target.checked);

// 按钮
document.getElementById("resetButton").addEventListener("click", () => rebuildCloth());
document.getElementById("walkButton").addEventListener("click", () => {
  state.walking = !state.walking;
  document.getElementById("walkButton").textContent = state.walking ? "停止蹦跳" : "Bunny 蹦跳";
});

// Info panel 折叠
document.getElementById("infoCollapse").addEventListener("click", () => {
  document.getElementById("infoPanel").classList.toggle("collapsed");
});

// ============================================================================
//  算法信息面板
// ============================================================================
const algorithmInfo = {
  pbd: {
    code: "PBD",
    title: "PBD 位置约束动力学",
    author: "Muller, Heidelberger, Hennix, Ratcliff (2007)",
    body: `
      <h3>核心思想</h3>
      <p>PBD 不直接解牛顿方程, 而是先预测粒子位置, 再通过<strong>迭代投影约束</strong>修正位置, 最后从位置差反推速度。位置即真相。</p>
      <h3>算法流程</h3>
      <code>for each particle:
  v += f * dt / m          // 施加外力
  p_prev = p
  p += v * dt              // 预测位置

for iter = 1..N:           // 约束迭代
  for each constraint C:
    project C onto p       // 修正位置

for each particle:
  v = (p - p_prev) / dt    // 速度从位置差得到</code>
      <h3>关键特性</h3>
      <ul>
        <li>直接操作位置, 天然避免穿模</li>
        <li>刚度由迭代次数决定 (N 越多越硬)</li>
        <li>简单快速, 但刚度与时间步耦合</li>
      </ul>
      <h3>公式</h3>
      <span class="formula">Δp₁ = -w₁ / (w₁+w₂) · |C|/|∇C| · ∇C
Δp₂ = +w₂ / (w₁+w₂) · |C|/|∇C| · ∇C</span>
      <h3>缺点</h3>
      <ul>
        <li>刚度依赖迭代次数, 难以精确控制</li>
        <li>大时间步会变软 (刚度-步长耦合)</li>
        <li>无能量守恒, 可能注入人工能量</li>
      </ul>
    `,
  },
  xpbd: {
    code: "XPBD",
    title: "XPBD 扩展位置约束动力学",
    author: "Macklin, Muller, Chentanez (2016)",
    body: `
      <h3>核心改进</h3>
      <p>XPBD 在 PBD 基础上引入<strong>柔量 (Compliance)</strong>参数 α, 将刚度与迭代次数解耦。用 Lagrange 乘子累积修正量, 实现物理正确的软约束。</p>
      <h3>算法流程</h3>
      <code>for each particle:
  v += f * dt / m
  p_prev = p
  p += v * dt

for each constraint:
  λ = 0                   // 重置乘子

for iter = 1..N:
  for each constraint C:
    α̃ = α / dt²           // 时间步归一化柔量
    Δλ = (-C - α̃·λ) / (w₁+w₂+α̃)
    λ += Δλ               // 累积乘子
    p₁ += w₁·Δλ·∇C
    p₂ -= w₂·Δλ·∇C

for each particle:
  v = (p - p_prev) / dt</code>
      <h3>关键特性</h3>
      <ul>
        <li>刚度与迭代次数<strong>解耦</strong> (α 是物理参数)</li>
        <li>Lagrange 乘子跨迭代累积, 收敛更平滑</li>
        <li>可模拟真实材料刚度 (类似胡克定律)</li>
      </ul>
      <h3>公式</h3>
      <span class="formula">Δλ = (-C - α̃·λ) / (∇Cᵀ·M⁻¹·∇C + α̃)

α = 1 / k   (compliance = 1 / stiffness)</span>
      <h3>对比 PBD</h3>
      <table class="compare-table">
        <tr><th>特性</th><th>PBD</th><th>XPBD</th></tr>
        <tr><td>刚度控制</td><td>迭代次数</td><td>compliance α</td></tr>
        <tr><td>步长耦合</td><td>有</td><td>无</td></tr>
        <tr><td>乘子累积</td><td>无</td><td>有 (λ)</td></tr>
        <tr><td>物理正确性</td><td>近似</td><td>更准确</td></tr>
      </table>
    `,
  },
  havok: {
    code: "Havok",
    title: "Havok-style 稳定约束投影",
    author: "教学近似 - 稳定约束投影 + 固定三子步",
    body: `
      <h3>核心技术路线</h3>
      <p>本模式借鉴 Havok Cloth 公开接口中的<strong>约束集、子步与刚度调节</strong>概念，用半隐式积分和位置/速度修正构成教学近似；它不是 Havok 求解内核的复现。</p>
      <h3>三大特性</h3>
      <ul>
        <li><strong>有界阻尼</strong>: v_new = v / (1 + dt·c), 避免大步长下阻尼翻转</li>
        <li><strong>固定三子步</strong>: 与当前 Python/Taichi 演示保持一致</li>
        <li><strong>位置+速度双投影</strong>: 约束同时修正位置和速度</li>
      </ul>
      <h3>算法流程</h3>
      <code>// 高风力下稳定运行的固定子步数
subSteps = 3

for sub = 1..subSteps:
  subDt = dt / subSteps

  // 半隐式积分 + 有界阻尼
  for each particle:
    v /= (1 + subDt·c)    // 隐式阻尼
    v += f·subDt / m      // 外力
    p_prev = p
    p += v·subDt

  // 约束投影 (距离优先, 弯曲在后)
  for iter = 1..N:
    for distance constraints:
      project position + velocity
    for bending constraints:
      project position + velocity
    resolve collisions

  // 速度混合 (稳定性)
  for each particle:
    v = lerp(v, (p-p_prev)/subDt, 0.85)</code>
      <h3>位移安全条件</h3>
      <span class="formula">subSteps = 3

|Δp| ≤ 0.032（每个物理子步）</span>
      <h3>为什么此教学模式更稳定</h3>
      <table class="compare-table">
        <tr><th>特性</th><th>PBD/XPBD</th><th>Havok</th></tr>
        <tr><td>时间积分</td><td>半隐式欧拉</td><td>半隐式欧拉 + 有界阻尼</td></tr>
        <tr><td>子步进</td><td>固定 2 步</td><td>固定 3 步</td></tr>
        <tr><td>约束投影</td><td>仅位置</td><td>位置+速度</td></tr>
        <tr><td>大步长稳定</td><td>差</td><td>好</td></tr>
        <tr><td>每步开销</td><td>低</td><td>较高</td></tr>
      </table>
      <h3>当前子步数</h3>
      <p id="havokSubStepInfo">Havok 物理子步: <strong id="havokSubSteps">--</strong> 步/帧</p>
    `,
  },
};

function updateInfoPanel() {
  const info = algorithmInfo[state.algorithm];
  document.getElementById("infoCode").textContent = info.code;
  document.getElementById("infoTitle").textContent = info.title;
  document.getElementById("infoAuthor").textContent = info.author;
  document.getElementById("infoBody").innerHTML = info.body;
}

updateInfoPanel();

// 更新 Havok 子步信息
setInterval(() => {
  if (state.algorithm === "havok") {
    const el = document.getElementById("havokSubSteps");
    if (el) el.textContent = state._havokSubSteps || "--";
  }
}, 200);
