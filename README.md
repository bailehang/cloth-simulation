# Cloth Simulation Lab - 布料模拟算法实验室

> 基于 Havok Cloth 5.8.1 源码分析, 实现并对比三种主流布料求解算法

## 演示内容

在 Stanford Bunny (图形学经典 3D 测试模型) 上, 实时对比三种布料物理求解算法:

> **Stanford Bunny** - 1994 年由 Greg Turk 和 Marc Levoy 在斯坦福大学创建,
> 3D 扫描陶土兔子雕像, 69,451 个三角面片, 图形学论文中使用频率最高的测试模型之一。
> 本 demo 加载其简化版 .ply 文件 (8,171 顶点 / 16,301 面)。

| 算法 | 全称 | 论文/来源 | 核心思想 |
|---|---|---|---|
| **PBD** | Position Based Dynamics | Muller et al. 2007 | 位置投影迭代, 从位置差反推速度 |
| **XPBD** | Extended PBD | Macklin et al. 2016 | 引入 compliance 柔量, 刚度与迭代解耦 |
| **Havok-style** | Stabilized Constraint Projection | 教学近似 | 半隐式积分 + 自适应子步进 + 位置速度双修正 |

## 布料类型

- **披风** - 默认首先展示；18 × 19 长款网格，固定在兔子头部后方中央
- **围巾** - 21 环网格，脖颈固定，有效垂坠长度约 1.4 场景单位（原来的 5 倍）
- **耳饰** - 8 束链式粒子, 耳尖固定, 随动作摆动

## Quick Start

需要 Python 3.x (用于启动本地 HTTP 服务器):

```bash
# Windows
start_server.bat

# 或手动
python -m http.server 8080
```

然后浏览器打开 http://localhost:8080

## 交互操作

- **鼠标拖拽** - 旋转视角
- **滚轮** - 缩放
- **算法切换** - 左侧面板选择 PBD / XPBD / Havok
- **布料切换** - 围巾 / 耳饰 / 披风
- **物理参数** - 重力、风力、刚度、迭代次数、阻尼
- **兔子蹦跳** - 观察布料在运动中的表现
- **风场开关** - 模拟风吹效果
- **碰撞开关** - 布料与 Bunny 真实三角网格碰撞；BVH 加速最近点与连续穿越检测

## 算法对比

### PBD (Position Based Dynamics)

```
预测位置 p' = p + v*dt
迭代 N 次:
  对每个约束 C: 投影修正位置
速度 v = (p' - p) / dt
```

- 刚度由迭代次数决定
- 简单快速
- 大时间步会变软 (刚度-步长耦合)

### XPBD (Extended PBD)

```
预测位置 p' = p + v*dt
迭代 N 次:
  对每个约束 C:
    Δλ = (-C - α̃·λ) / (w₁+w₂+α̃)
    λ += Δλ  (Lagrange 乘子累积)
    修正位置
速度 v = (p' - p) / dt
```

- compliance α = 1/k (物理刚度参数)
- 刚度与迭代次数解耦
- 更物理正确的软约束

### Havok-style（稳定约束投影 + 子步进）

```
自适应子步: subSteps = clamp(⌈maxVel·dt/(0.5·minEdge)⌉, 1, 8)
每子步:
  有界阻尼: v /= (1 + subDt·c)
  施加外力 + 预测位置
  约束投影 (距离优先, 弯曲在后, 位置+速度双修正)
  碰撞处理
  速度混合: v = lerp(v, Δp/subDt, 0.85)
```

- 半隐式积分 + 有界阻尼: 比朴素显式更新更稳定
- 自适应子步: 高速时自动细化
- 位置+速度双投影: 更真实
- 参考来源: Havok Cloth 5.8.1 UE 插件公开接口中的约束集、可配置子步与刚度调节概念；并非 Havok 内核复现

## 项目结构

```
cloth-simulation-demo/
├── index.html          # 入口页面
├── styles.css          # 暗色主题样式
├── src/
│   └── main.js         # 核心逻辑 (角色模型 + 三种求解器 + UI)
├── bunny.ply            # Stanford Bunny 扫描模型数据
├── start_server.bat    # Windows 启动脚本
├── start_server.ps1    # PowerShell 启动脚本
└── README.md
```

## 技术栈

- Three.js 0.166.1 (ES Modules + importmap)
- 纯前端, 无构建工具, 无依赖安装
- WebGL 渲染 + PCF 软阴影

## 参考来源

- Havok Cloth 5.8.1 源码: `Engine/Plugins/Havok/HavokCloth/`
- Muller et al., "Position Based Dynamics", 2007
- Macklin et al., "XPBD: Position-Based Simulation of Compliant Constrained Dynamics", 2016
- 100pathfinding-algorithms 演示格式参考

## License

MIT
