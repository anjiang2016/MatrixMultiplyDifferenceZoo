import numpy as np
import json
import matplotlib.pyplot as plt
import os
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from funcs import sigmoid

def save_all_class_heatmaps(heatmap_logits, heatmap_gt, epoch, save_dir='bev_vis', num_cols=4):
    """
    保存所有类别的热图对比（预测 vs GT），每个类别占两列（Pred, GT）
    heatmap_logits: (B, C, X, Y)
    heatmap_gt: (B, C, X, Y)
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # 取第一个batch，转为概率
    pred_prob = sigmoid(heatmap_logits)[0]  # (C, X, Y)
    gt = heatmap_gt[0]                     # (C, X, Y)
    
    C, X, Y = pred_prob.shape
    num_cols = min(num_cols, C)           # 每行显示多少个类别（每类别占2列）
    num_rows = (C + num_cols - 1) // num_cols
    
    fig, axes = plt.subplots(num_rows, num_cols * 2, figsize=(num_cols * 6, num_rows * 4))
    # 确保 axes 是二维数组（即使只有一个子图）
    if num_rows == 1 and num_cols * 2 == 1:
        axes = np.array([[axes]])
    elif num_rows == 1:
        axes = axes.reshape(1, -1)
    elif num_cols * 2 == 1:
        axes = axes.reshape(-1, 1)
    
    for c in range(C):
        row = c // num_cols
        col = (c % num_cols) * 2
        # 预测图
        ax_pred = axes[row, col]
        im = ax_pred.imshow(pred_prob[c], cmap='hot', vmin=0, vmax=1)
        ax_pred.set_title(f'Pred Class {c}')
        plt.colorbar(im, ax=ax_pred)
        # GT图
        ax_gt = axes[row, col+1]
        im2 = ax_gt.imshow(gt[c], cmap='hot', vmin=0, vmax=1)
        ax_gt.set_title(f'GT Class {c}')
        plt.colorbar(im2, ax=ax_gt)
    
    # 隐藏多余子图（如果类别数不能填满最后一行）
    for i in range(C, num_rows * num_cols):
        row = i // num_cols
        col = (i % num_cols) * 2
        axes[row, col].axis('off')
        axes[row, col+1].axis('off')
    
    plt.tight_layout()
    save_path = os.path.join(save_dir, f'heatmaps_all_classes_epoch_{epoch:04d}.png')
    plt.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"✅ All class heatmaps saved to {save_path}")
def save_bev_heatmap(heatmap_logits, heatmap_gt, epoch, save_dir='bev_vis'):
    """
    保存 BEV 热图预测和 GT 的对比图。
    heatmap_logits: (1, C, X, Y) 网络输出（未经 sigmoid）
    heatmap_gt: (1, C, X, Y) 真实标签
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # 取第一个样本，所有类别合并（或指定某一类，比如车辆类索引3）
    # 这里我们取所有类别的最大值，形成"综合目标概率图"
    pred_prob = sigmoid(heatmap_logits)  # (1, C, X, Y)
    pred_combined = np.max(pred_prob[0], axis=0)  # (X, Y) 取最大类别概率
    gt_combined = np.max(heatmap_gt[0], axis=0)   # (X, Y)
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # 1. 预测热图
    im1 = axes[0].imshow(pred_combined, cmap='hot', vmin=0, vmax=1)
    axes[0].set_title(f'Predicted Heatmap (Epoch {epoch})')
    plt.colorbar(im1, ax=axes[0])
    
    # 2. GT 热图
    im2 = axes[1].imshow(gt_combined, cmap='hot', vmin=0, vmax=1)
    axes[1].set_title('Ground Truth Heatmap')
    plt.colorbar(im2, ax=axes[1])
    
    # 3. 误差图（Pred - GT），红色表示过预测，蓝色表示欠预测
    diff = pred_combined - gt_combined
    im3 = axes[2].imshow(diff, cmap='RdBu', vmin=-0.5, vmax=0.5)
    axes[2].set_title('Difference (Pred - GT)')
    plt.colorbar(im3, ax=axes[2])
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'heatmap_epoch_{epoch:04d}.png'), dpi=150)
    plt.close(fig)
    print(f"✅ Heatmap saved to {save_dir}/heatmap_epoch_{epoch:04d}.png")

def generate_inline_html(heatmap_gt, output_html='view_heatmap_inline.html', step=1):
    # 处理维度...
    if heatmap_gt.ndim == 2:
        H, W = heatmap_gt.shape
        C = 1
        data = heatmap_gt[np.newaxis, :, :]
    elif heatmap_gt.ndim == 3:
        C, H, W = heatmap_gt.shape
        data = heatmap_gt
    elif heatmap_gt.ndim == 4:
        B, C, H, W = heatmap_gt.shape
        data = heatmap_gt[0]
    else:
        raise ValueError(f"不支持 shape: {heatmap_gt.shape}")

    if step > 1:
        data = data[:, ::step, ::step]
        H, W = data.shape[1], data.shape[2]

    shape = (C, H, W)
    values = data.tolist()
    flat_all = np.array(values).flatten()
    data_min = flat_all.min()
    data_max = flat_all.max()
    data_mean = flat_all.mean()
    data_std = flat_all.std()
    max_channel = C - 1

    shape_str = f"{C} x {H} x {W}"

    # 将 max_channel 加入 heatmapData
    js_data = f"""const heatmapData = {{
        shape: {json.dumps(shape)},
        values: {json.dumps(values)},
        stats: {{ min: {data_min:.4f}, max: {data_max:.4f}, mean: {data_mean:.4f}, std: {data_std:.4f} }},
        max_channel: {max_channel}
    }};"""

    html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>3D Heatmap Viewer</title>
    <style>
        body {{ margin: 0; overflow: hidden; font-family: Arial; }}
        #info {{
            position: absolute; top: 20px; left: 20px;
            color: white; background: rgba(0,0,0,0.8);
            padding: 12px 18px; border-radius: 8px;
            pointer-events: none; z-index: 10;
            font-size: 14px;
        }}
        #info div {{ margin: 2px 0; }}
        #controls {{
            position: absolute; bottom: 40px; left: 50%;
            transform: translateX(-50%);
            color: white; background: rgba(0,0,0,0.7);
            padding: 15px 30px; border-radius: 30px;
            display: flex; align-items: center; gap: 20px;
            pointer-events: auto; z-index: 10;
        }}
        #controls input[type="range"] {{ width: 300px; cursor: pointer; }}
    </style>
</head>
<body>
    <div id="info">
        <div>Shape: {shape_str}</div>
        <div>Channel: <span id="channelDisplay">0 / {max_channel}</span></div>
        <div>Min: {data_min:.4f}, Max: {data_max:.4f}</div>
        <div>Mean: {data_mean:.4f}, Std: {data_std:.4f}</div>
    </div>

    <div id="controls">
        <label for="channelSlider">Channel</label>
        <input type="range" id="channelSlider" min="0" max="{max_channel}" value="0" step="1">
        <span id="channelValue">0</span>
    </div>

    <script type="importmap">
    {{
        "imports": {{
            "three": "https://unpkg.com/three@0.160.0/build/three.module.js",
            "three/addons/": "https://unpkg.com/three@0.160.0/examples/jsm/"
        }}
    }}
    </script>

    <script type="module">
        import * as THREE from 'three';
        import {{ OrbitControls }} from 'three/addons/controls/OrbitControls.js';

        {js_data}

        const {{ shape, values, stats, max_channel }} = heatmapData;
        const [C, H, W] = shape;
        const flatData = values.map(ch => ch.flat());

        function getColor(val, minVal, maxVal) {{
            if (maxVal === minVal) return new THREE.Color(0x888888);
            const t = (val - minVal) / (maxVal - minVal);
            const r = Math.min(1, t * 2);
            const g = Math.min(1, 2 - t * 2);
            const b = Math.max(0, 1 - t * 2);
            return new THREE.Color(r, g, b);
        }}

        let scene, camera, renderer, controls;
        let cubeGroup;

        function initScene() {{
            scene = new THREE.Scene();
            scene.background = new THREE.Color(0x111122);

            camera = new THREE.PerspectiveCamera(45, window.innerWidth / window.innerHeight, 0.1, 1000);
            const maxDim = Math.max(W, H);
            camera.position.set(maxDim * 0.8, maxDim * 0.6, maxDim * 0.8);
            camera.lookAt(0, 0, 0);

            renderer = new THREE.WebGLRenderer({{ antialias: true }});
            renderer.setSize(window.innerWidth, window.innerHeight);
            renderer.setPixelRatio(window.devicePixelRatio);
            document.body.appendChild(renderer.domElement);

            controls = new OrbitControls(camera, renderer.domElement);
            controls.enableDamping = true;
            controls.target.set(0, 0, 0);
            controls.update();

            scene.add(new THREE.AmbientLight(0x404060));
            const dirLight = new THREE.DirectionalLight(0xffffff, 1);
            dirLight.position.set(1, 2, 1);
            scene.add(dirLight);
            const backLight = new THREE.DirectionalLight(0x88aaff, 0.5);
            backLight.position.set(-1, -0.5, -1);
            scene.add(backLight);

            const axes = new THREE.AxesHelper(Math.max(W, H) * 0.5);
            scene.add(axes);

            const gridHelper = new THREE.GridHelper(Math.max(W, H), 20, 0x88aaff, 0x446688);
            gridHelper.position.y = -0.5;
            scene.add(gridHelper);

            cubeGroup = new THREE.Group();
            scene.add(cubeGroup);

            window.addEventListener('resize', () => {{
                camera.aspect = window.innerWidth / window.innerHeight;
                camera.updateProjectionMatrix();
                renderer.setSize(window.innerWidth, window.innerHeight);
            }});
        }}

        function updateChannel(channelIdx) {{
            while(cubeGroup.children.length) cubeGroup.remove(cubeGroup.children[0]);

            const channelData = flatData[channelIdx];
            const maxVal = Math.max(...channelData);
            const minVal = Math.min(...channelData);
            const gap = 0.1;
            const cubeSize = 1 - gap;
            const geometry = new THREE.BoxGeometry(cubeSize, cubeSize, cubeSize);

            for (let i = 0; i < H; i++) {{
                for (let j = 0; j < W; j++) {{
                    const idx = i * W + j;
                    const val = channelData[idx];
                    const color = getColor(val, minVal, maxVal);
                    const mat = new THREE.MeshStandardMaterial({{
                        color, roughness: 0.3, metalness: 0.0,
                        emissive: color.clone().multiplyScalar(0.1),
                    }});
                    const cube = new THREE.Mesh(geometry, mat);
                    cube.position.set(j - (W-1)/2, 0, i - (H-1)/2);
                    cubeGroup.add(cube);
                }}
            }}

            document.getElementById('channelDisplay').textContent = `${{channelIdx}} / ${{max_channel}}`;
            document.getElementById('channelValue').textContent = channelIdx;
        }}

        initScene();
        updateChannel(0);

        function animate() {{
            requestAnimationFrame(animate);
            controls.update();
            renderer.render(scene, camera);
        }}
        animate();

        document.getElementById('channelSlider').addEventListener('input', (e) => {{
            updateChannel(parseInt(e.target.value));
        }});
    </script>
</body>
</html>
"""

    with open(output_html, 'w', encoding='utf-8') as f:
        f.write(html_content)
    print(f"✅ 已生成 HTML: {output_html}")
    print(f"   Shape: {shape_str}, 立方体数: {H*W}")
    print(f"   数据范围: [{data_min:.4f}, {data_max:.4f}]")
