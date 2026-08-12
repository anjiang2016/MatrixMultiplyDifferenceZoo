import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from tqdm import tqdm
from train_nuscenes import (
    load_json, load_sample_data, get_annotations,
    generate_heatmap_gt, generate_reg_gt, generate_depth_gt,
    build_geometry_indices,
    BEV_SIZE, DEPTH_BINS, DEPTH_RANGE, DATA_ROOT, CAMERAS, class_mapping,
    init_model_params, bev_forward, bev_backward, compute_losses, d_compute_losses
)
from funcs import (
     sigmoid
)
# 从 class_mapping 构建反向映射
from train_nuscenes import class_mapping
id_to_name = {v: k for k, v in class_mapping.items()}

# ========== 坐标转换 ==========
def grid_to_world(gx, gy, bev_size=BEV_SIZE, bev_range=(-50,50), bev_res=1.0):
    """
    将BEV网格坐标 (gx, gy) 转换为世界坐标 (x, y) 米。
    注意：训练时 heatmap_gt 生成时做了 gx = bev_size[0] - 1 - gx，gy = bev_size[1] - 1 - gy
    因此解码出的网格坐标已经是翻转后的。我们需要将其映射回世界坐标：
    world_x = (bev_size[0] - 1 - gx) * bev_res + bev_range[0]
    #world_y = (bev_size[1] - 1 - gy) * bev_res + bev_range[0]
    world_y = (gy) * bev_res + bev_range[0]
    """
    y = (gx) * bev_res + bev_range[0]
    x = (bev_size[1] - 1 - gy) * bev_res + bev_range[0]
    return x, y
def grid_to_world_gt(gx, gy, bev_size=BEV_SIZE, bev_range=(-50,50), bev_res=1.0):
    """
    将BEV网格坐标 (gx, gy) 转换为世界坐标 (x, y) 米。
    注意：训练时 heatmap_gt 生成时做了 gx = bev_size[0] - 1 - gx，gy = bev_size[1] - 1 - gy
    因此解码出的网格坐标已经是翻转后的。我们需要将其映射回世界坐标：
    world_x = (bev_size[0] - 1 - gx) * bev_res + bev_range[0]
    #world_y = (bev_size[1] - 1 - gy) * bev_res + bev_range[0]
    world_y = (gy) * bev_res + bev_range[0]
    """
    x = (gx) * bev_res + bev_range[0]
    y = (gy) * bev_res + bev_range[0]
    return x, y
def decode_heatmap(heatmap, reg, threshold=0.1, topk=30):
    """
    解码 heatmap 和 reg，返回预测框列表。
    heatmap: (1, num_classes, X, Y)  sigmoid 概率
    reg: (1, 8, X, Y)  回归输出
    """
    B, C, X, Y = heatmap.shape
    preds = []
    for c in range(C):
        hm = heatmap[0, c]  # (X, Y)
        # 3x3 最大池化找到局部峰值
        from scipy.ndimage import maximum_filter
        max_hm = maximum_filter(hm, size=3, mode='constant', cval=0.0)
        peaks = (hm == max_hm) & (hm > threshold)
        y_idxs, x_idxs = np.where(peaks)
        #print(f"y_indxs : {y_idxs}, x_idxs : {x_idxs}")
        if len(y_idxs) == 0:
            continue
        # 获取分数
        scores = hm[y_idxs, x_idxs]
        # 获取回归值（使用循环避免形状问题）
        reg_vals = np.array([reg[0, :, x, y] for y, x in zip(y_idxs, x_idxs)])  # (N, 8)
        reg_vals = reg_vals.T  # (8, N)
        # 按分数排序取 topk
        order = np.argsort(scores)[::-1][:topk]
        y_idxs = y_idxs[order]
        x_idxs = x_idxs[order]
        scores = scores[order]
        reg_vals = reg_vals[:, order]
        # 构建预测
        for i in range(len(x_idxs)):
            w, l, h, sin, cos, z = reg_vals[2, i], reg_vals[3, i], reg_vals[4, i], reg_vals[5, i], reg_vals[6, i], reg_vals[7, i]
            preds.append({
                'class_idx': c,
                'score': scores[i],
                'gx': x_idxs[i],
                'gy': y_idxs[i],
                'w': w,
                'l': l,
                'h': h,
                'sin': sin,
                'cos': cos,
                'z': z
            })
    return preds
# ========== 可视化函数 ==========
def visualize_bev(gt_boxes, pred_boxes, bev_size=BEV_SIZE, save_path=None):
    """
    在 BEV 俯视图上绘制 GT（绿色）和预测（红色）框。
    gt_boxes, pred_boxes: list of dict with keys 'gx','gy','w','l','sin','cos','score' etc.
    """
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    # 设置坐标轴范围（以米为单位）
    x_range = (-50, 50)
    y_range = (-50, 50)
    ax.set_xlim(x_range)
    ax.set_ylim(y_range)
    ax.set_aspect('equal')
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.grid(True, linestyle='--', alpha=0.5)
    # 绘制预测框（红色）
    for box in pred_boxes:
        x_world, y_world = grid_to_world_gt(box['gx'], box['gy'])
        w, l, sin, cos = box['w'], box['l'], box['sin'], box['cos']
        # 计算朝向角
        yaw = np.arctan2(sin, cos)
        # 创建矩形，中心在 (x_world, y_world)，宽度为 l（因为 BEV 中我们通常将 l 视为沿 y? 需要确认）
        # 在 nuScenes 中，w 是宽度（x方向），l 是长度（y方向），但实际在BEV中，我们通常将 w 视为 x 方向宽度，l 视为 y 方向长度
        # 为了简单，我们画矩形时用 w 作为宽度（x方向），l 作为高度（y方向），并旋转 yaw
        rect = patches.Rectangle(
            (x_world - w/2, y_world - l/2), w, l,
            angle=np.rad2deg(yaw),
            linewidth=2, edgecolor='red', facecolor='none', alpha=0.8
        )
        ax.add_patch(rect)
        # 可选：在中心画点
        ax.plot(x_world, y_world, 'ro', markersize=2)
    # 绘制 GT 框（绿色）
    for box in gt_boxes:
        x_world, y_world = grid_to_world_gt(box['gx'], box['gy'])
        w, l, sin, cos = box['w'], box['l'], box['sin'], box['cos']
        yaw = np.arctan2(sin, cos)
        rect = patches.Rectangle(
            (x_world - w/2, y_world - l/2), w, l,
            angle=np.rad2deg(yaw),
            linewidth=2, edgecolor='green', facecolor='none', alpha=0.8
        )
        ax.add_patch(rect)
        ax.plot(x_world, y_world, 'go', markersize=2)

    ax.set_title('BEV Visualization (Green: GT, Red: Pred)')
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved figure to {save_path}")
    else:
        plt.show()
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.patches import Rectangle
import matplotlib.patches as patches
def draw_3d_boxes(ax, boxes, color='blue', label=None,show_label=True):
    """在3D轴上绘制3D框（使用中心高度）"""
    if not boxes:
        print(f"⚠️ 没有要绘制的 {label} 框")
        return None

    first_handle = None
    for idx, box in enumerate(boxes):
        x, y = grid_to_world(box['gx'], box['gy'])
        center_z = box['z']          # 中心高度
        w, l, h = box['w'], box['l'], box['h']
        yaw = np.arctan2(box['sin'], box['cos'])

        # 底部和顶部Z坐标
        bottom_z = center_z - h/2
        top_z = center_z + h/2

        # 底部角点 (z=bottom_z)
        corners_bottom = np.array([
            [-l/2, -w/2, 0],
            [ l/2, -w/2, 0],
            [ l/2,  w/2, 0],
            [-l/2,  w/2, 0]
        ])
        rot = np.array([[np.cos(yaw), -np.sin(yaw), 0],
                        [np.sin(yaw),  np.cos(yaw), 0],
                        [0, 0, 1]])
        corners_bottom = corners_bottom @ rot.T
        corners_bottom[:, 0] += x
        corners_bottom[:, 1] += y
        corners_bottom[:, 2] += bottom_z

        # 顶部角点
        corners_top = corners_bottom.copy()
        corners_top[:, 2] += h   # 顶部 = 底部 + h

        # 绘制底部和顶部矩形（透明填充）
        from mpl_toolkits.mplot3d.art3d import Poly3DCollection
        bottom = Poly3DCollection([corners_bottom], facecolor=color, alpha=0.2, edgecolor=color, linewidth=1)
        top = Poly3DCollection([corners_top], facecolor=color, alpha=0.2, edgecolor=color, linewidth=1)
        ax.add_collection3d(bottom)
        ax.add_collection3d(top)

        # 绘制竖直边
        for i in range(4):
            line, = ax.plot([corners_bottom[i,0], corners_top[i,0]],
                            [corners_bottom[i,1], corners_top[i,1]],
                            [corners_bottom[i,2], corners_top[i,2]], color=color, linewidth=1)
            if idx == 0 and i == 0:
                first_handle = line
        # 添加类别标签（在框顶部上方）
        if show_label:
            class_name = id_to_name.get(box['class_idx'], f"cls{box['class_idx']}")
            ax.text(x, y, top_z + 0.3, class_name, color=color, fontsize=8, ha='center', va='bottom')

    if first_handle is None and boxes:
        first_handle = ax.plot([0], [0], [0], color=color)[0]

    if label is not None and first_handle is not None:
        first_handle.set_label(label)

    return first_handle
# ========== 主函数 ==========
def main():
    # 1. 加载模型
    BEST_MODEL_PATH = 'best_model.npz'
    if not os.path.exists(BEST_MODEL_PATH):
        print("⚠️ 未找到 best_model.npz，使用随机初始化")
        model_params = init_model_params(
            backbone_in=3,
            backbone_out=64,
            depth_bins=DEPTH_BINS,
            context_channels=64,
            bev_channels=64,
            bev_shape=BEV_SIZE,
            num_classes=len(class_mapping)
        )
    else:
        data = np.load(BEST_MODEL_PATH, allow_pickle=True)
        model_params = data['model_params'].item()
        print("✅ 加载 best_model.npz")

    # 2. 选择一个样本
    samples = load_json('v1.0-mini/sample.json')
    sample_tokens = [s['token'] for s in samples[:20]]  # 可修改为全部或指定
    instances = load_json('v1.0-mini/instance.json')
    categories = load_json('v1.0-mini/category.json')
    instance_to_category = {inst['token']: inst['category_token'] for inst in instances}
    cat_name_map = {cat['token']: cat['name'] for cat in categories}

    save_dir = "bev_vis_results"
    os.makedirs(save_dir, exist_ok=True)
    
    # 假设您循环处理多个样本（这里以单个样本为例，但可扩展）
    sample_index = 0   # 或使用 enumerate

    H, W = 256, 704
    Hf, Wf = H // 32, W // 32
    # 循环处理每个样本
    for sample_token in tqdm(sample_tokens[:3], desc="Processing samples"):
        # 3. 构建几何索引
        geom_indices = build_geometry_indices(sample_token, model_params, H, W, Hf, Wf)
        geom_indices_batch = [ geom_indices]
        # 4. 加载图像和标定
        img_list, calib_list, ego_pose_mat = load_sample_data(sample_token)
        images = np.stack([img.transpose(2,0,1) for img in img_list], axis=0)[None, ...]

        # 5. 前向
        heatmap, reg, depth_logits_list, bev_feat, caches = bev_forward(
            images, geom_indices_batch, BEV_SIZE, model_params
        )
        hm_logit = sigmoid(heatmap)
        # 6. 解码预测框
        preds = decode_heatmap(hm_logit, reg, threshold=0.5, topk=30)
        print(f"预测到 {len(preds)} 个目标")

        # 7. 获取 GT 框
        instances = load_json('v1.0-mini/instance.json')
        categories = load_json('v1.0-mini/category.json')
        instance_to_category = {inst['token']: inst['category_token'] for inst in instances}
        cat_name_map = {cat['token']: cat['name'] for cat in categories}
        anns = get_annotations(sample_token)
        heatmap_gt = generate_heatmap_gt(anns, ego_pose_mat, BEV_SIZE, class_mapping, instance_to_category, cat_name_map, sigma=3.0)
        reg_gt = generate_reg_gt(anns, ego_pose_mat, BEV_SIZE)
        gt_boxes = decode_heatmap(heatmap_gt, reg_gt, threshold=0.5)
        print(f"GT 目标数: {len(gt_boxes)}")
        # 8. 打印详细信息（前10个）
        print("\n--- 预测框 ---")
        for i, p in enumerate(preds[:10]):
            x_w, y_w = grid_to_world(p['gx'], p['gy'])
            yaw = np.arctan2(p['sin'], p['cos'])
            class_name = id_to_name.get(p['class_idx'], f"cls{p['class_idx']}")
            print(f"  {i}: {class_name}, score={p['score']:.3f}, pos=({x_w:.2f},{y_w:.2f},{p['z']:.2f}), size=({p['w']:.2f},{p['l']:.2f},{p['h']:.2f}), yaw={yaw:.2f} rad")

        print("\n--- GT 框 ---")
        for i, g in enumerate(gt_boxes[:10]):
            x_w, y_w = grid_to_world(g['gx'], g['gy'])
            yaw = np.arctan2(g['sin'], g['cos'])
            class_name = id_to_name.get(g['class_idx'], f"cls{g['class_idx']}")
            print(f"  {i}: {class_name}, pos=({x_w:.2f},{y_w:.2f},{g['z']:.2f}), size=({g['w']:.2f},{g['l']:.2f},{g['h']:.2f}), yaw={yaw:.2f} rad")

        # 9. 可视化
        save_path = os.path.join(save_dir, f"sample_{sample_index:04d}.png")
        visualize_bev(gt_boxes, preds, save_path=save_path)
        sample_index+=1        
        """
        #visualize_bev(gt_boxes, preds, save_path='bev_vis.png')
        # 在 main() 中调用

        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection='3d')
        # 设置视角
        ax.view_init(elev=30, azim=-60)
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.set_zlabel('Z (m)')
        ax.set_title('3D Object Detection Results')
        # 设置坐标轴范围（根据BEV范围调整）
        ax.set_xlim(-50, 50)
        ax.set_ylim(-50, 50)
        ax.set_zlim(0, 100)
        ax.set_box_aspect([1,1,1])
        # 绘制预测框（红色）和GT框（绿色）
        #draw_3d_boxes(ax, preds[:10], color='red', label='Pred')
        draw_3d_boxes(ax, gt_boxes[:10], color='green', label='GT')
        plt.legend()
        # 显示图像并等待用户操作
        plt.show(block=False)  # 非阻塞显示
        input("按 Enter 键继续下一个样本...")
        plt.close(fig)  # 关闭当前图像
        """

if __name__ == "__main__":
    main()
