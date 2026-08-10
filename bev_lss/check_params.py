import numpy as np
def check_bev_params(params, prefix='bev_encoder'):
    """打印 bev_encoder 各层权重的均值和范数"""
    d = params[prefix]
    for stage_key in sorted(d.keys()):
        stage = d[stage_key]
        for block_key in sorted(stage.keys()):
            block = stage[block_key]
            for param_name, arr in block.items():
                if 'b' in param_name:  # 只观察权重（也可改为 'gamma'）
                    print(f"{prefix}.{stage_key}.{block_key}.{param_name}: "
                          f"mean={np.abs(arr).sum():.6f}, norm={np.linalg.norm(arr):.4f}")
