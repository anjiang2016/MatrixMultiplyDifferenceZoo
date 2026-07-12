#include <metal_stdlib>
using namespace metal;

// 使用 Metal 4 的 tensor 类型进行矩阵乘法
// 每个线程组处理一个 TileSize x TileSize 的输出块
kernel void matrix_multiplication_kernel(
    uint2 threadgroupIdentifier [[threadgroup_position_in_grid]],
    tensor<device half, dextents<int, 2>> sourceA,
    tensor<device half, dextents<int, 2>> sourceB,
    tensor<device half, dextents<int, 2>> destination
) {
    const int TileSize = 64;
    int tileOriginX = TileSize * threadgroupIdentifier.x;
    int tileOriginY = TileSize * threadgroupIdentifier.y;
    
    // 切片：每个线程组负责一个 tile
    auto sliceA = sourceA.slice<dynamic_extent, dynamic_extent>(
        tileOriginX, tileOriginY,
        TileSize, TileSize
    );
    auto sliceB = sourceB.slice<dynamic_extent, dynamic_extent>(
        tileOriginY, tileOriginX,
        TileSize, TileSize
    );
    auto sliceC = destination.slice<dynamic_extent, dynamic_extent>(
        tileOriginX, tileOriginY,
        TileSize, TileSize
    );
    
    // Metal 4 张量运算：一次完成矩阵乘法
    // 结果直接写入 destination 的对应 tile
    sliceC = sliceA * sliceB;
}
