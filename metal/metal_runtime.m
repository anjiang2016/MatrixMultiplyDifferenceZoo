#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import <MetalPerformanceShaders/MetalPerformanceShaders.h>
#import <stdio.h>
#import <stdlib.h>

static id<MTLDevice> device = nil;
static id<MTLCommandQueue> commandQueue = nil;

void metal_init(void) {
    printf("[LOG] metal_init: entered\n");
    fflush(stdout);
    if (device) return;

    device = MTLCreateSystemDefaultDevice();
    printf("[LOG] metal_init: device = %p\n", (void*)device);
    fflush(stdout);
    if (!device) {
        printf("[LOG] metal_init: device is nil\n");
        fflush(stdout);
        return;
    }

    commandQueue = [device newCommandQueue];
    printf("[LOG] metal_init: commandQueue = %p\n", (void*)commandQueue);
    fflush(stdout);
}

int metal_is_available(void) {
    if (!device) metal_init();
    return (device != nil && commandQueue != nil);
}

void metal_matmul(const float* A, const float* B, float* C,
                  unsigned int M, unsigned int N, unsigned int K,
                  unsigned int* elapsed_ms) {
    printf("[LOG] metal_matmul: entered (M=%u, N=%u, K=%u)\n", M, N, K);
    fflush(stdout);

    if (!device || !commandQueue) {
        printf("[LOG] metal_matmul: Metal not available\n");
        fflush(stdout);
        if (elapsed_ms) *elapsed_ms = 0;
        return;
    }

    // 确保矩阵维度能被 MPS 处理（MPS 要求矩阵为行主序，且数据连续）
    // 如果 M 或 N 为 0，直接返回
    if (M == 0 || N == 0 || K == 0) {
        printf("[LOG] metal_matmul: zero dimension\n");
        fflush(stdout);
        if (elapsed_ms) *elapsed_ms = 0;
        return;
    }

    // 创建 MPS 矩阵描述符（行主序，float 类型）
    MPSMatrixDescriptor* descA = [MPSMatrixDescriptor
        matrixDescriptorWithRows:M columns:K
        rowBytes:K * sizeof(float)
        dataType:MPSDataTypeFloat32];
    MPSMatrixDescriptor* descB = [MPSMatrixDescriptor
        matrixDescriptorWithRows:K columns:N
        rowBytes:N * sizeof(float)
        dataType:MPSDataTypeFloat32];
    MPSMatrixDescriptor* descC = [MPSMatrixDescriptor
        matrixDescriptorWithRows:M columns:N
        rowBytes:N * sizeof(float)
        dataType:MPSDataTypeFloat32];

    // 创建 MPS 矩阵对象
    id<MTLBuffer> bufferA = [device newBufferWithBytes:A length:M * K * sizeof(float) options:MTLResourceStorageModeShared];
    id<MTLBuffer> bufferB = [device newBufferWithBytes:B length:K * N * sizeof(float) options:MTLResourceStorageModeShared];
    id<MTLBuffer> bufferC = [device newBufferWithBytes:C length:M * N * sizeof(float) options:MTLResourceStorageModeShared];
    if (!bufferA || !bufferB || !bufferC) {
        printf("[LOG] metal_matmul: failed to create buffers\n");
        fflush(stdout);
        if (elapsed_ms) *elapsed_ms = 0;
        return;
    }

    MPSMatrix* matrixA = [[MPSMatrix alloc] initWithBuffer:bufferA descriptor:descA];
    MPSMatrix* matrixB = [[MPSMatrix alloc] initWithBuffer:bufferB descriptor:descB];
    MPSMatrix* matrixC = [[MPSMatrix alloc] initWithBuffer:bufferC descriptor:descC];

    // 创建 MPS 矩阵乘法对象（alpha=1, beta=0）
    MPSMatrixMultiplication* mul = [[MPSMatrixMultiplication alloc]
        initWithDevice:device
        transposeLeft:NO
        transposeRight:NO
        resultRows:M
        resultColumns:N
        interiorColumns:K
        alpha:1.0
        beta:0.0];

    // 创建命令缓冲区
    id<MTLCommandBuffer> commandBuffer = [commandQueue commandBuffer];
    if (!commandBuffer) {
        printf("[LOG] metal_matmul: failed to create command buffer\n");
        fflush(stdout);
        if (elapsed_ms) *elapsed_ms = 0;
        return;
    }

    // 编码矩阵乘法
    [mul encodeToCommandBuffer:commandBuffer
                 leftMatrix:matrixA
                rightMatrix:matrixB
               resultMatrix:matrixC];

    printf("[LOG] metal_matmul: encoding done, committing...\n");
    fflush(stdout);

    CFAbsoluteTime start = CFAbsoluteTimeGetCurrent();
    [commandBuffer commit];
    [commandBuffer waitUntilCompleted];
    CFAbsoluteTime end = CFAbsoluteTimeGetCurrent();

    if (commandBuffer.error) {
        printf("[LOG] metal_matmul: command buffer error: %s\n",
               [[commandBuffer.error localizedDescription] UTF8String]);
        fflush(stdout);
        if (elapsed_ms) *elapsed_ms = 0;
        return;
    }

    if (elapsed_ms) {
        *elapsed_ms = (uint)((end - start) * 1000);
    }
    printf("[LOG] metal_matmul: command executed, elapsed=%u ms\n",
           elapsed_ms ? *elapsed_ms : 0);
    fflush(stdout);

    // 复制结果回 C（MPS 已经将结果写入 bufferC）
    void* cptr = [bufferC contents];
    if (cptr) {
        memcpy(C, cptr, M * N * sizeof(float));
        printf("[LOG] metal_matmul: result copied back\n");
    } else {
        printf("[LOG] metal_matmul: bufferC contents is NULL\n");
    }
    fflush(stdout);
}
