#include <stdio.h> // 标准IO，仅用于调试
#include <assert.h> // 断言库
#include "ATen/ATen.h" // 引入 ATen 张量头文件

typedef at::Half bf16; // 使用半精度 (fp16) 近似模拟 bf16 (可根据 PyTorch 版本切换)
// typedef at::BFloat16 bf16; // 亦可直接使用 BFloat16

template <typename F>
__global__ void kernel_forward(const int B, const int T, const int C, const int H,
                               float *__restrict__ _state, const F *__restrict__ const _r, const F *__restrict__ const _w, const F *__restrict__ const _k, const F *__restrict__ const _v, const F *__restrict__ const _a, const F *__restrict__ const _b,
                               F *__restrict__ const _y) // 前向计算 kernel
{
    const int e = blockIdx.x / H; // e 表示 batch *?? 此实现仅支持 B=1
    const int h = blockIdx.x % H; // 当前 head 索引
    const int i = threadIdx.x; // 通道索引 (0.._N_-1)
    _state += h*_N_*_N_ + i*_N_; // 将状态指针移动到当前 head & channel 所在的矩阵位置 (注意: B>1 时不正确)

    float state[_N_]; // 本线程寄存器中保存 _N_ ×1 的 state 向量
    #pragma unroll
    for (int j = 0; j < _N_; j++)
        state[j] = _state[j]; // 初始化 state

    __shared__ float r[_N_], k[_N_], w[_N_], a[_N_], b[_N_]; // 共享内存缓存当前时间步特征

    for (int _t = 0; _t < T; _t++) // 遍历序列长度 T
    {
        const int t = e*T*C + h*_N_ + i + _t * C; // 计算一维索引
        __syncthreads();
        r[i] = float(_r[t]); // 读取 r 并转换为 float
        w[i] = __expf(-__expf(float(_w[t]))); // 读取 w 并应用两次 exp 变换
        k[i] = float(_k[t]); // 读取 k
        a[i] = float(_a[t]); // 读取 a
        b[i] = float(_b[t]); // 读取 b
        __syncthreads();

        float sa = 0; // 计算 a·state
        #pragma unroll
        for (int j = 0; j < _N_; j++)
        {
            sa += a[j] * state[j];
        }

        float vv = float(_v[t]); // 读取 v
        float y = 0; // 输出
        #pragma unroll
        for (int j = 0; j < _N_; j++)
        {
            float& s = state[j]; // 引用 state 元素
            s = s * w[j] + k[j] * vv + sa * b[j]; // 更新状态
            y += s * r[j]; // 累积输出
        }
        _y[t] = F(y); // 写回输出 (自动转换为F)
    }
    #pragma unroll
    for (int j = 0; j < _N_; j++)
        _state[j] = state[j];    // 写回最终 state   
}

void cuda_forward(int B, int T, int C, int H, float *state, bf16 *r, bf16* w, bf16 *k, bf16 *v, bf16 *a, bf16 *b, bf16 *y)
{
    assert(H*_N_ == C); // 断言通道数匹配
    assert(B == 1); // 简化实现仅支持 B=1
    kernel_forward<<<dim3(B * H), dim3(_N_)>>>(B, T, C, H, state, r, w, k, v, a, b, y); // 启动 kernel
}
