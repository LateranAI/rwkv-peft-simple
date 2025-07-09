#include <cuda_bf16.h> // 引入CUDA对bfloat16数据类型支持的头文件
#include <assert.h> // 引入断言库，用于在调试阶段验证条件

using bf = __nv_bfloat16; // 将CUDA内置的bfloat16类型重新命名为bf，方便后续使用
__device__ inline float to_float(const bf & u) { return __bfloat162float(u); } // 设备函数：将bf16转换为32位浮点
__device__ inline bf to_bf(const float & u) { return __float2bfloat16_rn(u); } // 设备函数：将32位浮点数转换为bf16并四舍五入

typedef bf * __restrict__ F_; // 定义指针别名F_，并使用__restrict__提升编译器优化

__global__ void forward_kernel(int T, int H, F_ w_, F_ q_, F_ k_, F_ v_, F_ a_, F_ b_, bf* y_, float* s_, float* sa_) { // 前向核函数：计算RWKV7前向传播
    constexpr int C = _C_; // 每个头中隐藏维度的大小，由编译期宏注入
    int bb = blockIdx.y, hh = blockIdx.x, i = threadIdx.x; // bb: batch索引, hh: head索引, i: 通道索引

    float state[C] = {0}; // 本线程维护的状态向量
    __shared__ float q[C], k[C], w[C], a[C], b[C]; // 共享内存用于存储当前时间步的q,k,w,a,b

    for (int t = 0; t < T; t++) { // 遍历序列长度T
        int ind = bb*T*H*C + t*H*C + hh * C + i; // 计算当前元素在扁平张量中的索引
        __syncthreads(); // 同步所有线程以确保共享内存上一时刻写入已完成
        q[i] = to_float(q_[ind]); // 读取bf16并转换为float
        w[i] = __expf(-__expf(to_float(w_[ind]))); // 计算w = exp(-exp(raw_w))
        k[i] = to_float(k_[ind]); // 读取k
        a[i] = to_float(a_[ind]); // 读取a
        b[i] = to_float(b_[ind]); // 读取b
        __syncthreads(); // 确保共享内存写完再开始计算

        float sa = 0; // 累积变量: a * state 的内积
#pragma unroll
        for (int j = 0; j < C; j++) { // 遍历通道
            sa += a[j] * state[j]; // 计算sa
        }
        sa_[ind] = sa; // 将sa写到全局内存供反向传播使用

        float v = to_float(v_[ind]); // 读取v并转换为float
        float y = 0; // 当前输出
#pragma unroll
        for (int j = 0; j < C; j++) { // 更新状态并累积输出
            float& s = state[j]; // 取得引用便于就地更新
            s = s * w[j] + sa * b[j] + k[j] * v; // 状态更新公式
            y += s * q[j]; // 输出累加
        }
        y_[ind] = to_bf(y); // 将输出转换为bf16写回

        if ((t+1)%_CHUNK_LEN_ == 0) { // 每处理CHUNK_LEN步就快照一次状态
            int base = (bb*H+hh)*(T/_CHUNK_LEN_)*C*C + (t/_CHUNK_LEN_)*C*C + i; // 计算写入基址
#pragma unroll
            for (int j = 0; j < C; j++) { // 将state矩阵写入s_
                s_[base + j*C] = state[j];
            }
        }
    }
}

__global__ void backward_kernel(int T, int H, F_ w_, F_ q_, F_ k_, F_ v_, F_ a_, F_ b_, F_ dy_, float * __restrict__ s_, float * __restrict__ sa_, bf* dw_, bf* dq_, bf* dk_, bf* dv_, bf* da_, bf* db_) { // 反向核函数
    constexpr int C = _C_; // 通道维度大小
    int bb = blockIdx.y, hh = blockIdx.x, i = threadIdx.x; // 同前向

    float stateT[C] = {0}, dstate[C] = {0}, dstateT[C] = {0}; // 保存快照状态及其梯度
    __shared__ float w[C], q[C], k[C], v[C], a[C], b[C], dy[C], sa[C], dSb_shared[C]; // 共享内存
    float qi, wi, ki, ai, bi, dyi; // 局部寄存器变量

    for (int t = T-1; t >= 0; t--) { // 反向遍历序列
        int ind = bb*T*H*C + t*H*C + hh * C + i; // 扁平索引
        __syncthreads(); // 同步
        q[i] = qi = to_float(q_[ind]); // 读取q
        float wi_fac = -__expf(to_float(w_[ind])); // 计算dw所需的exp(-exp(w))
        w[i] = wi = __expf(wi_fac); // w = exp(-exp(raw_w))
        k[i] = ki = to_float(k_[ind]); // 读取k
        a[i] = ai = to_float(a_[ind]); // a
        b[i] = bi = to_float(b_[ind]); // b
        v[i] = to_float(v_[ind]); // v
        dy[i] = dyi = to_float(dy_[ind]); // 梯度dy
        sa[i] = sa_[ind]; // 读取前向保存的sa
        __syncthreads(); // 确保共享内存写完

        if ((t+1)%_CHUNK_LEN_ == 0) { // 读取状态快照
            int base = (bb*H+hh)*(T/_CHUNK_LEN_)*C*C + (t/_CHUNK_LEN_)*C*C + i*C; // 计算基址
#pragma unroll
            for (int j = 0; j < C; j++) { // 读入stateT
                stateT[j] = s_[base + j];
            }
        }

        float dq = 0; // 梯度dq
#pragma unroll
        for (int j = 0; j < C; j++) {
            dq += stateT[j]*dy[j]; // dq = stateT · dy
        }
        dq_[ind] = to_bf(dq); // 写dq

        float iwi = 1.0f/wi; // 预计算1/w
#pragma unroll        
        for (int j = 0; j < C; j++) { // 反推前一状态
            stateT[j] = (stateT[j] - ki*v[j] - bi*sa[j]) * iwi;
            dstate[j] += dyi * q[j]; // 梯度累加
            dstateT[j] += qi * dy[j];
        }

        float dw = 0, dk = 0, dv = 0, db = 0, dSb = 0; // 梯度累积变量
#pragma unroll
        for (int j = 0; j < C; j++) {
            dw += dstateT[j]*stateT[j];
            dk += dstateT[j]*v[j];
            dv += dstate[j]*k[j];
            dSb += dstate[j]*b[j];
            db += dstateT[j]*sa[j];
        }
        dw_[ind] = to_bf(dw * wi * wi_fac); // 写dw
        dk_[ind] = to_bf(dk); // 写dk
        dv_[ind] = to_bf(dv); // 写dv
        db_[ind] = to_bf(db); // 写db

        __syncthreads(); // 准备共享dSb
        dSb_shared[i] = dSb; // 每线程写入局部dSb
        __syncthreads(); // 使dSb_shared可见

        float da = 0; // 计算da
#pragma unroll
        for (int j = 0; j < C; j++) {
            da += stateT[j]*dSb_shared[j];
        }
        da_[ind] = to_bf(da); // 写da

#pragma unroll        
        for (int j = 0; j < C; j++) { // 传递梯度到前一状态
            dstate[j] = dstate[j]*w[j] + dSb * a[j];
            dstateT[j] = dstateT[j]*wi + ai * dSb_shared[j];
        }
    }
}

void cuda_forward(int B, int T, int H, bf*w, bf*q, bf*k, bf*v, bf*z, bf*a, bf*y, float*s, float*sa) { // C++接口：前向调用
    forward_kernel<<<dim3(H,B), dim3(_C_)>>>(T,H,w,q,k,v,z,a,y,s,sa); // 启动forward_kernel
}
void cuda_backward(int B, int T, int H, bf*w, bf*q, bf*k, bf*v, bf*z, bf*a, bf*dy, float*s, float*sa, bf*dw, bf*dq, bf*dk, bf*dv, bf*dz, bf*da) { // C++接口：反向调用
    assert(T%_CHUNK_LEN_ == 0); // 保证T是CHUNK_LEN的整数倍
    backward_kernel<<<dim3(H,B), dim3(_C_)>>>(T,H,w,q,k,v,z,a,dy,s,sa,dw,dq,dk,dv,dz,da); // 启动backward_kernel
}
