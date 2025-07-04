#include <torch/extension.h> // 引入PyTorch扩展 API
#include <cuda_bf16.h> // 引入CUDA bfloat16 相关头文件
using bf = __nv_bfloat16; // 为CUDA bfloat16 类型取别名 bf

// 声明在 wkv7_cuda.cu 中实现的前向 CUDA 函数接口
void cuda_forward(int B, int T, int H, bf*w, bf*q, bf*k, bf*v, bf*z, bf*a, bf*y, float*s, float*sa);

// Python -> C++ 前向桥接函数，将 PyTorch 张量信息解包后调用 CUDA 实现
void forward(torch::Tensor &w, torch::Tensor &q, torch::Tensor &k, torch::Tensor &v, torch::Tensor &z, torch::Tensor &a, torch::Tensor &y, torch::Tensor &s, torch::Tensor &sa) {
    int B = w.sizes()[0], T = w.sizes()[1], H = w.sizes()[2]; // 从张量尺寸推断批次、序列长度、头数
    cuda_forward(B, T, H, (bf*)w.data_ptr(), (bf*)q.data_ptr(), (bf*)k.data_ptr(), (bf*)v.data_ptr(), (bf*)z.data_ptr(), (bf*)a.data_ptr(), (bf*)y.data_ptr(), (float*)s.data_ptr(), (float*)sa.data_ptr()); // 调用 CUDA 内核
}

// 声明在 wkv7_cuda.cu 中实现的反向 CUDA 函数接口
void cuda_backward(int B, int T, int H, bf*w, bf*q, bf*k, bf*v, bf*z, bf*a, bf*dy, float*s, float*sa, bf*dw, bf*dq, bf*dk, bf*dv, bf*dz, bf*da);

// Python -> C++ 反向桥接函数，负责将梯度张量传递给 CUDA 实现
void backward(torch::Tensor &w, torch::Tensor &q, torch::Tensor &k, torch::Tensor &v, torch::Tensor &z, torch::Tensor &a, torch::Tensor &dy,
        torch::Tensor &s, torch::Tensor &sa, torch::Tensor &dw, torch::Tensor &dq, torch::Tensor &dk, torch::Tensor &dv, torch::Tensor &dz, torch::Tensor &da) {
    int B = w.sizes()[0], T = w.sizes()[1], H = w.sizes()[2]; // 尺寸推断
    cuda_backward(B, T, H, (bf*)w.data_ptr(), (bf*)q.data_ptr(), (bf*)k.data_ptr(), (bf*)v.data_ptr(), (bf*)z.data_ptr(), (bf*)a.data_ptr(), (bf*)dy.data_ptr(), 
            (float*)s.data_ptr(), (float*)sa.data_ptr(), (bf*)dw.data_ptr(), (bf*)dq.data_ptr(), (bf*)dk.data_ptr(), (bf*)dv.data_ptr(), (bf*)dz.data_ptr(), (bf*)da.data_ptr()); // 调用 CUDA 反向内核
}

// 将 forward / backward 函数注册为 wind_backstepping PyTorch 扩展中的 op
torch::Library::Options options;
TORCH_LIBRARY(wind_backstepping, m) {
    m.def("forward(Tensor w, Tensor q, Tensor k, Tensor v, Tensor z, Tensor a, Tensor(a!) y, Tensor(b!) s, Tensor(c!) sa) -> ()"); // 前向 op 签名
    m.def("backward(Tensor w, Tensor q, Tensor k, Tensor v, Tensor z, Tensor a, Tensor dy, Tensor s, Tensor sa, Tensor(a!) dw, Tensor(b!) dq, Tensor(c!) dk, Tensor(d!) dv, Tensor(e!) dz, Tensor(f!) da) -> ()"); // 反向 op 签名
}

// 指定在 CUDA 后端下的实现映射
torch::Library::Options options2;
TORCH_LIBRARY_IMPL(wind_backstepping, CUDA, m) {
    m.impl("forward", &forward); // 绑定 forward
    m.impl("backward", &backward); // 绑定 backward
}
