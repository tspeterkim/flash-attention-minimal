#include <torch/types.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <mma.h>
using namespace nvcuda;

const int WARP_SIZE = 32;

// WMMA for TF32 - 16x16x8
const int WMMA_M = 16;
const int WMMA_N = 16;
const int WMMA_K = 8;

__global__ void forward_kernel_wmma(const float* Q, const float* K, const float* V, const int N, const int d,
                                    const int Tc, const int Tr, const int Bc, const int Br, const float softmax_scale,
                                    float* l, float *m, float* O) {
    int tx = threadIdx.x;
    int bx = blockIdx.x; int by = blockIdx.y;  // batch and head index

    // Offset into Q,K,V,O,l,m - different for each batch and head
    int qkv_offset = (bx * gridDim.y * N * d) + (by * N * d);  // gridDim.y = nh
    int lm_offset = (bx * gridDim.y * N) + (by * N);  // offset for l and m

    // Define SRAM for Q,K,V,S
    extern __shared__ float sram[];
    int tile_size = Bc * d;  // size of Qi, Kj, Vj
    float* Qi = sram;
    float* Kj = &sram[tile_size];
    float* Vj = &sram[tile_size * 2];
    float* Sij = &sram[tile_size * 3];

    // Initialize l and m
    for (int x = 0; x < N; x += WARP_SIZE) {
        if (x + tx < N) {
            l[lm_offset + tx + x] = 0; m[lm_offset + tx + x] = -INFINITY;
        }
    }

    for (int j = 0; j < Tc; j++) {

        // Load Kj, Vj to SRAM
        for (int x = 0; x < tile_size; x += WARP_SIZE) {
            if (x + tx < tile_size) {
                Kj[x + tx] = K[qkv_offset + (tile_size * j) + x + tx]; // TF32 conversion for WMMA
                Vj[x + tx] = V[qkv_offset + (tile_size * j) + x + tx];
            }
        }
        __syncthreads();

        for (int i = 0; i < Tr; i++)  {

            // Load Qi to SRAM
            for (int x = 0; x < tile_size; x += WARP_SIZE) {
                if (x + tx < tile_size)
                    Qi[x + tx] = Q[qkv_offset + (tile_size * i) + x + tx]; 
            }
            __syncthreads(); 

            // Load l and m to registers
            float row_m_prev, row_l_prev;
            if (tx < Br) {
                row_m_prev = m[lm_offset + (Br * i) + tx];
                row_l_prev = l[lm_offset + (Br * i) + tx];
            }

            // S = QK^T - tensor cores going brrr
            wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, wmma::precision::tf32, wmma::row_major> q_frag;
            wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, wmma::precision::tf32, wmma::col_major> k_frag;  // note col_major for transpose
            wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> s_frag;
            wmma::fill_fragment(s_frag, 0.0f);

            for (int k = 0; k < d; k += WMMA_K) {
                wmma::load_matrix_sync(q_frag, Qi + k, d);
                wmma::load_matrix_sync(k_frag, Kj + k, d);
                wmma::mma_sync(s_frag, q_frag, k_frag, s_frag);
            }
            wmma::store_matrix_sync(Sij, s_frag, WMMA_M, wmma::mem_row_major);

            float row_m = -INFINITY; float row_l = 0; 
            if (tx < Br) {
                // Softmax scaling, row_m = rowmax(S)
                for (int x = 0; x < Bc; x++) {
                    Sij[(Bc * tx) + x] *= softmax_scale;
                    row_m = max(row_m, Sij[(Bc * tx) + x]);
                }

                // P = exp(S - row_m), row_l = rowsum(P)
                for (int x = 0; x < Bc; x++) {
                    Sij[(Bc * tx) + x] = __expf(Sij[(Bc * tx) + x] - row_m);
                    row_l += Sij[(Bc * tx) + x];
                }
            }

            // PV = Pij * Vj - tensor cores going brrr again
            wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, wmma::precision::tf32, wmma::row_major> p_frag;
            wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, wmma::precision::tf32, wmma::row_major> v_frag;
            wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> pv_frag;
            
            for (int x = 0; x < d; x += WMMA_M) {
                wmma::fill_fragment(pv_frag, 0.0f);
                for (int k = 0; k < Br; k += WMMA_K) {
                    wmma::load_matrix_sync(p_frag, Sij + k, Bc);
                    wmma::load_matrix_sync(v_frag, Vj + x + (k * d), d);
                    wmma::mma_sync(pv_frag, p_frag, v_frag, pv_frag);
                }
                wmma::store_matrix_sync(Qi + x, pv_frag, d, wmma::mem_row_major);  // store it in unused Qi
            }

            if (tx < Br) {
                // Compute new m and l
                float row_m_new = max(row_m_prev, row_m);
                float row_l_new = (__expf(row_m_prev - row_m_new) * row_l_prev) + (__expf(row_m - row_m_new) * row_l);

                // Write O, l, m to HBM
                for (int x = 0; x < d; x++) {
                    O[qkv_offset + (tile_size * i) + (tx * d) + x] = (1 / row_l_new) \
                        * ((row_l_prev * __expf(row_m_prev - row_m_new) * O[qkv_offset + (tile_size * i) + (tx * d) + x]) \
                        + (__expf(row_m - row_m_new) * Qi[(tx * d) + x]));
                }
                m[lm_offset + (Br * i) + tx] = row_m_new;
                l[lm_offset + (Br * i) + tx] = row_l_new;
            }
        }
        __syncthreads();
    }
}

__global__ void forward_kernel_naive(const float* Q, const float* K, const float* V, const int N, const int d,
                                     const int Tc, const int Tr, const int Bc, const int Br, const float softmax_scale,
                                     float* l, float* m, float* O) {
    int tx = threadIdx.x;
    int bx = blockIdx.x; int by = blockIdx.y;  // batch and head index

    // Offset into Q,K,V,O,l,m - different for each batch and head
    int qkv_offset = (bx * gridDim.y * N * d) + (by * N * d);  // gridDim.y = nh
    int lm_offset = (bx * gridDim.y * N) + (by * N);  // offset for l and m

    // Define SRAM for Q,K,V,S
    extern __shared__ float sram[];
    int tile_size = Bc * d;  // size of Qi, Kj, Vj
    float* Qi = sram;
    float* Kj = &sram[tile_size];
    float* Vj = &sram[tile_size * 2];
    float* S = &sram[tile_size * 3];

    // Initialize l and m
    for (int x = 0; x < N; x += WARP_SIZE) {
        if (x + tx < N) {
            l[lm_offset + tx + x] = 0; m[lm_offset + tx + x] = -INFINITY;
        }
    }

    for (int j = 0; j < Tc; j++) {

        // Load Kj, Vj to SRAM
        for (int x = 0; x < d; x++) {
            Kj[(tx * d) + x] = K[qkv_offset + (tile_size * j) + (tx * d) + x];
            Vj[(tx * d) + x] = V[qkv_offset + (tile_size * j) + (tx * d) + x];
        }
        __syncthreads();  // such that the inner loop can use the correct Kj, Vj

        for (int i = 0; i < Tr; i++)  {

            // Load Qi to SRAM, l and m to registers
            for (int x = 0; x < d; x++) {
                Qi[(tx * d) + x] = Q[qkv_offset + (tile_size * i) + (tx * d) + x];
            }
            float row_m_prev = m[lm_offset + (Br * i) + tx];
            float row_l_prev = l[lm_offset + (Br * i) + tx];

            // S = QK^T, row_m = rowmax(S)
            float row_m = -INFINITY;
            for (int y = 0; y < Bc; y++) {
                float sum = 0;
                for (int x = 0; x < d; x++) {
                    sum += Qi[(tx * d) + x] * Kj[(y * d) + x];
                }
                sum *= softmax_scale;
                S[(Bc * tx) + y] = sum;

                if (sum > row_m)
                    row_m = sum;
            }

            // P = exp(S - row_m), row_l = rowsum(P)
            float row_l = 0;
            for (int y = 0; y < Bc; y++) {
                S[(Bc * tx) + y] = __expf(S[(Bc * tx) + y] - row_m);
                row_l += S[(Bc * tx) + y];
            }

            // Compute new m and l
            float row_m_new = max(row_m_prev, row_m);
            float row_l_new = (__expf(row_m_prev - row_m_new) * row_l_prev) + (__expf(row_m - row_m_new) * row_l);

            // Write O, l, m to HBM
            for (int x = 0; x < d; x++) {
                float pv = 0;  // Pij * Vj
                for (int y = 0; y < Bc; y++) {
                    pv += S[(Bc * tx) + y] * Vj[(y * d) + x];
                }
                O[qkv_offset + (tile_size * i) + (tx * d) + x] = (1 / row_l_new) \
                    * ((row_l_prev * __expf(row_m_prev - row_m_new) * O[qkv_offset + (tile_size * i) + (tx * d) + x]) \
                    + (__expf(row_m - row_m_new) * pv));
            }
            m[lm_offset + (Br * i) + tx] = row_m_new;
            l[lm_offset + (Br * i) + tx] = row_l_new;
        }
        __syncthreads();  // otherwise, thread can use the wrong Kj, Vj in inner loop
    }
}

torch::Tensor forward(torch::Tensor Q, torch::Tensor K, torch::Tensor V, bool use_tensor_cores) {
    int Bc, Br;
    if (use_tensor_cores) {
        Bc = WMMA_M; Br = WMMA_M;  // Must be 16 to use WMMA for this kernel
    } else {
        Bc = 32; Br = 32;
    }

    const int B = Q.size(0); const int nh = Q.size(1);
    const int N = Q.size(2); const int d = Q.size(3);

    const int Tc = ceil((float) N / Bc); const int Tr = ceil((float) N / Br);
    const float softmax_scale = 1.0 / sqrt(d);

    // Initialize O, l, m to HBM
    auto O = torch::zeros_like(Q);
    // auto O = torch::empty_like(Q);
    // auto l = torch::zeros({B, nh, N});
    // auto m = torch::full({B, nh, N}, -INFINITY);
    // torch::Device device(torch::kCUDA);
    // l = l.to(device); m = m.to(device);

    float* l; float* m;
    cudaMalloc((void**)&l, B * nh * N * sizeof(float));
    cudaMalloc((void**)&m, B * nh * N * sizeof(float));

    // Calculate SRAM size needed per block
    const int sram_size = (3 * Bc * d * sizeof(float)) + (Bc * Br * sizeof(float));

    dim3 grid_dim(B, nh);  // batch_size x num_heads
    dim3 block_dim(WARP_SIZE);  // one warp per block

    if (use_tensor_cores) {
        forward_kernel_wmma<<<grid_dim, block_dim, sram_size>>>(
            Q.data_ptr<float>(), K.data_ptr<float>(), V.data_ptr<float>(),
            N, d, Tc, Tr, Bc, Br, softmax_scale,
            l, m, O.data_ptr<float>()
        );
    } else {
        forward_kernel_naive<<<grid_dim, block_dim, sram_size>>>(
            Q.data_ptr<float>(), K.data_ptr<float>(), V.data_ptr<float>(),
            N, d, Tc, Tr, Bc, Br, softmax_scale,
            l, m, O.data_ptr<float>()
        );
    }
    return O;
}