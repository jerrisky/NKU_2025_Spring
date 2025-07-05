
// //运行命令：
// //nvcc test.cu -o test -lcublas -std=c++11
// //./test

//目前表现最好的策略和参数的代码
#include <vector>
#include <cstring>
#include <string>
#include <iostream>
#include <fstream>
#include <set>
#include <iomanip>
#include <sstream>
#include <sys/time.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <queue>
#include <cublas_v2.h>
#include <algorithm>
#include <cmath>
#include <cfloat>
#include <chrono> // 添加高精度计时库

using namespace std;

// 检查CUDA和cuBLAS调用结果
#define CHECK_CUDA(call) do { \
    cudaError_t cudaStatus = call; \
    if (cudaStatus != cudaSuccess) { \
        fprintf(stderr, "CUDA error at %s:%d: %s\n", \
                __FILE__, __LINE__, cudaGetErrorString(cudaStatus)); \
        exit(EXIT_FAILURE); \
    } \
} while(0)

#define CHECK_CUBLAS(call) do { \
    cublasStatus_t status = call; \
    if (status != CUBLAS_STATUS_SUCCESS) { \
        fprintf(stderr, "cuBLAS error at %s:%d: %d\n", \
                __FILE__, __LINE__, status); \
        exit(EXIT_FAILURE); \
    } \
} while(0)

// 批量大小和线程块大小
const size_t BATCH_SIZE = 2000;
const size_t BLOCK_SIZE = 64;  // TopK核函数的线程块大小
const size_t CONV_BLOCK_SIZE = 8; // 转换核函数的线程块大小
const size_t TOPK = 10;

// 自定义交换函数（用于设备代码）
__device__ void swap_float(float &a, float &b) {
    float temp = a;
    a = b;
    b = temp;
}

__device__ void swap_uint(uint32_t &a, uint32_t &b) {
    uint32_t temp = a;
    a = b;
    b = temp;
}

// 将内积转换为距离的核函数
__global__ void convert_to_distance(float* d_P, float* d_D, size_t n, size_t m) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x; // base index
    size_t idy = blockIdx.y * blockDim.y + threadIdx.y; // query index
    if (idx < n && idy < m) {
        size_t p_index = idx + idy * n; // column-major index for d_P
        size_t d_index = idy * n + idx; // row-major index for d_D
        d_D[d_index] = 1.0f - d_P[p_index];
    }
}

// 每个线程块处理一个查询的TopK核函数
__global__ void topk_kernel(float* distances, uint32_t* indices, size_t base_number, size_t k, size_t query_num) {
    extern __shared__ float sdata[]; // 共享内存用于存储距离和索引
    
    // 当前处理的查询索引
    size_t query_idx = blockIdx.x;
    if (query_idx >= query_num) return;
    
    float* dist_ptr = distances + query_idx * base_number;
    uint32_t* idx_ptr = indices + query_idx * k;
    
    // 每个线程的私有TopK数组（距离+索引）
    float* local_dists = sdata;
    uint32_t* local_idxs = (uint32_t*)(local_dists + k * blockDim.x);
    
    // 初始化私有TopK
    size_t tid = threadIdx.x;
    for (size_t i = 0; i < k; i++) {
        local_dists[tid * k + i] = FLT_MAX;
        local_idxs[tid * k + i] = 0;
    }
    
    // 计算每个线程处理的数据范围
    size_t items_per_thread = (base_number + blockDim.x - 1) / blockDim.x;
    size_t start = tid * items_per_thread;
    size_t end = min(start + items_per_thread, base_number);
    
    // 局部TopK筛选
    for (size_t i = start; i < end; i++) {
        float dist = dist_ptr[i];
        
        // 如果比当前堆顶小，则替换堆顶并调整堆
        if (dist < local_dists[tid * k]) {
            // 替换堆顶
            local_dists[tid * k] = dist;
            local_idxs[tid * k] = i;
            
            // 调整堆（下沉）
            size_t cur = 0;
            while (cur < k) {
                size_t left = 2 * cur + 1;
                size_t right = 2 * cur + 2;
                size_t smallest = cur;
                
                if (left < k && local_dists[tid * k + left] > local_dists[tid * k + smallest]) 
                    smallest = left;
                if (right < k && local_dists[tid * k + right] > local_dists[tid * k + smallest])
                    smallest = right;
                
                if (smallest != cur) {
                    swap_float(local_dists[tid * k + cur], local_dists[tid * k + smallest]);
                    swap_uint(local_idxs[tid * k + cur], local_idxs[tid * k + smallest]);
                    cur = smallest;
                } else {
                    break;
                }
            }
        }
    }
    __syncthreads();
    
    // 合并所有线程的局部TopK
    if (tid == 0) {
        // 初始化全局TopK
        float global_dists[TOPK];
        uint32_t global_idxs[TOPK];
        for (size_t i = 0; i < k; i++) {
            global_dists[i] = FLT_MAX;
            global_idxs[i] = 0;
        }
        
        // 合并所有线程的局部结果
        for (size_t t = 0; t < blockDim.x; t++) {
            for (size_t i = 0; i < k; i++) {
                float dist = local_dists[t * k + i];
                uint32_t idx = local_idxs[t * k + i];
                
                // 如果比当前堆顶小，则插入堆
                if (dist < global_dists[0]) {
                    // 替换堆顶
                    global_dists[0] = dist;
                    global_idxs[0] = idx;
                    
                    // 调整堆（下沉）
                    size_t cur = 0;
                    while (cur < k) {
                        size_t left = 2 * cur + 1;
                        size_t right = 2 * cur + 2;
                        size_t largest = cur;
                        
                        if (left < k && global_dists[left] > global_dists[largest])
                            largest = left;
                        if (right < k && global_dists[right] > global_dists[largest])
                            largest = right;
                        
                        if (largest != cur) {
                            swap_float(global_dists[cur], global_dists[largest]);
                            swap_uint(global_idxs[cur], global_idxs[largest]);
                            cur = largest;
                        } else {
                            break;
                        }
                    }
                }
            }
        }
        
        // 对TopK结果排序（从小到大）
        for (size_t i = k - 1; i > 0; i--) {
            swap_float(global_dists[0], global_dists[i]);
            swap_uint(global_idxs[0], global_idxs[i]);
            
            // 调整堆（下沉）
            size_t cur = 0;
            while (cur < i) {
                size_t left = 2 * cur + 1;
                size_t right = 2 * cur + 2;
                size_t largest = cur;
                
                if (left < i && global_dists[left] > global_dists[largest])
                    largest = left;
                if (right < i && global_dists[right] > global_dists[largest])
                    largest = right;
                
                if (largest != cur) {
                    swap_float(global_dists[cur], global_dists[largest]);
                    swap_uint(global_idxs[cur], global_idxs[largest]);
                    cur = largest;
                } else {
                    break;
                }
            }
        }
        
        // 写入最终结果
        for (size_t i = 0; i < k; i++) {
            idx_ptr[i] = global_idxs[i];
        }
    }
}

// Search结果结构体
struct SearchResult {
    float recall;
    int64_t latency; // 单位us
};

// 优化后的GPU批量搜索函数（包含TopK）
SearchResult flat_search_batch_gpu(float* base, float* queries, 
                                  uint32_t* h_topk_indices,
                                  size_t base_number, size_t vecdim, size_t query_num,
                                  bool use_precomputed_transpose = false) {
    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));
    CHECK_CUDA(cudaEventRecord(start, 0));
    
    float *d_base = nullptr, *d_queries = nullptr, *d_P = nullptr, *d_D = nullptr;
    uint32_t* d_topk_indices = nullptr;
    cublasHandle_t handle;
    CHECK_CUBLAS(cublasCreate(&handle));

    CHECK_CUDA(cudaMalloc((void**)&d_base, base_number * vecdim * sizeof(float)));
    CHECK_CUDA(cudaMalloc((void**)&d_queries, query_num * vecdim * sizeof(float)));
    CHECK_CUDA(cudaMalloc((void**)&d_P, base_number * query_num * sizeof(float)));
    CHECK_CUDA(cudaMalloc((void**)&d_D, base_number * query_num * sizeof(float)));
    CHECK_CUDA(cudaMalloc((void**)&d_topk_indices, query_num * TOPK * sizeof(uint32_t)));

    cudaDeviceProp prop;
    CHECK_CUDA(cudaGetDeviceProperties(&prop, 0));
    bool use_tensor_core = (prop.major >= 7);
    
    if (use_tensor_core) {
        CHECK_CUBLAS(cublasSetMathMode(handle, CUBLAS_TENSOR_OP_MATH));
    }

    float* transposed_base = nullptr;
    if (use_precomputed_transpose) {
        transposed_base = new float[base_number * vecdim];
        for (size_t i = 0; i < vecdim; i++) {
            for (size_t j = 0; j < base_number; j++) {
                transposed_base[i * base_number + j] = base[j * vecdim + i];
            }
        }
        CHECK_CUDA(cudaMemcpy(d_base, transposed_base, base_number * vecdim * sizeof(float), cudaMemcpyHostToDevice));
    } else {
        CHECK_CUDA(cudaMemcpy(d_base, base, base_number * vecdim * sizeof(float), cudaMemcpyHostToDevice));
    }
    
    CHECK_CUDA(cudaMemcpy(d_queries, queries, query_num * vecdim * sizeof(float), cudaMemcpyHostToDevice));

    float alpha = 1.0f, beta = 0.0f;
    
    if (use_precomputed_transpose) {
        CHECK_CUBLAS(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                                 base_number, query_num, vecdim,
                                 &alpha, d_base, base_number,
                                 d_queries, vecdim, &beta, d_P, base_number));
    } else {
        CHECK_CUBLAS(cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N,
                                 base_number, query_num, vecdim,
                                 &alpha, d_base, vecdim,
                                 d_queries, vecdim, &beta, d_P, base_number));
    }

    // 使用更小的线程块配置进行距离转换
    dim3 conv_block(CONV_BLOCK_SIZE, CONV_BLOCK_SIZE);
    dim3 conv_grid((base_number + CONV_BLOCK_SIZE - 1) / CONV_BLOCK_SIZE, 
                   (query_num + CONV_BLOCK_SIZE - 1) / CONV_BLOCK_SIZE);
    
    convert_to_distance<<<conv_grid, conv_block>>>(d_P, d_D, base_number, query_num);
    
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());
    
    // 在GPU上执行TopK筛选
    dim3 topk_block(BLOCK_SIZE);
    dim3 topk_grid(query_num);
    size_t shared_mem_size = TOPK * BLOCK_SIZE * (sizeof(float) + sizeof(uint32_t));
    topk_kernel<<<topk_grid, topk_block, shared_mem_size>>>(d_D, d_topk_indices, base_number, TOPK, query_num);
    
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());
    
    // 将TopK结果复制回主机
    CHECK_CUDA(cudaMemcpy(h_topk_indices, d_topk_indices, query_num * TOPK * sizeof(uint32_t), cudaMemcpyDeviceToHost));
    
    CHECK_CUDA(cudaFree(d_base));
    CHECK_CUDA(cudaFree(d_queries));
    CHECK_CUDA(cudaFree(d_P));
    CHECK_CUDA(cudaFree(d_D));
    CHECK_CUDA(cudaFree(d_topk_indices));
    
    if (use_tensor_core) {
        CHECK_CUBLAS(cublasSetMathMode(handle, CUBLAS_DEFAULT_MATH));
    }
    
    CHECK_CUBLAS(cublasDestroy(handle));
    
    if (use_precomputed_transpose && transposed_base) {
        delete[] transposed_base;
    }
    
    CHECK_CUDA(cudaEventRecord(stop, 0));
    CHECK_CUDA(cudaEventSynchronize(stop));
    
    float milliseconds;
    CHECK_CUDA(cudaEventElapsedTime(&milliseconds, start, stop));
    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));
    
    SearchResult result;
    result.latency = static_cast<int64_t>(milliseconds * 1000); // 转换为微秒
    return result;
}

// 读取数据
template<typename T>
T* LoadData(string data_path, size_t& n, size_t& d) {
    ifstream fin(data_path, ios::in | ios::binary);
    if (!fin) {
        cerr << "Error: Failed to open file " << data_path << "\n";
        exit(EXIT_FAILURE);
    }
    int n_int, d_int;
    fin.read(reinterpret_cast<char*>(&n_int), 4);
    fin.read(reinterpret_cast<char*>(&d_int), 4);
    n = n_int;
    d = d_int;
    T* data = new T[n * d];
    int sz = sizeof(T);
    for (int i = 0; i < n; ++i) {
        if (!fin.read(reinterpret_cast<char*>(data) + i * d * sz, d * sz)) {
            cerr << "Error: Failed to read data from " << data_path << "\n";
            delete[] data;
            exit(EXIT_FAILURE);
        }
    }
    fin.close();
    cerr << "Loaded data: " << data_path << "\n";
    cerr << "Dimensions: " << d << ", Vectors: " << n << ", Element size: " << sizeof(T) << " bytes\n";
    return data;
}

// 主函数
int main(int argc, char *argv[]) {
    bool use_precomputed_transpose = false;
    if (argc > 1 && string(argv[1]) == "--transpose") {
        use_precomputed_transpose = true;
        cout << "Using precomputed transpose optimization\n";
    }

    int deviceCount;
    CHECK_CUDA(cudaGetDeviceCount(&deviceCount));
    if (deviceCount == 0) {
        cerr << "Error: No CUDA-capable devices found\n";
        return EXIT_FAILURE;
    }

    int deviceId;
    CHECK_CUDA(cudaGetDevice(&deviceId));
    cudaDeviceProp deviceProp;
    CHECK_CUDA(cudaGetDeviceProperties(&deviceProp, deviceId));
    cout << "Using CUDA device " << deviceId << ": " << deviceProp.name << "\n";
    cout << "Compute Capability: " << deviceProp.major << "." << deviceProp.minor << "\n";
    cout << "Max threads per block: " << deviceProp.maxThreadsPerBlock << "\n";
    cout << "Max grid dimensions: " << deviceProp.maxGridSize[0] << " x " 
         << deviceProp.maxGridSize[1] << " x " << deviceProp.maxGridSize[2] << "\n";

    size_t test_number = 0, base_number = 0, test_gt_d = 0, vecdim = 0;
    string data_path = "anndata/";
    auto test_query = LoadData<float>(data_path + "DEEP100K.query.fbin", test_number, vecdim);
    auto test_gt = LoadData<uint32_t>(data_path + "DEEP100K.gt.query.100k.top100.bin", test_number, test_gt_d);
    auto base = LoadData<float>(data_path + "DEEP100K.base.100k.fbin", base_number, vecdim);

    if (test_number > 2000) {
        test_number = 2000;
        cout << "Limiting to first 2000 test queries\n";
    }

    const size_t k = TOPK;
    vector<SearchResult> results(test_number);

    size_t num_batches = (test_number + BATCH_SIZE - 1) / BATCH_SIZE;
    cout << "Processing " << test_number << " queries in " << num_batches << " batches of " << BATCH_SIZE << " queries each\n";

    // 预热GPU
    cout << "Warming up GPU...\n";
    uint32_t* warmup_indices = new uint32_t[BATCH_SIZE * k];
    flat_search_batch_gpu(base, test_query, warmup_indices, base_number, vecdim, min(BATCH_SIZE, test_number), use_precomputed_transpose);
    delete[] warmup_indices;

    // 用于存储TopK结果的缓冲区
    uint32_t* h_topk_indices = new uint32_t[BATCH_SIZE * k];

    // 记录总开始时间
    auto total_start_time = chrono::high_resolution_clock::now();
    
    for (size_t batch = 0; batch < num_batches; ++batch) {
        size_t batch_start = batch * BATCH_SIZE;
        size_t batch_end = min(batch_start + BATCH_SIZE, test_number);
        size_t batch_size = batch_end - batch_start;

        cout << "Processing batch " << (batch + 1) << "/" << num_batches << " (" << batch_start << "-" << (batch_end - 1) << ")\n";

        float* batch_queries = new float[batch_size * vecdim];
        for (size_t i = 0; i < batch_size; ++i) {
            memcpy(batch_queries + i * vecdim, test_query + (batch_start + i) * vecdim, vecdim * sizeof(float));
        }

        // GPU计算 + TopK
        SearchResult batch_result = flat_search_batch_gpu(
            base, batch_queries, h_topk_indices, 
            base_number, vecdim, batch_size, use_precomputed_transpose
        );

        // 计算召回率
        for (size_t q = 0; q < batch_size; ++q) {
            set<uint32_t> gtset;
            for (int j = 0; j < k; ++j) {
                gtset.insert(test_gt[(batch_start + q) * test_gt_d + j]);
            }

            size_t correct = 0;
            for (size_t j = 0; j < k; ++j) {
                uint32_t idx = h_topk_indices[q * k + j];
                if (gtset.find(idx) != gtset.end()) {
                    correct++;
                }
            }

            results[batch_start + q] = {
                static_cast<float>(correct) / static_cast<float>(k),
                batch_result.latency / static_cast<int64_t>(batch_size)  // 平均每个查询的延迟
            };
        }

        delete[] batch_queries;
    }
    
    // 记录总结束时间
    auto total_end_time = chrono::high_resolution_clock::now();
    auto total_duration = chrono::duration_cast<chrono::microseconds>(total_end_time - total_start_time).count();
    double total_seconds = total_duration / 1000000.0;
    double qps = test_number / total_seconds;

    float avg_recall = 0.0f;
    int64_t avg_latency = 0;
    for (const auto& result : results) {
        avg_recall += result.recall;
        avg_latency += result.latency;
    }
    avg_recall /= test_number;
    avg_latency /= test_number;

    cout << fixed << setprecision(6);
    cout << "\n=== Final Results ===\n";
    cout << "Average recall: " << avg_recall << "\n";
    cout << "Average latency: " << avg_latency << " us\n";
    cout << "Total processing time: " << total_seconds << " seconds\n";
    cout << "Queries per second (QPS): " << qps << "\n";

    delete[] test_query;
    delete[] test_gt;
    delete[] base;
    delete[] h_topk_indices;
    CHECK_CUDA(cudaDeviceReset());

    return 0;
}

//表现一般的Ivf版本


// #include <vector>
// #include <cstring>
// #include <string>
// #include <iostream>
// #include <fstream>
// #include <set>
// #include <iomanip>
// #include <sstream>
// #include <sys/time.h>
// #include <cuda_runtime.h>
// #include <device_launch_parameters.h>
// #include <queue>
// #include <cublas_v2.h>
// #include <algorithm>
// #include <cmath>
// #include <unordered_set>
// #include <omp.h>
// #include <map>
// #include <climits>
// #include <cfloat>

// using namespace std;

// // Search结果结构体
// struct SearchResult {
//     float recall;
//     int64_t latency; // 单位us
// };

// // 检查CUDA和cuBLAS调用结果
// #define CHECK_CUDA(call) do { \
//     cudaError_t cudaStatus = call; \
//     if (cudaStatus != cudaSuccess) { \
//         fprintf(stderr, "CUDA error at %s:%d: %s\n", \
//                 __FILE__, __LINE__, cudaGetErrorString(cudaStatus)); \
//         exit(EXIT_FAILURE); \
//     } \
// } while(0)

// #define CHECK_CUBLAS(call) do { \
//     cublasStatus_t status = call; \
//     if (status != CUBLAS_STATUS_SUCCESS) { \
//         fprintf(stderr, "cuBLAS error at %s:%d: %d\n", \
//                 __FILE__, __LINE__, status); \
//         exit(EXIT_FAILURE); \
//     } \
// } while(0)

// // 批量大小
// const size_t BATCH_SIZE = 500;
// const size_t IVF_NPROBE = 24;  // IVF探查的聚类数量

// // 将内积转换为距离的核函数
// __global__ void convert_to_distance(float* d_P, float* d_D, size_t n, size_t m) {
//     size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
//     if (idx < n * m) {
//         d_D[idx] = 1.0f - d_P[idx];
//     }
// }

// // 映射全局ID的核函数
// __global__ void map_global_ids_kernel(uint32_t* results, uint32_t* global_ids, 
//                                      size_t total_candidates, size_t batch_size, size_t k) {
//     size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
//     if (idx < batch_size * k) {
//         uint32_t local_idx = results[idx];
//         if (local_idx < total_candidates) {
//             results[idx] = global_ids[local_idx];
//         }
//     }
// }

// // TopK核函数
// __global__ void topk_kernel(float* distances, uint32_t* topk_indices, float* topk_dists,
//                             size_t num_candidates, size_t k, size_t batch_size) {
//     size_t query_id = blockIdx.x;
//     size_t start = query_id * num_candidates;
    
//     // 每个线程处理一个查询
//     if (threadIdx.x == 0) {
//         // 初始化TopK数组
//         for (size_t i = 0; i < k; i++) {
//             topk_dists[query_id * k + i] = FLT_MAX;
//             topk_indices[query_id * k + i] = UINT_MAX;
//         }
        
//         // 遍历所有候选项
//         for (size_t i = 0; i < num_candidates; i++) {
//             float dist = distances[start + i];
            
//             // 如果比当前最大值小，则插入
//             if (dist < topk_dists[query_id * k + k - 1]) {
//                 // 找到插入位置
//                 int pos = k - 2;
//                 while (pos >= 0 && dist < topk_dists[query_id * k + pos]) {
//                     pos--;
//                 }
//                 pos++; // 插入位置
                
//                 // 移动元素
//                 for (int j = k - 1; j > pos; j--) {
//                     topk_dists[query_id * k + j] = topk_dists[query_id * k + j - 1];
//                     topk_indices[query_id * k + j] = topk_indices[query_id * k + j - 1];
//                 }
                
//                 // 插入新元素
//                 topk_dists[query_id * k + pos] = dist;
//                 topk_indices[query_id * k + pos] = i;
//             }
//         }
//     }
// }

// // IVF索引结构
// struct IVFIndex {
//     int nlist, dim;
//     vector<vector<float>> centroids;
//     vector<vector<uint32_t>> inverted_lists;
//     vector<float*> d_cluster_vectors; // GPU上的簇向量矩阵
//     vector<size_t> cluster_sizes;     // 每个簇的向量数量

//     void load(const string& filename, float* base) {
//         ifstream file(filename, ios::binary);
//         if (!file) {
//             cerr << "Error loading IVF index from " << filename << endl;
//             exit(EXIT_FAILURE);
//         }

//         file.read(reinterpret_cast<char*>(&nlist), sizeof(int));
//         file.read(reinterpret_cast<char*>(&dim), sizeof(int));
        
//         centroids.resize(nlist, vector<float>(dim));
//         for (int i = 0; i < nlist; ++i) {
//             file.read(reinterpret_cast<char*>(centroids[i].data()), dim * sizeof(float));
//         }

//         inverted_lists.resize(nlist);
//         for (int i = 0; i < nlist; ++i) {
//             int size;
//             file.read(reinterpret_cast<char*>(&size), sizeof(int));
//             inverted_lists[i].resize(size);
//             if (size > 0) {
//                 file.read(reinterpret_cast<char*>(inverted_lists[i].data()), size * sizeof(uint32_t));
//             }
//         }
//         file.close();

//         // 预处理簇向量矩阵并上传到GPU
//         d_cluster_vectors.resize(nlist, nullptr);
//         cluster_sizes.resize(nlist, 0);
//         for (int i = 0; i < nlist; ++i) {
//             cluster_sizes[i] = inverted_lists[i].size();
//             if (cluster_sizes[i] > 0) {
//                 float* cluster_vectors = new float[cluster_sizes[i] * dim];
//                 for (size_t j = 0; j < cluster_sizes[i]; ++j) {
//                     uint32_t id = inverted_lists[i][j];
//                     memcpy(cluster_vectors + j * dim, base + id * dim, dim * sizeof(float));
//                 }
//                 CHECK_CUDA(cudaMalloc((void**)&d_cluster_vectors[i], cluster_sizes[i] * dim * sizeof(float)));
//                 CHECK_CUDA(cudaMemcpy(d_cluster_vectors[i], cluster_vectors, cluster_sizes[i] * dim * sizeof(float), cudaMemcpyHostToDevice));
//                 delete[] cluster_vectors;
//             }
//         }
//         cout << "Loaded IVF index: nlist=" << nlist << ", dim=" << dim << endl;
//     }

//     void process_clusters(const float* query, vector<int>& selected_clusters, size_t nprobe = IVF_NPROBE) {
//         vector<pair<float, int>> all_distances(nlist);

//         #pragma omp parallel for
//         for (int cid = 0; cid < nlist; ++cid) {
//             float dist_sq = 0.0f;
//             for (int d = 0; d < dim; ++d) {
//                 float diff = centroids[cid][d] - query[d];
//                 dist_sq += diff * diff;
//             }
//             all_distances[cid] = make_pair(dist_sq, cid);
//         }

//         // 部分排序
//         nth_element(all_distances.begin(), all_distances.begin() + nprobe, all_distances.end());
//         sort(all_distances.begin(), all_distances.begin() + nprobe);
        
//         selected_clusters.resize(nprobe);
//         for (size_t i = 0; i < nprobe; ++i) {
//             selected_clusters[i] = all_distances[i].second;
//         }
//     }

//     ~IVFIndex() {
//         for (auto ptr : d_cluster_vectors) {
//             if (ptr) CHECK_CUDA(cudaFree(ptr));
//         }
//     }
// };

// // GPU批量搜索函数（优化版）
// void optimized_ivf_search_batch(IVFIndex& ivf_index, float* queries, 
//                                 uint32_t* d_results, float* d_result_dists,
//                                 size_t vecdim, size_t batch_size, 
//                                 size_t k) {
//     // 步骤1: 为batch中每个查询选择候选簇
//     vector<vector<int>> query_clusters(batch_size);
//     #pragma omp parallel for
//     for (size_t q = 0; q < batch_size; ++q) {
//         ivf_index.process_clusters(queries + q * vecdim, query_clusters[q]);
//     }

//     // 步骤2: 统计所有候选簇
//     unordered_set<int> all_clusters;
//     for (auto& clusters : query_clusters) {
//         all_clusters.insert(clusters.begin(), clusters.end());
//     }
//     vector<int> selected_clusters(all_clusters.begin(), all_clusters.end());

//     // 步骤3: 计算每个簇的向量总数和全局ID映射
//     vector<size_t> cluster_offsets(ivf_index.nlist, 0);
//     size_t total_candidates = 0;
//     vector<uint32_t> global_ids;
//     vector<size_t> cluster_vector_counts;
    
//     for (int cid : selected_clusters) {
//         cluster_offsets[cid] = total_candidates;
//         size_t cluster_size = ivf_index.cluster_sizes[cid];
//         cluster_vector_counts.push_back(cluster_size);
//         total_candidates += cluster_size;
        
//         // 构建全局ID映射
//         global_ids.insert(global_ids.end(), 
//                           ivf_index.inverted_lists[cid].begin(),
//                           ivf_index.inverted_lists[cid].end());
//     }
    
//     if (total_candidates == 0) {
//         cerr << "Warning: No candidates found for this batch" << endl;
//         return;
//     }

//     // 步骤4: 创建统一内存布局
//     float* d_all_vectors = nullptr;
//     CHECK_CUDA(cudaMalloc((void**)&d_all_vectors, total_candidates * vecdim * sizeof(float)));
    
//     // 将簇向量复制到统一内存
//     size_t offset = 0;
//     for (int cid : selected_clusters) {
//         size_t cluster_size = ivf_index.cluster_sizes[cid];
//         if (cluster_size == 0) continue;
        
//         CHECK_CUDA(cudaMemcpy(d_all_vectors + offset * vecdim,
//                               ivf_index.d_cluster_vectors[cid],
//                               cluster_size * vecdim * sizeof(float),
//                               cudaMemcpyDeviceToDevice));
//         offset += cluster_size;
//     }

//     // 步骤5: 执行批处理矩阵乘法
//     float* d_queries = nullptr;
//     float* d_distances = nullptr;
//     CHECK_CUDA(cudaMalloc((void**)&d_queries, batch_size * vecdim * sizeof(float)));
//     CHECK_CUDA(cudaMalloc((void**)&d_distances, batch_size * total_candidates * sizeof(float)));
    
//     CHECK_CUDA(cudaMemcpy(d_queries, queries, batch_size * vecdim * sizeof(float), cudaMemcpyHostToDevice));
    
//     cublasHandle_t handle;
//     CHECK_CUBLAS(cublasCreate(&handle));
    
//     float alpha = 1.0f, beta = 0.0f;
//     CHECK_CUBLAS(cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N,
//                              total_candidates, batch_size, vecdim,
//                              &alpha, d_all_vectors, vecdim,
//                              d_queries, vecdim,
//                              &beta, d_distances, total_candidates));
    
//     // 转换为距离
//     dim3 convert_block(256);
//     dim3 convert_grid((batch_size * total_candidates + convert_block.x - 1) / convert_block.x);
//     convert_to_distance<<<convert_grid, convert_block>>>(d_distances, d_distances, 
//                                                         total_candidates, batch_size);
//     CHECK_CUDA(cudaDeviceSynchronize());

//     // 步骤6: 在设备上计算TopK
//     float* d_topk_dists = nullptr;
//     uint32_t* d_topk_indices = nullptr;
//     CHECK_CUDA(cudaMalloc((void**)&d_topk_dists, batch_size * k * sizeof(float)));
//     CHECK_CUDA(cudaMalloc((void**)&d_topk_indices, batch_size * k * sizeof(uint32_t)));
    
//     // 调用自定义topk核函数
//     dim3 topk_block(1); // 每个查询使用一个线程块
//     dim3 topk_grid(batch_size);
//     topk_kernel<<<topk_grid, topk_block>>>(d_distances, d_topk_indices, d_topk_dists,
//                                           total_candidates, k, batch_size);
//     CHECK_CUDA(cudaDeviceSynchronize());
    
//     // 复制TopK结果
//     CHECK_CUDA(cudaMemcpy(d_results, d_topk_indices, batch_size * k * sizeof(uint32_t), 
//                          cudaMemcpyDeviceToDevice));
//     CHECK_CUDA(cudaMemcpy(d_result_dists, d_topk_dists, batch_size * k * sizeof(float), 
//                          cudaMemcpyDeviceToDevice));

//     // 步骤7: 映射回全局ID
//     uint32_t* d_global_ids = nullptr;
//     CHECK_CUDA(cudaMalloc((void**)&d_global_ids, global_ids.size() * sizeof(uint32_t)));
//     CHECK_CUDA(cudaMemcpy(d_global_ids, global_ids.data(), 
//                          global_ids.size() * sizeof(uint32_t), 
//                          cudaMemcpyHostToDevice));
    
//     // 映射全局ID
//     dim3 map_block(256);
//     dim3 map_grid((batch_size * k + map_block.x - 1) / map_block.x);
//     map_global_ids_kernel<<<map_grid, map_block>>>(d_results, d_global_ids, total_candidates, batch_size, k);
//     CHECK_CUDA(cudaDeviceSynchronize());
    
//     // 清理设备内存
//     CHECK_CUDA(cudaFree(d_all_vectors));
//     CHECK_CUDA(cudaFree(d_queries));
//     CHECK_CUDA(cudaFree(d_distances));
//     CHECK_CUDA(cudaFree(d_topk_dists));
//     CHECK_CUDA(cudaFree(d_topk_indices));
//     CHECK_CUDA(cudaFree(d_global_ids));
//     CHECK_CUBLAS(cublasDestroy(handle));
// }

// // 读取数据
// template<typename T>
// T* LoadData(string data_path, size_t& n, size_t& d) {
//     ifstream fin(data_path, ios::in | ios::binary);
//     if (!fin) {
//         cerr << "Error: Failed to open file " << data_path << "\n";
//         exit(EXIT_FAILURE);
//     }
//     int n_int, d_int;
//     fin.read(reinterpret_cast<char*>(&n_int), 4);
//     fin.read(reinterpret_cast<char*>(&d_int), 4);
//     n = n_int;
//     d = d_int;
//     T* data = new T[n * d];
//     int sz = sizeof(T);
//     for (int i = 0; i < n; ++i) {
//         if (!fin.read(reinterpret_cast<char*>(data) + i * d * sz, d * sz)) {
//             cerr << "Error: Failed to read data from " << data_path << "\n";
//             delete[] data;
//             exit(EXIT_FAILURE);
//         }
//     }
//     fin.close();
//     cerr << "Loaded data: " << data_path << "\n";
//     cerr << "Dimensions: " << d << ", Vectors: " << n << ", Element size: " << sizeof(T) << " bytes\n";
//     return data;
// }

// // 主函数
// int main(int argc, char *argv[]) {
//     bool use_ivf = false;
    
//     // 解析命令行参数
//     for (int i = 1; i < argc; i++) {
//         if (string(argv[i]) == "--ivf") {
//             use_ivf = true;
//             cout << "Using optimized IVF search with batch processing\n";
//         }
//     }

//     int deviceCount;
//     CHECK_CUDA(cudaGetDeviceCount(&deviceCount));
//     if (deviceCount == 0) {
//         cerr << "Error: No CUDA-capable devices found\n";
//         return EXIT_FAILURE;
//     }

//     int deviceId;
//     CHECK_CUDA(cudaGetDevice(&deviceId));
//     cudaDeviceProp deviceProp;
//     CHECK_CUDA(cudaGetDeviceProperties(&deviceProp, deviceId));
//     cout << "Using CUDA device " << deviceId << ": " << deviceProp.name << "\n";
//     cout << "Compute Capability: " << deviceProp.major << "." << deviceProp.minor << "\n";

//     size_t test_number = 0, base_number = 0, test_gt_d = 0, vecdim = 0;
//     string data_path = "anndata/";
//     auto test_query = LoadData<float>(data_path + "DEEP100K.query.fbin", test_number, vecdim);
//     auto test_gt = LoadData<uint32_t>(data_path + "DEEP100K.gt.query.100k.top100.bin", test_number, test_gt_d);
//     auto base = LoadData<float>(data_path + "DEEP100K.base.100k.fbin", base_number, vecdim);

//     if (test_number > 2000) {
//         test_number = 2000;
//         cout << "Limiting to first 2000 test queries\n";
//     }

//     const size_t k = 10;
//     vector<SearchResult> results(test_number);

//     // 加载IVF索引
//     IVFIndex ivf_index;
//     if (use_ivf) {
//         ivf_index.load("files/ivf.index", base);
//         if (ivf_index.dim != vecdim) {
//             cerr << "Error: IVF index dimension (" << ivf_index.dim 
//                  << ") does not match vector dimension (" << vecdim << ")\n";
//             exit(EXIT_FAILURE);
//         }
//     } else {
//         cerr << "Non-IVF mode requires different implementation\n";
//         exit(EXIT_FAILURE);
//     }

//     size_t num_batches = (test_number + BATCH_SIZE - 1) / BATCH_SIZE;
//     cout << "Processing " << test_number << " queries in " << num_batches << " batches of " << BATCH_SIZE << " queries each\n";

//     // 预热GPU
//     cout << "Warming up GPU...\n";
//     {
//         uint32_t* d_warmup_results;
//         float* d_warmup_dists;
//         CHECK_CUDA(cudaMalloc((void**)&d_warmup_results, min(BATCH_SIZE, test_number) * k * sizeof(uint32_t)));
//         CHECK_CUDA(cudaMalloc((void**)&d_warmup_dists, min(BATCH_SIZE, test_number) * k * sizeof(float)));
        
//         optimized_ivf_search_batch(ivf_index, test_query, d_warmup_results, d_warmup_dists,
//                                   vecdim, min(BATCH_SIZE, test_number), k);
        
//         CHECK_CUDA(cudaFree(d_warmup_results));
//         CHECK_CUDA(cudaFree(d_warmup_dists));
//     }

//     // 记录总耗时
//     cudaEvent_t total_start, total_stop;
//     CHECK_CUDA(cudaEventCreate(&total_start));
//     CHECK_CUDA(cudaEventCreate(&total_stop));
//     CHECK_CUDA(cudaEventRecord(total_start, 0));

//     // 分配设备内存用于结果
//     uint32_t* d_batch_results;
//     float* d_batch_dists;
//     CHECK_CUDA(cudaMalloc((void**)&d_batch_results, BATCH_SIZE * k * sizeof(uint32_t)));
//     CHECK_CUDA(cudaMalloc((void**)&d_batch_dists, BATCH_SIZE * k * sizeof(float)));
    
//     uint32_t* h_batch_results = new uint32_t[BATCH_SIZE * k];
//     float* h_batch_dists = new float[BATCH_SIZE * k];

//     for (size_t batch = 0; batch < num_batches; ++batch) {
//         size_t batch_start = batch * BATCH_SIZE;
//         size_t batch_end = min(batch_start + BATCH_SIZE, test_number);
//         size_t batch_size = batch_end - batch_start;

//         cout << "Processing batch " << (batch + 1) << "/" << num_batches 
//              << " (" << batch_start << "-" << (batch_end - 1) << ")\n";

//         float* batch_queries = test_query + batch_start * vecdim;

//         cudaEvent_t start, stop;
//         CHECK_CUDA(cudaEventCreate(&start));
//         CHECK_CUDA(cudaEventCreate(&stop));
//         CHECK_CUDA(cudaEventRecord(start, 0));

//         // 执行优化搜索
//         optimized_ivf_search_batch(ivf_index, batch_queries, d_batch_results, d_batch_dists,
//                                   vecdim, batch_size, k);
        
//         // 复制结果回主机
//         CHECK_CUDA(cudaMemcpy(h_batch_results, d_batch_results, batch_size * k * sizeof(uint32_t),
//                              cudaMemcpyDeviceToHost));
//         CHECK_CUDA(cudaMemcpy(h_batch_dists, d_batch_dists, batch_size * k * sizeof(float),
//                              cudaMemcpyDeviceToHost));

//         CHECK_CUDA(cudaEventRecord(stop, 0));
//         CHECK_CUDA(cudaEventSynchronize(stop));
//         float milliseconds = 0;
//         CHECK_CUDA(cudaEventElapsedTime(&milliseconds, start, stop));
//         int64_t batch_latency = static_cast<int64_t>(milliseconds * 1000);
//         int64_t per_query_latency = batch_latency / static_cast<int64_t>(batch_size);

//         // 处理结果
//         for (size_t q = 0; q < batch_size; ++q) {
//             set<uint32_t> gtset;
//             for (int j = 0; j < k; ++j) {
//                 gtset.insert(test_gt[(batch_start + q) * test_gt_d + j]);
//             }

//             size_t correct = 0;
//             for (size_t j = 0; j < k; ++j) {
//                 uint32_t result_id = h_batch_results[q * k + j];
//                 if (gtset.find(result_id) != gtset.end()) {
//                     correct++;
//                 }
//             }

//             results[batch_start + q] = {static_cast<float>(correct) / static_cast<float>(k), per_query_latency};
//         }

//         CHECK_CUDA(cudaEventDestroy(start));
//         CHECK_CUDA(cudaEventDestroy(stop));
//     }

//     // 计算总耗时
//     CHECK_CUDA(cudaEventRecord(total_stop, 0));
//     CHECK_CUDA(cudaEventSynchronize(total_stop));
//     float total_milliseconds = 0;
//     CHECK_CUDA(cudaEventElapsedTime(&total_milliseconds, total_start, total_stop));
//     int64_t total_latency = static_cast<int64_t>(total_milliseconds * 1000);
//     CHECK_CUDA(cudaEventDestroy(total_start));
//     CHECK_CUDA(cudaEventDestroy(total_stop));

//     // 计算平均指标
//     float avg_recall = 0.0f;
//     int64_t avg_latency = 0;
//     for (const auto& result : results) {
//         avg_recall += result.recall;
//         avg_latency += result.latency;
//     }
//     avg_recall /= test_number;
//     avg_latency /= test_number;

//     // 计算QPS
//     double qps = (test_number * 1000000.0) / total_latency;

//     cout << fixed << setprecision(6);
//     cout << "\n=== Final Results ===\n";
//     cout << "Average recall: " << avg_recall << "\n";
//     cout << "Average per-query latency: " << avg_latency << " us\n";
//     cout << "Total processing time: " << total_milliseconds << " ms\n";
//     cout << "Total queries processed: " << test_number << "\n";
//     cout << "QPS (queries per second): " << qps << "\n";

//     // 清理资源
//     delete[] test_query;
//     delete[] test_gt;
//     delete[] base;
//     delete[] h_batch_results;
//     delete[] h_batch_dists;
//     CHECK_CUDA(cudaFree(d_batch_results));
//     CHECK_CUDA(cudaFree(d_batch_dists));
//     CHECK_CUDA(cudaDeviceReset());

//     return 0;
// }

// #include <vector>
// #include <cstring>
// #include <string>
// #include <iostream>
// #include <fstream>
// #include <set>
// #include <iomanip>
// #include <sstream>
// #include <sys/time.h>
// #include <cuda_runtime.h>
// #include <device_launch_parameters.h>
// #include <queue>
// #include <cublas_v2.h>
// #include <algorithm>
// #include <cmath>
// #include <cfloat>
// #include <chrono>

// using namespace std;

// // 检查CUDA和cuBLAS调用结果
// #define CHECK_CUDA(call) do { \
//     cudaError_t cudaStatus = call; \
//     if (cudaStatus != cudaSuccess) { \
//         fprintf(stderr, "CUDA error at %s:%d: %s\n", \
//                 __FILE__, __LINE__, cudaGetErrorString(cudaStatus)); \
//         exit(EXIT_FAILURE); \
//     } \
// } while(0)

// #define CHECK_CUBLAS(call) do { \
//     cublasStatus_t status = call; \
//     if (status != CUBLAS_STATUS_SUCCESS) { \
//         fprintf(stderr, "cuBLAS error at %s:%d: %d\n", \
//                 __FILE__, __LINE__, status); \
//         exit(EXIT_FAILURE); \
//     } \
// } while(0)

// // 默认值
// const size_t TOPK = 10;

// // 自定义交换函数（用于设备代码）
// __device__ void swap_float(float &a, float &b) {
//     float temp = a;
//     a = b;
//     b = temp;
// }

// __device__ void swap_uint(uint32_t &a, uint32_t &b) {
//     uint32_t temp = a;
//     a = b;
//     b = temp;
// }

// // 将内积转换为距离的核函数
// __global__ void convert_to_distance(float* d_P, float* d_D, size_t n, size_t m, size_t conv_block_size) {
//     size_t idx = blockIdx.x * conv_block_size + threadIdx.x; // base index
//     size_t idy = blockIdx.y * conv_block_size + threadIdx.y; // query index
//     if (idx < n && idy < m) {
//         size_t p_index = idx + idy * n; // column-major index for d_P
//         size_t d_index = idy * n + idx; // row-major index for d_D
//         d_D[d_index] = 1.0f - d_P[p_index];
//     }
// }

// // 策略1: 每个线程块处理一个查询的TopK核函数
// __global__ void topk_kernel_strategy1(float* distances, uint32_t* indices, size_t base_number, size_t k, size_t query_num, size_t block_size) {
//     extern __shared__ float sdata[]; // 共享内存用于存储距离和索引
    
//     // 当前处理的查询索引
//     size_t query_idx = blockIdx.x;
//     if (query_idx >= query_num) return;
    
//     float* dist_ptr = distances + query_idx * base_number;
//     uint32_t* idx_ptr = indices + query_idx * k;
    
//     // 每个线程的私有TopK数组（距离+索引）
//     float* local_dists = sdata;
//     uint32_t* local_idxs = (uint32_t*)(local_dists + k * block_size);
    
//     // 初始化私有TopK
//     size_t tid = threadIdx.x;
//     for (size_t i = 0; i < k; i++) {
//         local_dists[tid * k + i] = FLT_MAX;
//         local_idxs[tid * k + i] = 0;
//     }
    
//     // 计算每个线程处理的数据范围
//     size_t items_per_thread = (base_number + block_size - 1) / block_size;
//     size_t start = tid * items_per_thread;
//     size_t end = min(start + items_per_thread, base_number);
    
//     // 局部TopK筛选
//     for (size_t i = start; i < end; i++) {
//         float dist = dist_ptr[i];
        
//         // 如果比当前堆顶小，则替换堆顶并调整堆
//         if (dist < local_dists[tid * k]) {
//             // 替换堆顶
//             local_dists[tid * k] = dist;
//             local_idxs[tid * k] = i;
            
//             // 调整堆（下沉）
//             size_t cur = 0;
//             while (cur < k) {
//                 size_t left = 2 * cur + 1;
//                 size_t right = 2 * cur + 2;
//                 size_t smallest = cur;
                
//                 if (left < k && local_dists[tid * k + left] > local_dists[tid * k + smallest]) 
//                     smallest = left;
//                 if (right < k && local_dists[tid * k + right] > local_dists[tid * k + smallest])
//                     smallest = right;
                
//                 if (smallest != cur) {
//                     swap_float(local_dists[tid * k + cur], local_dists[tid * k + smallest]);
//                     swap_uint(local_idxs[tid * k + cur], local_idxs[tid * k + smallest]);
//                     cur = smallest;
//                 } else {
//                     break;
//                 }
//             }
//         }
//     }
//     __syncthreads();
    
//     // 合并所有线程的局部TopK
//     if (tid == 0) {
//         // 初始化全局TopK
//         float global_dists[TOPK];
//         uint32_t global_idxs[TOPK];
//         for (size_t i = 0; i < k; i++) {
//             global_dists[i] = FLT_MAX;
//             global_idxs[i] = 0;
//         }
        
//         // 合并所有线程的局部结果
//         for (size_t t = 0; t < block_size; t++) {
//             for (size_t i = 0; i < k; i++) {
//                 float dist = local_dists[t * k + i];
//                 uint32_t idx = local_idxs[t * k + i];
                
//                 // 如果比当前堆顶小，则插入堆
//                 if (dist < global_dists[0]) {
//                     // 替换堆顶
//                     global_dists[0] = dist;
//                     global_idxs[0] = idx;
                    
//                     // 调整堆（下沉）
//                     size_t cur = 0;
//                     while (cur < k) {
//                         size_t left = 2 * cur + 1;
//                         size_t right = 2 * cur + 2;
//                         size_t largest = cur;
                        
//                         if (left < k && global_dists[left] > global_dists[largest])
//                             largest = left;
//                         if (right < k && global_dists[right] > global_dists[largest])
//                             largest = right;
                        
//                         if (largest != cur) {
//                             swap_float(global_dists[cur], global_dists[largest]);
//                             swap_uint(global_idxs[cur], global_idxs[largest]);
//                             cur = largest;
//                         } else {
//                             break;
//                         }
//                     }
//                 }
//             }
//         }
        
//         // 对TopK结果排序（从小到大）
//         for (size_t i = k - 1; i > 0; i--) {
//             swap_float(global_dists[0], global_dists[i]);
//             swap_uint(global_idxs[0], global_idxs[i]);
            
//             // 调整堆（下沉）
//             size_t cur = 0;
//             while (cur < i) {
//                 size_t left = 2 * cur + 1;
//                 size_t right = 2 * cur + 2;
//                 size_t largest = cur;
                
//                 if (left < i && global_dists[left] > global_dists[largest])
//                     largest = left;
//                 if (right < i && global_dists[right] > global_dists[largest])
//                     largest = right;
                
//                 if (largest != cur) {
//                     swap_float(global_dists[cur], global_dists[largest]);
//                     swap_uint(global_idxs[cur], global_idxs[largest]);
//                     cur = largest;
//                 } else {
//                     break;
//                 }
//             }
//         }
        
//         // 写入最终结果
//         for (size_t i = 0; i < k; i++) {
//             idx_ptr[i] = global_idxs[i];
//         }
//     }
// }

// // 策略2: 整个线程块协作处理一个查询的TopK核函数（优化版）
// __global__ void topk_kernel_strategy2(float* distances, uint32_t* indices, 
//                                      size_t base_number, size_t k, 
//                                      size_t query_num, size_t block_size) {
//     extern __shared__ float sdata[]; // 共享内存用于存储距离和索引
    
//     // 每个线程的局部TopK存储在共享内存中
//     // 结构: [线程0距离, 线程0索引, 线程1距离, 线程1索引, ...]
//     float* local_dists = sdata;
//     // 总共享内存大小: block_size * 2 * k * sizeof(float)
    
//     // 当前处理的查询索引
//     size_t query_idx = blockIdx.x;
//     if (query_idx >= query_num) return;
    
//     float* dist_ptr = distances + query_idx * base_number;
//     uint32_t* idx_ptr = indices + query_idx * k;
    
//     size_t tid = threadIdx.x;
//     // 当前线程的局部TopK在共享内存中的位置
//     float* my_local_dists = local_dists + tid * 2 * k;
//     uint32_t* my_local_idxs = (uint32_t*)(my_local_dists + k);
    
//     // 初始化局部TopK（最大堆，堆顶是最大值）
//     for (size_t i = 0; i < k; i++) {
//         my_local_dists[i] = FLT_MAX; // 初始化为正的最大值
//         my_local_idxs[i] = 0;
//     }

//     // 计算每个线程处理的数据范围
//     size_t items_per_thread = (base_number + block_size - 1) / block_size;
//     size_t start = tid * items_per_thread;
//     size_t end = min(start + items_per_thread, base_number);

//     // 局部TopK筛选（维护一个最大堆）
//     for (size_t i = start; i < end; i++) {
//         float dist = dist_ptr[i];
//         // 如果当前距离比堆顶小（堆顶是当前局部TopK中最大的距离）
//         if (dist < my_local_dists[0]) {
//             // 替换堆顶
//             my_local_dists[0] = dist;
//             my_local_idxs[0] = i;

//             // 调整堆（下沉）
//             size_t cur = 0;
//             while (cur < k) {
//                 size_t left = 2 * cur + 1;
//                 size_t right = 2 * cur + 2;
//                 size_t largest = cur;

//                 if (left < k && my_local_dists[left] > my_local_dists[largest]) 
//                     largest = left;
//                 if (right < k && my_local_dists[right] > my_local_dists[largest])
//                     largest = right;

//                 if (largest != cur) {
//                     swap_float(my_local_dists[cur], my_local_dists[largest]);
//                     swap_uint(my_local_idxs[cur], my_local_idxs[largest]);
//                     cur = largest;
//                 } else {
//                     break;
//                 }
//             }
//         }
//     }
//     __syncthreads();

//     // 由线程0收集所有候选元素，并合并成全局TopK
//     if (tid == 0) {
//         // 最大堆，用于选取最小的k个元素
//         float heap_dists[TOPK];
//         uint32_t heap_idxs[TOPK];
//         size_t heap_count = 0;

//         // 遍历所有线程的局部TopK
//         for (size_t t = 0; t < block_size; t++) {
//             float* t_dists = local_dists + t * 2 * k;
//             uint32_t* t_idxs = (uint32_t*)(t_dists + k);

//             for (size_t i = 0; i < k; i++) {
//                 float dist_val = t_dists[i];
//                 // 跳过无效的初始值
//                 if (dist_val == FLT_MAX) 
//                     continue;

//                 // 如果堆还没满，直接加入
//                 if (heap_count < k) {
//                     heap_dists[heap_count] = dist_val;
//                     heap_idxs[heap_count] = t_idxs[i];
//                     heap_count++;
                    
//                     // 堆满后构建最大堆
//                     if (heap_count == k) {
//                         // 构建最大堆（堆顶是最大值）
//                         for (int j = (k-1)/2; j >=0; j--) {
//                             size_t cur = j;
//                             while (cur < k) {
//                                 size_t left = 2 * cur + 1;
//                                 size_t right = 2 * cur + 2;
//                                 size_t largest = cur;
//                                 if (left < k && heap_dists[left] > heap_dists[largest])
//                                     largest = left;
//                                 if (right < k && heap_dists[right] > heap_dists[largest])
//                                     largest = right;
//                                 if (largest != cur) {
//                                     swap_float(heap_dists[cur], heap_dists[largest]);
//                                     swap_uint(heap_idxs[cur], heap_idxs[largest]);
//                                     cur = largest;
//                                 } else {
//                                     break;
//                                 }
//                             }
//                         }
//                     }
//                 } 
//                 // 堆已满，如果当前元素比堆顶小则替换堆顶
//                 else if (dist_val < heap_dists[0]) {
//                     heap_dists[0] = dist_val;
//                     heap_idxs[0] = t_idxs[i];
                    
//                     // 调整堆：下沉
//                     size_t cur = 0;
//                     while (cur < k) {
//                         size_t left = 2 * cur + 1;
//                         size_t right = 2 * cur + 2;
//                         size_t largest = cur;
//                         if (left < k && heap_dists[left] > heap_dists[largest])
//                             largest = left;
//                         if (right < k && heap_dists[right] > heap_dists[largest])
//                             largest = right;
//                         if (largest != cur) {
//                             swap_float(heap_dists[cur], heap_dists[largest]);
//                             swap_uint(heap_idxs[cur], heap_idxs[largest]);
//                             cur = largest;
//                         } else {
//                             break;
//                         }
//                     }
//                 }
//             }
//         }

//         // 处理候选元素不足k的情况
//         if (heap_count < k) {
//             // 直接排序（因为元素数量少）
//             for (size_t i = 0; i < heap_count; i++) {
//                 for (size_t j = i+1; j < heap_count; j++) {
//                     if (heap_dists[j] < heap_dists[i]) {
//                         swap_float(heap_dists[i], heap_dists[j]);
//                         swap_uint(heap_idxs[i], heap_idxs[j]);
//                     }
//                 }
//             }
//         } 
//         // 堆排序（从小到大）
//         else {
//             for (int i = k-1; i > 0; i--) {
//                 swap_float(heap_dists[0], heap_dists[i]);
//                 swap_uint(heap_idxs[0], heap_idxs[i]);
                
//                 // 调整堆（0~i-1）
//                 size_t cur = 0;
//                 while (cur < i) {
//                     size_t left = 2 * cur + 1;
//                     size_t right = 2 * cur + 2;
//                     size_t largest = cur;
//                     if (left < i && heap_dists[left] > heap_dists[largest])
//                         largest = left;
//                     if (right < i && heap_dists[right] > heap_dists[largest])
//                         largest = right;
//                     if (largest != cur) {
//                         swap_float(heap_dists[cur], heap_dists[largest]);
//                         swap_uint(heap_idxs[cur], heap_idxs[largest]);
//                         cur = largest;
//                     } else {
//                         break;
//                     }
//                 }
//             }
//         }

//         // 将结果写入全局内存
//         for (size_t i = 0; i < k; i++) {
//             if (i < heap_count) {
//                 idx_ptr[i] = heap_idxs[i];
//             } else {
//                 idx_ptr[i] = 0; // 不足k个时用0填充
//             }
//         }
//     }
// }

// // Search结果结构体
// struct SearchResult {
//     float recall;
//     int64_t latency; // 单位us
//     double qps;
//     double total_time;
//     size_t batch_size;
//     size_t block_size;
//     size_t conv_block_size;
//     string strategy;
// };

// // 优化后的GPU批量搜索函数（包含TopK）
// SearchResult flat_search_batch_gpu(float* base, float* queries, 
//                                   uint32_t* h_topk_indices,
//                                   size_t base_number, size_t vecdim, size_t query_num,
//                                   size_t batch_size, 
//                                   size_t block_size,
//                                   size_t conv_block_size,
//                                   const string& strategy,
//                                   bool use_precomputed_transpose = false) {
//     cudaEvent_t start, stop;
//     CHECK_CUDA(cudaEventCreate(&start));
//     CHECK_CUDA(cudaEventCreate(&stop));
//     CHECK_CUDA(cudaEventRecord(start, 0));
    
//     float *d_base = nullptr, *d_queries = nullptr, *d_P = nullptr, *d_D = nullptr;
//     uint32_t* d_topk_indices = nullptr;
//     cublasHandle_t handle;
//     CHECK_CUBLAS(cublasCreate(&handle));

//     CHECK_CUDA(cudaMalloc((void**)&d_base, base_number * vecdim * sizeof(float)));
//     CHECK_CUDA(cudaMalloc((void**)&d_queries, query_num * vecdim * sizeof(float)));
//     CHECK_CUDA(cudaMalloc((void**)&d_P, base_number * query_num * sizeof(float)));
//     CHECK_CUDA(cudaMalloc((void**)&d_D, base_number * query_num * sizeof(float)));
//     CHECK_CUDA(cudaMalloc((void**)&d_topk_indices, query_num * TOPK * sizeof(uint32_t)));

//     cudaDeviceProp prop;
//     CHECK_CUDA(cudaGetDeviceProperties(&prop, 0));
//     bool use_tensor_core = (prop.major >= 7);
    
//     if (use_tensor_core) {
//         CHECK_CUBLAS(cublasSetMathMode(handle, CUBLAS_TENSOR_OP_MATH));
//     }

//     float* transposed_base = nullptr;
//     if (use_precomputed_transpose) {
//         transposed_base = new float[base_number * vecdim];
//         for (size_t i = 0; i < vecdim; i++) {
//             for (size_t j = 0; j < base_number; j++) {
//                 transposed_base[i * base_number + j] = base[j * vecdim + i];
//             }
//         }
//         CHECK_CUDA(cudaMemcpy(d_base, transposed_base, base_number * vecdim * sizeof(float), cudaMemcpyHostToDevice));
//     } else {
//         CHECK_CUDA(cudaMemcpy(d_base, base, base_number * vecdim * sizeof(float), cudaMemcpyHostToDevice));
//     }
    
//     CHECK_CUDA(cudaMemcpy(d_queries, queries, query_num * vecdim * sizeof(float), cudaMemcpyHostToDevice));

//     float alpha = 1.0f, beta = 0.0f;
    
//     if (use_precomputed_transpose) {
//         CHECK_CUBLAS(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
//                                  base_number, query_num, vecdim,
//                                  &alpha, d_base, base_number,
//                                  d_queries, vecdim, &beta, d_P, base_number));
//     } else {
//         CHECK_CUBLAS(cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N,
//                                  base_number, query_num, vecdim,
//                                  &alpha, d_base, vecdim,
//                                  d_queries, vecdim, &beta, d_P, base_number));
//     }

//     // 使用参数化的线程块配置进行距离转换
//     dim3 conv_block(conv_block_size, conv_block_size);
//     dim3 conv_grid((base_number + conv_block_size - 1) / conv_block_size, 
//                    (query_num + conv_block_size - 1) / conv_block_size);
    
//     convert_to_distance<<<conv_grid, conv_block>>>(d_P, d_D, base_number, query_num, conv_block_size);
    
//     CHECK_CUDA(cudaGetLastError());
//     CHECK_CUDA(cudaDeviceSynchronize());
    
//     // 在GPU上执行TopK筛选 - 根据策略选择不同的内核
//     if (strategy == "Strategy1") {
//         // 策略1: 每个线程块处理一个查询
//         dim3 topk_grid(query_num);
//         size_t shared_mem_size = TOPK * block_size * (sizeof(float) + sizeof(uint32_t));
//         topk_kernel_strategy1<<<topk_grid, block_size, shared_mem_size>>>(d_D, d_topk_indices, base_number, TOPK, query_num, block_size);
//     }
//     else if (strategy == "Strategy2") {
//         // 策略2: 整个线程块协作处理一个查询（优化版）
//         dim3 topk_grid(query_num);
//         // 共享内存大小 = block_size * 2 * k * sizeof(float)
//         size_t shared_mem_size = block_size * 2 * TOPK * sizeof(float);
//         topk_kernel_strategy2<<<topk_grid, block_size, shared_mem_size>>>(d_D, d_topk_indices, base_number, TOPK, query_num, block_size);
//     }
//     else {
//         cerr << "Error: Unknown strategy " << strategy << endl;
//         exit(EXIT_FAILURE);
//     }
    
//     CHECK_CUDA(cudaGetLastError());
//     CHECK_CUDA(cudaDeviceSynchronize());
    
//     // 将TopK结果复制回主机
//     CHECK_CUDA(cudaMemcpy(h_topk_indices, d_topk_indices, query_num * TOPK * sizeof(uint32_t), cudaMemcpyDeviceToHost));
    
//     CHECK_CUDA(cudaFree(d_base));
//     CHECK_CUDA(cudaFree(d_queries));
//     CHECK_CUDA(cudaFree(d_P));
//     CHECK_CUDA(cudaFree(d_D));
//     CHECK_CUDA(cudaFree(d_topk_indices));
    
//     if (use_tensor_core) {
//         CHECK_CUBLAS(cublasSetMathMode(handle, CUBLAS_DEFAULT_MATH));
//     }
    
//     CHECK_CUBLAS(cublasDestroy(handle));
    
//     if (use_precomputed_transpose && transposed_base) {
//         delete[] transposed_base;
//     }
    
//     CHECK_CUDA(cudaEventRecord(stop, 0));
//     CHECK_CUDA(cudaEventSynchronize(stop));
    
//     float milliseconds;
//     CHECK_CUDA(cudaEventElapsedTime(&milliseconds, start, stop));
//     CHECK_CUDA(cudaEventDestroy(start));
//     CHECK_CUDA(cudaEventDestroy(stop));
    
//     SearchResult result;
//     result.latency = static_cast<int64_t>(milliseconds * 1000); // 转换为微秒
//     result.strategy = strategy;
//     return result;
// }

// // 读取数据
// template<typename T>
// T* LoadData(string data_path, size_t& n, size_t& d) {
//     ifstream fin(data_path, ios::in | ios::binary);
//     if (!fin) {
//         cerr << "Error: Failed to open file " << data_path << "\n";
//         exit(EXIT_FAILURE);
//     }
//     int n_int, d_int;
//     fin.read(reinterpret_cast<char*>(&n_int), 4);
//     fin.read(reinterpret_cast<char*>(&d_int), 4);
//     n = n_int;
//     d = d_int;
//     T* data = new T[n * d];
//     int sz = sizeof(T);
//     for (int i = 0; i < n; ++i) {
//         if (!fin.read(reinterpret_cast<char*>(data) + i * d * sz, d * sz)) {
//             cerr << "Error: Failed to read data from " << data_path << "\n";
//             delete[] data;
//             exit(EXIT_FAILURE);
//         }
//     }
//     fin.close();
//     cerr << "Loaded data: " << data_path << "\n";
//     cerr << "Dimensions: " << d << ", Vectors: " << n << ", Element size: " << sizeof(T) << " bytes\n";
//     return data;
// }

// // 执行测试的函数
// vector<SearchResult> run_experiments(float* base, float* queries, uint32_t* test_gt,
//                                      size_t base_number, size_t test_gt_d, size_t vecdim, 
//                                      size_t test_number, bool use_precomputed_transpose,
//                                      const vector<size_t>& batch_sizes, 
//                                      const vector<size_t>& block_sizes,
//                                      const vector<size_t>& conv_block_sizes,
//                                      const vector<string>& strategies) {
//     vector<SearchResult> all_results;
//     const size_t k = TOPK;

//     for (const auto& strategy : strategies) {
//         for (size_t batch_size : batch_sizes) {
//             for (size_t block_size : block_sizes) {
//                 for (size_t conv_block_size : conv_block_sizes) {
//                     cout << "\n=== Testing configuration ===" << endl;
//                     cout << "Strategy: " << strategy << endl;
//                     cout << "Batch size: " << batch_size 
//                          << ", TopK block size: " << block_size
//                          << ", Conv block size: " << conv_block_size << endl;
                    
//                     vector<SearchResult> results(test_number);
//                     size_t num_batches = (test_number + batch_size - 1) / batch_size;

//                     // 用于存储TopK结果的缓冲区
//                     uint32_t* h_topk_indices = new uint32_t[batch_size * k];

//                     // 记录总开始时间
//                     auto total_start_time = chrono::high_resolution_clock::now();
                    
//                     for (size_t batch = 0; batch < num_batches; ++batch) {
//                         size_t batch_start = batch * batch_size;
//                         size_t batch_end = min(batch_start + batch_size, test_number);
//                         size_t actual_batch_size = batch_end - batch_start;

//                         float* batch_queries = new float[actual_batch_size * vecdim];
//                         for (size_t i = 0; i < actual_batch_size; ++i) {
//                             memcpy(batch_queries + i * vecdim, queries + (batch_start + i) * vecdim, vecdim * sizeof(float));
//                         }

//                         // GPU计算 + TopK
//                         SearchResult batch_result = flat_search_batch_gpu(
//                             base, batch_queries, h_topk_indices, 
//                             base_number, vecdim, actual_batch_size, 
//                             batch_size, block_size, conv_block_size, strategy,
//                             use_precomputed_transpose
//                         );

//                         // 计算召回率
//                         for (size_t q = 0; q < actual_batch_size; ++q) {
//                             set<uint32_t> gtset;
//                             for (int j = 0; j < k; ++j) {
//                                 gtset.insert(test_gt[(batch_start + q) * test_gt_d + j]);
//                             }

//                             size_t correct = 0;
//                             for (size_t j = 0; j < k; ++j) {
//                                 uint32_t idx = h_topk_indices[q * k + j];
//                                 if (gtset.find(idx) != gtset.end()) {
//                                     correct++;
//                                 }
//                             }

//                             results[batch_start + q] = {
//                                 static_cast<float>(correct) / static_cast<float>(k),
//                                 batch_result.latency / static_cast<int64_t>(actual_batch_size)  // 平均每个查询的延迟
//                             };
//                         }

//                         delete[] batch_queries;
//                     }
                    
//                     // 记录总结束时间
//                     auto total_end_time = chrono::high_resolution_clock::now();
//                     auto total_duration = chrono::duration_cast<chrono::microseconds>(total_end_time - total_start_time).count();
//                     double total_seconds = total_duration / 1000000.0;
//                     double qps = test_number / total_seconds;

//                     float avg_recall = 0.0f;
//                     int64_t avg_latency = 0;
//                     for (const auto& result : results) {
//                         avg_recall += result.recall;
//                         avg_latency += result.latency;
//                     }
//                     avg_recall /= test_number;
//                     avg_latency /= test_number;

//                     SearchResult final_result;
//                     final_result.recall = avg_recall;
//                     final_result.latency = avg_latency;
//                     final_result.qps = qps;
//                     final_result.total_time = total_seconds;
//                     final_result.batch_size = batch_size;
//                     final_result.block_size = block_size;
//                     final_result.conv_block_size = conv_block_size;
//                     final_result.strategy = strategy;
                    
//                     all_results.push_back(final_result);
                    
//                     cout << fixed << setprecision(6);
//                     cout << "Average recall: " << avg_recall << endl;
//                     cout << "Average latency: " << avg_latency << " us" << endl;
//                     cout << "Total processing time: " << total_seconds << " seconds" << endl;
//                     cout << "Queries per second (QPS): " << qps << endl;
                    
//                     delete[] h_topk_indices;
//                 }
//             }
//         }
//     }
    
//     return all_results;
// }

// // 主函数
// int main(int argc, char *argv[]) {
//     bool use_precomputed_transpose = false;
//     if (argc > 1 && string(argv[1]) == "--transpose") {
//         use_precomputed_transpose = true;
//         cout << "Using precomputed transpose optimization\n";
//     }

//     int deviceCount;
//     CHECK_CUDA(cudaGetDeviceCount(&deviceCount));
//     if (deviceCount == 0) {
//         cerr << "Error: No CUDA-capable devices found\n";
//         return EXIT_FAILURE;
//     }

//     int deviceId;
//     CHECK_CUDA(cudaGetDevice(&deviceId));
//     cudaDeviceProp deviceProp;
//     CHECK_CUDA(cudaGetDeviceProperties(&deviceProp, deviceId));
//     cout << "Using CUDA device " << deviceId << ": " << deviceProp.name << "\n";
//     cout << "Compute Capability: " << deviceProp.major << "." << deviceProp.minor << "\n";
//     cout << "Max threads per block: " << deviceProp.maxThreadsPerBlock << "\n";
//     cout << "Max grid dimensions: " << deviceProp.maxGridSize[0] << " x " 
//          << deviceProp.maxGridSize[1] << " x " << deviceProp.maxGridSize[2] << "\n";

//     size_t test_number = 0, base_number = 0, test_gt_d = 0, vecdim = 0;
//     string data_path = "anndata/";
//     auto test_query = LoadData<float>(data_path + "DEEP100K.query.fbin", test_number, vecdim);
//     auto test_gt = LoadData<uint32_t>(data_path + "DEEP100K.gt.query.100k.top100.bin", test_number, test_gt_d);
//     auto base = LoadData<float>(data_path + "DEEP100K.base.100k.fbin", base_number, vecdim);

//     // 限制测试查询数量以提高测试速度
//     const size_t MAX_TEST_QUERIES = 2000;
//     if (test_number > MAX_TEST_QUERIES) {
//         test_number = MAX_TEST_QUERIES;
//         cout << "Limiting to first " << MAX_TEST_QUERIES << " test queries\n";
//     }

//     // 定义要测试的参数组合
//     vector<size_t> batch_sizes = {500,1000,2000};
//     vector<size_t> block_sizes = {64, 128, 256,512};
//     vector<size_t> conv_block_sizes = {8, 16,32};
//     vector<string> strategies = {"Strategy1", "Strategy2"}; // 移除了原Strategy2，原Strategy3改名为Strategy2

//     // 预热GPU
//     cout << "\nWarming up GPU...\n";
//     uint32_t* warmup_indices = new uint32_t[batch_sizes[0] * TOPK];
//     flat_search_batch_gpu(
//         base, test_query, warmup_indices, 
//         base_number, vecdim, min(batch_sizes[0], test_number), 
//         batch_sizes[0], block_sizes[0], conv_block_sizes[0], strategies[0],
//         use_precomputed_transpose
//     );
//     delete[] warmup_indices;

//     // 运行所有参数组合的测试
//     vector<SearchResult> results = run_experiments(
//         base, test_query, test_gt,
//         base_number, test_gt_d, vecdim,
//         test_number, use_precomputed_transpose,
//         batch_sizes, block_sizes, conv_block_sizes, strategies
//     );

//     // 输出所有测试结果
//     cout << "\n\n=== All Results Summary ===" << endl;
//     cout << "==================================================================================================" << endl;
//     cout << "| Strategy   | Batch | TopK Block | Conv Block | Avg Recall | Avg Latency (us) | QPS     |" << endl;
//     cout << "|------------|-------|------------|------------|------------|------------------|---------|" << endl;
    
//     for (const auto& res : results) {
//         cout << "| " << setw(10) << res.strategy
//              << " | " << setw(5) << res.batch_size 
//              << " | " << setw(10) << res.block_size 
//              << " | " << setw(10) << res.conv_block_size
//              << " | " << setw(10) << fixed << setprecision(4) << res.recall 
//              << " | " << setw(16) << res.latency 
//              << " | " << setw(7) << fixed << setprecision(2) << res.qps << " |" << endl;
//     }
//     cout << "==================================================================================================" << endl;

//     // 按策略分组找出最佳配置
//     cout << "\n=== Best Configurations per Strategy ===" << endl;
//     for (const auto& strategy : strategies) {
//         // 找出当前策略下QPS最高的配置
//         auto best_qps = results.end();
//         double max_qps = 0.0;
        
//         for (auto it = results.begin(); it != results.end(); ++it) {
//             if (it->strategy == strategy && it->qps > max_qps) {
//                 best_qps = it;
//                 max_qps = it->qps;
//             }
//         }
        
//         if (best_qps != results.end()) {
//             cout << "Strategy: " << strategy << endl;
//             cout << "  Highest QPS: " << best_qps->qps << endl;
//             cout << "  Batch size: " << best_qps->batch_size 
//                  << ", TopK block size: " << best_qps->block_size
//                  << ", Conv block size: " << best_qps->conv_block_size << endl;
//             cout << "  Avg latency: " << best_qps->latency << " us" << endl;
//             cout << "  Avg recall: " << best_qps->recall << endl;
//             cout << "-----------------------------------" << endl;
//         }
//     }
    
//     // 找出全局最佳配置
//     auto best_qps = max_element(results.begin(), results.end(), 
//         [](const SearchResult& a, const SearchResult& b) {
//             return a.qps < b.qps;
//         });
    
//     auto best_latency = min_element(results.begin(), results.end(), 
//         [](const SearchResult& a, const SearchResult& b) {
//             return a.latency < b.latency;
//         });
    
//     cout << "\n=== Global Best Configurations ===" << endl;
//     cout << "Highest QPS: " << best_qps->qps << " (Strategy: " << best_qps->strategy 
//          << ", Batch: " << best_qps->batch_size 
//          << ", TopK Block: " << best_qps->block_size 
//          << ", Conv Block: " << best_qps->conv_block_size << ")" << endl;
//     cout << "Lowest Latency: " << best_latency->latency << " us (Strategy: " << best_latency->strategy 
//          << ", Batch: " << best_latency->batch_size 
//          << ", TopK Block: " << best_latency->block_size 
//          << ", Conv Block: " << best_latency->conv_block_size << ")" << endl;

//     delete[] test_query;
//     delete[] test_gt;
//     delete[] base;
//     CHECK_CUDA(cudaDeviceReset());

//     return 0;
// }