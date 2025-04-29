// #include <vector>
// #include <cstring>
// #include <string>
// #include <iostream>
// #include <fstream>
// #include <set>
// #include <chrono>
// #include <iomanip>
// #include <sstream>
// #include <sys/time.h>
// #include <omp.h>
// #include "hnswlib/hnswlib/hnswlib.h"
// #include "flat_scan.h"
// // 可以自行添加需要的头文件

// using namespace hnswlib;

// template<typename T>
// T *LoadData(std::string data_path, size_t& n, size_t& d)
// {
//     std::ifstream fin;
//     fin.open(data_path, std::ios::in | std::ios::binary);
//     fin.read((char*)&n,4);
//     fin.read((char*)&d,4);
//     T* data = new T[n*d];
//     int sz = sizeof(T);
//     for(int i = 0; i < n; ++i){
//         fin.read(((char*)data + i*d*sz), d*sz);
//     }
//     fin.close();

//     std::cerr<<"load data "<<data_path<<"\n";
//     std::cerr<<"dimension: "<<d<<"  number:"<<n<<"  size_per_element:"<<sizeof(T)<<"\n";

//     return data;
// }

// struct SearchResult
// {
//     float recall;
//     int64_t latency; // 单位us
// };

// void build_index(float* base, size_t base_number, size_t vecdim)
// {
//     const int efConstruction = 150; // 为防止索引构建时间过长，efc建议设置200以下
//     const int M = 16; // M建议设置为16以下

//     HierarchicalNSW<float> *appr_alg;
//     InnerProductSpace ipspace(vecdim);
//     appr_alg = new HierarchicalNSW<float>(&ipspace, base_number, M, efConstruction);

//     appr_alg->addPoint(base, 0);
//     #pragma omp parallel for
//     for(int i = 1; i < base_number; ++i) {
//         appr_alg->addPoint(base + 1ll*vecdim*i, i);
//     }

//     char path_index[1024] = "files/hnsw.index";
//     appr_alg->saveIndex(path_index);
// }


// int main(int argc, char *argv[])
// {
//     size_t test_number = 0, base_number = 0;
//     size_t test_gt_d = 0, vecdim = 0;

//     std::string data_path = "/anndata/"; 
//     auto test_query = LoadData<float>(data_path + "DEEP100K.query.fbin", test_number, vecdim);
//     auto test_gt = LoadData<int>(data_path + "DEEP100K.gt.query.100k.top100.bin", test_number, test_gt_d);
//     auto base = LoadData<float>(data_path + "DEEP100K.base.100k.fbin", base_number, vecdim);
//     // 只测试前2000条查询
//     test_number = 2000;

//     const size_t k = 10;

//     std::vector<SearchResult> results;
//     results.resize(test_number);

//     // 如果你需要保存索引，可以在这里添加你需要的函数，你可以将下面的注释删除来查看pbs是否将build.index返回到你的files目录中
//     // 要保存的目录必须是files/*
//     // 每个人的目录空间有限，不需要的索引请及时删除，避免占空间太大
//     // 不建议在正式测试查询时同时构建索引，否则性能波动会较大
//     // 下面是一个构建hnsw索引的示例
//     // build_index(base, base_number, vecdim);

    
//     // 查询测试代码
//     for(int i = 0; i < test_number; ++i) {
//         const unsigned long Converter = 1000 * 1000;
//         struct timeval val;
//         int ret = gettimeofday(&val, NULL);

//         // 该文件已有代码中你只能修改该函数的调用方式
//         // 可以任意修改函数名，函数参数或者改为调用成员函数，但是不能修改函数返回值。
//         auto res = flat_search(base, test_query + i*vecdim, base_number, vecdim, k);

//         struct timeval newVal;
//         ret = gettimeofday(&newVal, NULL);
//         int64_t diff = (newVal.tv_sec * Converter + newVal.tv_usec) - (val.tv_sec * Converter + val.tv_usec);

//         std::set<uint32_t> gtset;
//         for(int j = 0; j < k; ++j){
//             int t = test_gt[j + i*test_gt_d];
//             gtset.insert(t);
//         }

//         size_t acc = 0;
//         while (res.size()) {   
//             int x = res.top().second;
//             if(gtset.find(x) != gtset.end()){
//                 ++acc;
//             }
//             res.pop();
//         }
//         float recall = (float)acc/k;

//         results[i] = {recall, diff};
//     }

//     float avg_recall = 0, avg_latency = 0;
//     for(int i = 0; i < test_number; ++i) {
//         avg_recall += results[i].recall;
//         avg_latency += results[i].latency;
//     }

//     // 浮点误差可能导致一些精确算法平均recall不是1
//     std::cout << "average recall: "<<avg_recall / test_number<<"\n";
//     std::cout << "average latency (us): "<<avg_latency / test_number<<"\n";
//     return 0;
// }

//没有预计算的方式
// #include <iostream>
// #include <fstream>
// #include <vector>
// #include <queue>
// #include <algorithm>
// #include <cmath>
// #include <cassert>
// #include <sys/time.h>
// #include <set>

// // 读取码本文件
// std::vector<std::vector<std::vector<float>>> read_codebooks(const std::string& filename) {
//     std::ifstream file(filename, std::ios::binary);
//     int M, sub_dim;
//     file.read(reinterpret_cast<char*>(&M), sizeof(int));
//     file.read(reinterpret_cast<char*>(&sub_dim), sizeof(int));

//     std::vector<std::vector<std::vector<float>>> codebooks(M, std::vector<std::vector<float>>(256, std::vector<float>(sub_dim)));
//     for (int m = 0; m < M; ++m) {
//         for (int k = 0; k < 256; ++k) {
//             for (int d = 0; d < sub_dim; ++d) {
//                 file.read(reinterpret_cast<char*>(&codebooks[m][k][d]), sizeof(float));
//             }
//         }
//     }
//     return codebooks;
// }

// // 加载 PQ 编码文件
// std::pair<std::vector<std::vector<uint8_t>>, std::pair<int, int>> load_pq_codes(const std::string& filename) {
//     std::ifstream file(filename, std::ios::binary);
//     int n, M;
//     file.read(reinterpret_cast<char*>(&n), sizeof(int));
//     file.read(reinterpret_cast<char*>(&M), sizeof(int));

//     std::vector<std::vector<uint8_t>> pq_codes(n, std::vector<uint8_t>(M));
//     for (int i = 0; i < n; ++i) {
//         for (int m = 0; m < M; ++m) {
//             file.read(reinterpret_cast<char*>(&pq_codes[i][m]), sizeof(uint8_t));
//         }
//     }
//     return std::make_pair(pq_codes, std::make_pair(n, M));
// }

// // 暴力搜索函数（保持与你原始代码一致）
// std::priority_queue<std::pair<float, uint32_t>> flat_search(float* base, float* query, size_t base_number, size_t vecdim, size_t k) {
//     std::priority_queue<std::pair<float, uint32_t>> q;
//     for (size_t i = 0; i < base_number; ++i) {
//         float dis = 0;
//         for (size_t d = 0; d < vecdim; ++d) {
//             dis += base[i * vecdim + d] * query[d];
//         }
//         dis = 1 - dis;

//         if (q.size() < k) {
//             q.push({dis, static_cast<uint32_t>(i)});
//         } else {
//             if (dis < q.top().first) {
//                 q.push({dis, static_cast<uint32_t>(i)});
//                 q.pop();
//             }
//         }
//     }
//     return q;
// }

// // 带 rerank 的 PQ 搜索算法
// std::priority_queue<std::pair<float, uint32_t>> pq_search_rerank(float* query, const std::vector<std::vector<uint8_t>>& pq_codes, const std::vector<std::vector<std::vector<float>>>& codebooks, float* base_vectors, size_t M, size_t sub_dim, size_t k, size_t rerank, size_t vecdim) {
//     size_t n = pq_codes.size();
//     std::vector<float> pq_scores(n, 0.0);

//     for (size_t m = 0; m < M; ++m) {
//         float* sub_query = query + m * sub_dim;
//         std::vector<float> table(256);
//         for (int k_idx = 0; k_idx < 256; ++k_idx) {
//             for (size_t d = 0; d < sub_dim; ++d) {
//                 table[k_idx] += codebooks[m][k_idx][d] * sub_query[d];
//             }
//         }
//         for (size_t i = 0; i < n; ++i) {
//             pq_scores[i] += table[pq_codes[i][m]];
//         }
//     }

//     std::vector<size_t> indices(n);
//     for (size_t i = 0; i < n; ++i) indices[i] = i;
//     std::nth_element(indices.begin(), indices.begin() + rerank, indices.end(), [&pq_scores](size_t a, size_t b) { return pq_scores[a] > pq_scores[b]; });

//     std::priority_queue<std::pair<float, uint32_t>> final_result;
//     for (size_t i = 0; i < rerank; ++i) {  // 直接取前rerank个索引
//         size_t idx = indices[i];
//         float* candidate = base_vectors + idx * vecdim;
//         float dis = 0;
//         for (size_t d = 0; d < vecdim; ++d) {
//             dis += candidate[d] * query[d];
//         }
//         dis = 1 - dis;
//         if (final_result.size() < k) {
//             final_result.push({dis, static_cast<uint32_t>(idx)});
//         } else if (dis < final_result.top().first) {
//             final_result.pop();
//             final_result.push({dis, static_cast<uint32_t>(idx)});
//         }
//     }

//     return final_result;
// }

// // 加载数据函数
// template<typename T>
// T* LoadData(const std::string& data_path, size_t& n, size_t& d) {
//     std::ifstream fin(data_path, std::ios::in | std::ios::binary);
//     fin.read(reinterpret_cast<char*>(&n), sizeof(int));
//     fin.read(reinterpret_cast<char*>(&d), sizeof(int));
//     T* data = new T[n * d];
//     for (size_t i = 0; i < n; ++i) {
//         fin.read(reinterpret_cast<char*>(data + i * d), d * sizeof(T));
//     }
//     fin.close();
//     std::cerr << "load data " << data_path << "\n";
//     std::cerr << "dimension: " << d << "  number:" << n << "  size_per_element:" << sizeof(T) << "\n";
//     return data;
// }

// int main() {
//     size_t test_number = 0, base_number = 0, test_gt_d = 0, vecdim = 0;
//     std::string data_path = "/anndata/";

//     float* test_query = LoadData<float>(data_path + "DEEP100K.query.fbin", test_number, vecdim);
//     int* test_gt = LoadData<int>(data_path + "DEEP100K.gt.query.100k.top100.bin", test_number, test_gt_d);
//     float* base = LoadData<float>(data_path + "DEEP100K.base.100k.fbin", base_number, vecdim);

//     const size_t k = 10;
//     std::vector<size_t> rerank_values = {10, 50, 100, 200,300,400,500,600,700,800,900,1000};

//     // 加载预处理文件
//     auto codebooks = read_codebooks("files/codebooks.bin");
//     auto pq_codes_pair = load_pq_codes("files/pq_codes.bin");
//     const auto& pq_codes = pq_codes_pair.first;
//     const size_t M = codebooks.size();
//     const size_t sub_dim = codebooks[0][0].size();

//     for (size_t rerank : rerank_values) {
//         double total_recall = 0.0;
//         long long total_latency = 0;
//         struct timeval start_all, end_all;
//         gettimeofday(&start_all, nullptr);  // 记录整个rerank的开始时间

//         for (size_t i = 0; i < test_number; ++i) {
//             struct timeval start, end;
//             gettimeofday(&start, nullptr);

//             auto result = pq_search_rerank(
//                 test_query + i * vecdim,
//                 pq_codes,
//                 codebooks,
//                 base,
//                 M,
//                 sub_dim,
//                 k,
//                 rerank,
//                 vecdim
//             );

//             gettimeofday(&end, nullptr);
//             long long latency = (end.tv_sec * 1000000 + end.tv_usec) - (start.tv_sec * 1000000 + start.tv_usec);
//             total_latency += latency;

//             // 计算召回率
//             std::set<uint32_t> gtset;
//             for (size_t j = 0; j < k; ++j) {
//                 gtset.insert(static_cast<uint32_t>(test_gt[i * test_gt_d + j]));
//             }
//             size_t correct = 0;
//             while (!result.empty()) {
//                 if (gtset.count(result.top().second)) correct++;
//                 result.pop();
//             }
//             total_recall += static_cast<float>(correct) / k;
//         }

//         gettimeofday(&end_all, nullptr);
//         double average_recall = total_recall / test_number;
//         long long average_latency = total_latency / test_number;

//         std::cout << "Rerank: " << rerank << "\tAverage Recall: " << average_recall << "\tAverage Latency: " << average_latency << " us\n";
//     }

//     delete[] test_query;
//     delete[] test_gt;
//     delete[] base;
//     return 0;
// }

//无查询量化neon优化版本
// #include <iostream>
// #include <fstream>
// #include <vector>
// #include <queue>
// #include <algorithm>
// #include <cmath>
// #include <cassert>
// #include <sys/time.h>
// #include <set>
// #include <arm_neon.h>

// // 读取码本文件
// std::vector<std::vector<std::vector<float>>> read_codebooks(const std::string& filename) {
//     std::ifstream file(filename, std::ios::binary);
//     int M, sub_dim;
//     file.read(reinterpret_cast<char*>(&M), sizeof(int));
//     file.read(reinterpret_cast<char*>(&sub_dim), sizeof(int));

//     std::vector<std::vector<std::vector<float>>> codebooks(M, std::vector<std::vector<float>>(256, std::vector<float>(sub_dim)));
//     for (int m = 0; m < M; ++m) {
//         for (int k = 0; k < 256; ++k) {
//             for (int d = 0; d < sub_dim; ++d) {
//                 file.read(reinterpret_cast<char*>(&codebooks[m][k][d]), sizeof(float));
//             }
//         }
//     }
//     return codebooks;
// }

// // 加载 PQ 编码文件
// std::pair<std::vector<std::vector<uint8_t>>, std::pair<int, int>> load_pq_codes(const std::string& filename) {
//     std::ifstream file(filename, std::ios::binary);
//     int n, M;
//     file.read(reinterpret_cast<char*>(&n), sizeof(int));
//     file.read(reinterpret_cast<char*>(&M), sizeof(int));

//     std::vector<std::vector<uint8_t>> pq_codes(n, std::vector<uint8_t>(M));
//     for (int i = 0; i < n; ++i) {
//         for (int m = 0; m < M; ++m) {
//             file.read(reinterpret_cast<char*>(&pq_codes[i][m]), sizeof(uint8_t));
//         }
//     }
//     return std::make_pair(pq_codes, std::make_pair(n, M));
// }

// // 暴力搜索函数，使用NEON优化
// std::priority_queue<std::pair<float, uint32_t>> flat_search(float* base, float* query, size_t base_number, size_t vecdim, size_t k) {
//     std::priority_queue<std::pair<float, uint32_t>> q;
//     const size_t vecdim_aligned = vecdim & ~3;  // 确保向量维度是4的倍数
//     for (size_t i = 0; i < base_number; ++i) {
//         float32x4_t sum_vec = vdupq_n_f32(0.0f);
//         for (size_t d = 0; d < vecdim_aligned; d += 4) {
//             float32x4_t base_vec = vld1q_f32(base + i * vecdim + d);
//             float32x4_t query_vec = vld1q_f32(query + d);
//             sum_vec = vmlaq_f32(sum_vec, base_vec, query_vec);
//         }
//         float sum = vaddvq_f32(sum_vec);
//         // 处理剩余的维度
//         for (size_t d = vecdim_aligned; d < vecdim; ++d) {
//             sum += base[i * vecdim + d] * query[d];
//         }
//         float dis = 1 - sum;

//         if (q.size() < k) {
//             q.push({dis, static_cast<uint32_t>(i)});
//         } else {
//             if (dis < q.top().first) {
//                 q.push({dis, static_cast<uint32_t>(i)});
//                 q.pop();
//             }
//         }
//     }
//     return q;
// }

// // 带 rerank 的 PQ 搜索算法，使用NEON优化
// std::priority_queue<std::pair<float, uint32_t>> pq_search_rerank(float* query, const std::vector<std::vector<uint8_t>>& pq_codes, const std::vector<std::vector<std::vector<float>>>& codebooks, float* base_vectors, size_t M, size_t sub_dim, size_t k, size_t rerank, size_t vecdim) {
//     size_t n = pq_codes.size();
//     std::vector<float> pq_scores(n, 0.0);

//     for (size_t m = 0; m < M; ++m) {
//         float* sub_query = query + m * sub_dim;
//         std::vector<float> table(256);
//         for (int k_idx = 0; k_idx < 256; ++k_idx) {
//             float32x4_t sum_vec = vdupq_n_f32(0.0f);
//             const size_t sub_dim_aligned = sub_dim & ~3;
//             for (size_t d = 0; d < sub_dim_aligned; d += 4) {
//                 float32x4_t codebook_vec = vld1q_f32(&codebooks[m][k_idx][d]);
//                 float32x4_t sub_query_vec = vld1q_f32(sub_query + d);
//                 sum_vec = vmlaq_f32(sum_vec, codebook_vec, sub_query_vec);
//             }
//             float sum = vaddvq_f32(sum_vec);
//             // 处理剩余的维度
//             for (size_t d = sub_dim_aligned; d < sub_dim; ++d) {
//                 sum += codebooks[m][k_idx][d] * sub_query[d];
//             }
//             table[k_idx] = sum;
//         }
//         for (size_t i = 0; i < n; ++i) {
//             pq_scores[i] += table[pq_codes[i][m]];
//         }
//     }

//     std::vector<size_t> indices(n);
//     for (size_t i = 0; i < n; ++i) indices[i] = i;
//     std::nth_element(indices.begin(), indices.begin() + rerank, indices.end(), [&pq_scores](size_t a, size_t b) { return pq_scores[a] > pq_scores[b]; });

//     std::priority_queue<std::pair<float, uint32_t>> final_result;
//     const size_t vecdim_aligned = vecdim & ~3;
//     for (size_t i = 0; i < rerank; ++i) {  // 直接取前rerank个索引
//         size_t idx = indices[i];
//         float32x4_t sum_vec = vdupq_n_f32(0.0f);
//         for (size_t d = 0; d < vecdim_aligned; d += 4) {
//             float32x4_t candidate_vec = vld1q_f32(base_vectors + idx * vecdim + d);
//             float32x4_t query_vec = vld1q_f32(query + d);
//             sum_vec = vmlaq_f32(sum_vec, candidate_vec, query_vec);
//         }
//         float sum = vaddvq_f32(sum_vec);
//         // 处理剩余的维度
//         for (size_t d = vecdim_aligned; d < vecdim; ++d) {
//             sum += base_vectors[idx * vecdim + d] * query[d];
//         }
//         float dis = 1 - sum;
//         if (final_result.size() < k) {
//             final_result.push({dis, static_cast<uint32_t>(idx)});
//         } else if (dis < final_result.top().first) {
//             final_result.pop();
//             final_result.push({dis, static_cast<uint32_t>(idx)});
//         }
//     }

//     return final_result;
// }

// // 加载数据函数
// template<typename T>
// T* LoadData(const std::string& data_path, size_t& n, size_t& d) {
//     std::ifstream fin(data_path, std::ios::in | std::ios::binary);
//     fin.read(reinterpret_cast<char*>(&n), sizeof(int));
//     fin.read(reinterpret_cast<char*>(&d), sizeof(int));
//     T* data = new T[n * d];
//     for (size_t i = 0; i < n; ++i) {
//         fin.read(reinterpret_cast<char*>(data + i * d), d * sizeof(T));
//     }
//     fin.close();
//     std::cerr << "load data " << data_path << "\n";
//     std::cerr << "dimension: " << d << "  number:" << n << "  size_per_element:" << sizeof(T) << "\n";
//     return data;
// }

// int main() {
//     size_t test_number = 0, base_number = 0, test_gt_d = 0, vecdim = 0;
//     std::string data_path = "/anndata/";

//     float* test_query = LoadData<float>(data_path + "DEEP100K.query.fbin", test_number, vecdim);
//     int* test_gt = LoadData<int>(data_path + "DEEP100K.gt.query.100k.top100.bin", test_number, test_gt_d);
//     float* base = LoadData<float>(data_path + "DEEP100K.base.100k.fbin", base_number, vecdim);

//     const size_t k = 10;
//     std::vector<size_t> rerank_values = {10, 50, 100, 200,300,400,500,600,700,800,900,1000,1200,1400,1600,1800,2000};

//     // 加载预处理文件
//     auto codebooks = read_codebooks("files/codebooks.bin");
//     auto pq_codes_pair = load_pq_codes("files/pq_codes.bin");
//     const auto& pq_codes = pq_codes_pair.first;
//     const size_t M = codebooks.size();
//     const size_t sub_dim = codebooks[0][0].size();

//     for (size_t rerank : rerank_values) {
//         double total_recall = 0.0;
//         long long total_latency = 0;
//         struct timeval start_all, end_all;
//         gettimeofday(&start_all, nullptr);  // 记录整个rerank的开始时间

//         for (size_t i = 0; i < test_number; ++i) {
//             struct timeval start, end;
//             gettimeofday(&start, nullptr);

//             auto result = pq_search_rerank(
//                 test_query + i * vecdim,
//                 pq_codes,
//                 codebooks,
//                 base,
//                 M,
//                 sub_dim,
//                 k,
//                 rerank,
//                 vecdim
//             );

//             gettimeofday(&end, nullptr);
//             long long latency = (end.tv_sec * 1000000 + end.tv_usec) - (start.tv_sec * 1000000 + start.tv_usec);
//             total_latency += latency;

//             // 计算召回率
//             std::set<uint32_t> gtset;
//             for (size_t j = 0; j < k; ++j) {
//                 gtset.insert(static_cast<uint32_t>(test_gt[i * test_gt_d + j]));
//             }
//             size_t correct = 0;
//             while (!result.empty()) {
//                 if (gtset.count(result.top().second)) correct++;
//                 result.pop();
//             }
//             total_recall += static_cast<float>(correct) / k;
//         }

//         gettimeofday(&end_all, nullptr);
//         double average_recall = total_recall / test_number;
//         long long average_latency = total_latency / test_number;

//         std::cout << "Rerank: " << rerank << "\tAverage Recall: " << average_recall << "\tAverage Latency: " << average_latency << " us\n";
//     }

//     delete[] test_query;
//     delete[] test_gt;
//     delete[] base;
//     return 0;
// }    

//有预计算无优化版本
// #include <iostream>
// #include <fstream>
// #include <vector>
// #include <queue>
// #include <algorithm>
// #include <cmath>
// #include <cassert>
// #include <sys/time.h>
// #include <set>

// // 读取码本文件
// std::vector<std::vector<std::vector<float>>> read_codebooks(const std::string& filename) {
//     std::ifstream file(filename, std::ios::binary);
//     int M, sub_dim;
//     file.read(reinterpret_cast<char*>(&M), sizeof(int));
//     file.read(reinterpret_cast<char*>(&sub_dim), sizeof(int));

//     std::vector<std::vector<std::vector<float>>> codebooks(M, std::vector<std::vector<float>>(256, std::vector<float>(sub_dim)));
//     for (int m = 0; m < M; ++m) {
//         for (int k = 0; k < 256; ++k) {
//             for (int d = 0; d < sub_dim; ++d) {
//                 file.read(reinterpret_cast<char*>(&codebooks[m][k][d]), sizeof(float));
//             }
//         }
//     }
//     return codebooks;
// }

// // 加载 PQ 编码文件
// std::pair<std::vector<std::vector<uint8_t>>, std::pair<int, int>> load_pq_codes(const std::string& filename) {
//     std::ifstream file(filename, std::ios::binary);
//     int n, M;
//     file.read(reinterpret_cast<char*>(&n), sizeof(int));
//     file.read(reinterpret_cast<char*>(&M), sizeof(int));

//     std::vector<std::vector<uint8_t>> pq_codes(n, std::vector<uint8_t>(M));
//     for (int i = 0; i < n; ++i) {
//         for (int m = 0; m < M; ++m) {
//             file.read(reinterpret_cast<char*>(&pq_codes[i][m]), sizeof(uint8_t));
//         }
//     }
//     return std::make_pair(pq_codes, std::make_pair(n, M));
// }

// // 读取预计算的簇内积表（对称矩阵）
// std::vector<std::vector<std::vector<float>>> read_cluster_products(const std::string& filename) {
//     std::ifstream file(filename, std::ios::binary);
//     int M;
//     file.read(reinterpret_cast<char*>(&M), sizeof(int));

//     std::vector<std::vector<std::vector<float>>> products(M, std::vector<std::vector<float>>(256, std::vector<float>(256, 0.0f)));
//     for (int m = 0; m < M; ++m) {
//         for (int i = 0; i < 256; ++i) {
//             for (int j = i; j < 256; ++j) {
//                 float val;
//                 file.read(reinterpret_cast<char*>(&val), sizeof(float));
//                 products[m][i][j] = val;
//                 products[m][j][i] = val;  // 补全下三角
//             }
//         }
//     }
//     return products;
// }

// // 暴力搜索函数
// std::priority_queue<std::pair<float, uint32_t>> flat_search(float* base, float* query, size_t base_number, size_t vecdim, size_t k) {
//     std::priority_queue<std::pair<float, uint32_t>> q;
//     for (size_t i = 0; i < base_number; ++i) {
//         float sum = 0.0;
//         for (size_t d = 0; d < vecdim; ++d) {
//             sum += base[i * vecdim + d] * query[d];
//         }
//         float dis = 1 - sum;
//         if (q.size() < k) {
//             q.push({dis, static_cast<uint32_t>(i)});
//         } else if (dis < q.top().first) {
//             q.push({dis, static_cast<uint32_t>(i)});
//             q.pop();
//         }
//     }
//     return q;
// }

// // 带 Rerank 的 PQ 搜索（核心修改：使用预计算内积）
// std::priority_queue<std::pair<float, uint32_t>> pq_search_rerank(
//     float* query, 
//     const std::vector<std::vector<uint8_t>>& pq_codes, 
//     const std::vector<std::vector<std::vector<float>>>& codebooks,
//     const std::vector<std::vector<std::vector<float>>>& cluster_products,  // 新增：预计算内积表
//     float* base_vectors, 
//     size_t M, 
//     size_t sub_dim, 
//     size_t k, 
//     size_t rerank, 
//     size_t vecdim
// ) {
//     size_t n = pq_codes.size();
//     std::vector<float> pq_scores(n, 0.0);

//     // 1. 量化查询向量（与原代码一致，实时计算查询的量化码）
//     std::vector<uint8_t> q_code(M);
//     for (size_t m = 0; m < M; ++m) {
//         float* sub_query = query + m * sub_dim;
//         float max_inner = -1e9;
//         uint8_t best_code = 0;
//         for (int k_idx = 0; k_idx < 256; ++k_idx) {
//             float inner = 0.0;
//             for (size_t d = 0; d < sub_dim; ++d) {
//                 inner += codebooks[m][k_idx][d] * sub_query[d];
//             }
//             if (inner > max_inner) {
//                 max_inner = inner;
//                 best_code = k_idx;
//             }
//         }
//         q_code[m] = best_code;
//     }

//     // 2. 使用预计算内积计算 PQ 得分（核心修改：查表替代实时计算）
//     for (size_t i = 0; i < n; ++i) {
//         float score = 0.0;
//         for (size_t m = 0; m < M; ++m) {
//             uint8_t q_idx = q_code[m];        // 查询在子空间 m 的量化码
//             uint8_t db_idx = pq_codes[i][m];  // 数据库向量在子空间 m 的量化码
//             score += cluster_products[m][q_idx][db_idx];  // 直接查表
//         }
//         pq_scores[i] = score;
//     }

//     // 3. 筛选候选集（与原代码一致）
//     std::vector<size_t> indices(n);
//     for (size_t i = 0; i < n; ++i) indices[i] = i;
//     std::nth_element(indices.begin(), indices.begin() + rerank, indices.end(),
//                      [&pq_scores](size_t a, size_t b) { return pq_scores[a] > pq_scores[b]; });

//     // 4. 全精度重排（去掉NEON优化）
//     std::priority_queue<std::pair<float, uint32_t>> final_result;
//     for (size_t i = 0; i < rerank; ++i) {
//         size_t idx = indices[i];
//         float sum = 0.0;
//         for (size_t d = 0; d < vecdim; ++d) {
//             sum += base_vectors[idx * vecdim + d] * query[d];
//         }
//         float dis = 1 - sum;
//         if (final_result.size() < k) {
//             final_result.push({dis, static_cast<uint32_t>(idx)});
//         } else if (dis < final_result.top().first) {
//             final_result.pop();
//             final_result.push({dis, static_cast<uint32_t>(idx)});
//         }
//     }

//     return final_result;
// }

// // 加载数据函数（原代码保留）
// template<typename T>
// T* LoadData(const std::string& data_path, size_t& n, size_t& d) {
//     std::ifstream fin(data_path, std::ios::in | std::ios::binary);
//     fin.read(reinterpret_cast<char*>(&n), sizeof(int));
//     fin.read(reinterpret_cast<char*>(&d), sizeof(int));
//     T* data = new T[n * d];
//     for (size_t i = 0; i < n; ++i) {
//         fin.read(reinterpret_cast<char*>(data + i * d), d * sizeof(T));
//     }
//     fin.close();
//     std::cerr << "load data " << data_path << "\n";
//     std::cerr << "dimension: " << d << "  number:" << n << "  size_per_element:" << sizeof(T) << "\n";
//     return data;
// }

// int main() {
//     size_t test_number = 0, base_number = 0, test_gt_d = 0, vecdim = 0;
//     std::string data_path = "/anndata/";

//     // 加载数据（原代码保留）
//     float* test_query = LoadData<float>(data_path + "DEEP100K.query.fbin", test_number, vecdim);
//     int* test_gt = LoadData<int>(data_path + "DEEP100K.gt.query.100k.top100.bin", test_number, test_gt_d);
//     float* base = LoadData<float>(data_path + "DEEP100K.base.100k.fbin", base_number, vecdim);

//     const size_t k = 10;
//     std::vector<size_t> rerank_values = {10,20,30,40,50,60,70,80,90,100,150,200,300,400,500,600,700,800,1000,2000,3000,4000};

//     // 加载预处理文件（新增：预计算簇内积）
//     auto codebooks = read_codebooks("files/codebooks.bin");
//     auto pq_codes_pair = load_pq_codes("files/pq_codes.bin");
//     auto cluster_products = read_cluster_products("files/cluster_products.bin");  // 新增
//     const auto& pq_codes = pq_codes_pair.first;
//     const size_t M = codebooks.size();
//     const size_t sub_dim = codebooks[0][0].size();

//     for (size_t rerank : rerank_values) {
//         double total_recall = 0.0;
//         long long total_latency = 0;

//         for (size_t i = 0; i < test_number; ++i) {
//             struct timeval start, end;
//             gettimeofday(&start, nullptr);

//             // 调用修改后的 pq_search_rerank，传入 cluster_products
//             auto result = pq_search_rerank(
//                 test_query + i * vecdim,
//                 pq_codes,
//                 codebooks,
//                 cluster_products,  // 传入预计算内积表
//                 base,
//                 M,
//                 sub_dim,
//                 k,
//                 rerank,
//                 vecdim
//             );

//             gettimeofday(&end, nullptr);
//             long long latency = (end.tv_sec * 1000000 + end.tv_usec) - (start.tv_sec * 1000000 + start.tv_usec);
//             total_latency += latency;

//             // 计算召回率（原代码保留）
//             std::set<uint32_t> gtset;
//             for (size_t j = 0; j < k; ++j) {
//                 gtset.insert(static_cast<uint32_t>(test_gt[i * test_gt_d + j]));
//             }
//             size_t correct = 0;
//             while (!result.empty()) {
//                 if (gtset.count(result.top().second)) correct++;
//                 result.pop();
//             }
//             total_recall += static_cast<float>(correct) / k;
//         }

//         double average_recall = total_recall / test_number;
//         long long average_latency = total_latency / test_number;

//         std::cout << "Rerank: " << rerank << "\tAverage Recall: " << average_recall << "\tAverage Latency: " << average_latency << " us\n";
//     }

//     delete[] test_query;
//     delete[] test_gt;
//     delete[] base;
//     return 0;
// }
// //实现了预读取内积但是没有向量化并行查询的代码,用了neon优化rerank的内基向量计算
// #include <iostream>
// #include <fstream>
// #include <vector>
// #include <queue>
// #include <algorithm>
// #include <cmath>
// #include <cassert>
// #include <sys/time.h>
// #include <set>
// #include <arm_neon.h>

// // 读取码本文件
// std::vector<std::vector<std::vector<float>>> read_codebooks(const std::string& filename) {
//     std::ifstream file(filename, std::ios::binary);
//     int M, sub_dim;
//     file.read(reinterpret_cast<char*>(&M), sizeof(int));
//     file.read(reinterpret_cast<char*>(&sub_dim), sizeof(int));

//     std::vector<std::vector<std::vector<float>>> codebooks(M, std::vector<std::vector<float>>(256, std::vector<float>(sub_dim)));
//     for (int m = 0; m < M; ++m) {
//         for (int k = 0; k < 256; ++k) {
//             for (int d = 0; d < sub_dim; ++d) {
//                 file.read(reinterpret_cast<char*>(&codebooks[m][k][d]), sizeof(float));
//             }
//         }
//     }
//     return codebooks;
// }

// // 加载 PQ 编码文件
// std::pair<std::vector<std::vector<uint8_t>>, std::pair<int, int>> load_pq_codes(const std::string& filename) {
//     std::ifstream file(filename, std::ios::binary);
//     int n, M;
//     file.read(reinterpret_cast<char*>(&n), sizeof(int));
//     file.read(reinterpret_cast<char*>(&M), sizeof(int));

//     std::vector<std::vector<uint8_t>> pq_codes(n, std::vector<uint8_t>(M));
//     for (int i = 0; i < n; ++i) {
//         for (int m = 0; m < M; ++m) {
//             file.read(reinterpret_cast<char*>(&pq_codes[i][m]), sizeof(uint8_t));
//         }
//     }
//     return std::make_pair(pq_codes, std::make_pair(n, M));
// }

// // 读取预计算的簇内积表（对称矩阵）
// std::vector<std::vector<std::vector<float>>> read_cluster_products(const std::string& filename) {
//     std::ifstream file(filename, std::ios::binary);
//     int M;
//     file.read(reinterpret_cast<char*>(&M), sizeof(int));

//     std::vector<std::vector<std::vector<float>>> products(M, std::vector<std::vector<float>>(256, std::vector<float>(256, 0.0f)));
//     for (int m = 0; m < M; ++m) {
//         for (int i = 0; i < 256; ++i) {
//             for (int j = i; j < 256; ++j) {
//                 float val;
//                 file.read(reinterpret_cast<char*>(&val), sizeof(float));
//                 products[m][i][j] = val;
//                 products[m][j][i] = val;  // 补全下三角
//             }
//         }
//     }
//     return products;
// }

// // 暴力搜索函数（NEON 优化，第二种代码保留）
// std::priority_queue<std::pair<float, uint32_t>> flat_search(float* base, float* query, size_t base_number, size_t vecdim, size_t k) {
//     std::priority_queue<std::pair<float, uint32_t>> q;
//     const size_t vecdim_aligned = vecdim & ~3;
//     for (size_t i = 0; i < base_number; ++i) {
//         float32x4_t sum_vec = vdupq_n_f32(0.0f);
//         for (size_t d = 0; d < vecdim_aligned; d += 4) {
//             float32x4_t base_vec = vld1q_f32(base + i * vecdim + d);
//             float32x4_t query_vec = vld1q_f32(query + d);
//             sum_vec = vmlaq_f32(sum_vec, base_vec, query_vec);
//         }
//         float sum = vaddvq_f32(sum_vec);
//         for (size_t d = vecdim_aligned; d < vecdim; ++d) {
//             sum += base[i * vecdim + d] * query[d];
//         }
//         float dis = 1 - sum;
//         if (q.size() < k) {
//             q.push({dis, static_cast<uint32_t>(i)});
//         } else if (dis < q.top().first) {
//             q.push({dis, static_cast<uint32_t>(i)});
//             q.pop();
//         }
//     }
//     return q;
// }

// // 带 Rerank 的 PQ 搜索（核心修改：使用预计算内积）
// std::priority_queue<std::pair<float, uint32_t>> pq_search_rerank(
//     float* query, 
//     const std::vector<std::vector<uint8_t>>& pq_codes, 
//     const std::vector<std::vector<std::vector<float>>>& codebooks,
//     const std::vector<std::vector<std::vector<float>>>& cluster_products,  // 新增：预计算内积表
//     float* base_vectors, 
//     size_t M, 
//     size_t sub_dim, 
//     size_t k, 
//     size_t rerank, 
//     size_t vecdim
// ) {
//     size_t n = pq_codes.size();
//     std::vector<float> pq_scores(n, 0.0);

//     // 1. 量化查询向量（与第二种代码一致，实时计算查询的量化码）
//     std::vector<uint8_t> q_code(M);
//     for (size_t m = 0; m < M; ++m) {
//         float* sub_query = query + m * sub_dim;
//         float max_inner = -1e9;
//         uint8_t best_code = 0;
//         for (int k_idx = 0; k_idx < 256; ++k_idx) {
//             float inner = 0.0;
//             for (size_t d = 0; d < sub_dim; ++d) {
//                 inner += codebooks[m][k_idx][d] * sub_query[d];
//             }
//             if (inner > max_inner) {
//                 max_inner = inner;
//                 best_code = k_idx;
//             }
//         }
//         q_code[m] = best_code;
//     }

//     // 2. 使用预计算内积计算 PQ 得分（核心修改：查表替代实时计算）
//     for (size_t i = 0; i < n; ++i) {
//         float score = 0.0;
//         for (size_t m = 0; m < M; ++m) {
//             uint8_t q_idx = q_code[m];        // 查询在子空间 m 的量化码
//             uint8_t db_idx = pq_codes[i][m];  // 数据库向量在子空间 m 的量化码
//             score += cluster_products[m][q_idx][db_idx];  // 直接查表
//         }
//         pq_scores[i] = score;
//     }

//     // 3. 筛选候选集（与第二种代码一致）
//     std::vector<size_t> indices(n);
//     for (size_t i = 0; i < n; ++i) indices[i] = i;
//     std::nth_element(indices.begin(), indices.begin() + rerank, indices.end(),
//                      [&pq_scores](size_t a, size_t b) { return pq_scores[a] > pq_scores[b]; });

//     // 4. 全精度重排（NEON 优化，与第二种代码一致）
//     std::priority_queue<std::pair<float, uint32_t>> final_result;
//     const size_t vecdim_aligned = vecdim & ~3;
//     for (size_t i = 0; i < rerank; ++i) {
//         size_t idx = indices[i];
//         float32x4_t sum_vec = vdupq_n_f32(0.0f);
//         for (size_t d = 0; d < vecdim_aligned; d += 4) {
//             float32x4_t candidate_vec = vld1q_f32(base_vectors + idx * vecdim + d);
//             float32x4_t query_vec = vld1q_f32(query + d);
//             sum_vec = vmlaq_f32(sum_vec, candidate_vec, query_vec);
//         }
//         float sum = vaddvq_f32(sum_vec);
//         for (size_t d = vecdim_aligned; d < vecdim; ++d) {
//             sum += base_vectors[idx * vecdim + d] * query[d];
//         }
//         float dis = 1 - sum;
//         if (final_result.size() < k) {
//             final_result.push({dis, static_cast<uint32_t>(idx)});
//         } else if (dis < final_result.top().first) {
//             final_result.pop();
//             final_result.push({dis, static_cast<uint32_t>(idx)});
//         }
//     }

//     return final_result;
// }

// // 加载数据函数（第二种代码保留）
// template<typename T>
// T* LoadData(const std::string& data_path, size_t& n, size_t& d) {
//     std::ifstream fin(data_path, std::ios::in | std::ios::binary);
//     fin.read(reinterpret_cast<char*>(&n), sizeof(int));
//     fin.read(reinterpret_cast<char*>(&d), sizeof(int));
//     T* data = new T[n * d];
//     for (size_t i = 0; i < n; ++i) {
//         fin.read(reinterpret_cast<char*>(data + i * d), d * sizeof(T));
//     }
//     fin.close();
//     std::cerr << "load data " << data_path << "\n";
//     std::cerr << "dimension: " << d << "  number:" << n << "  size_per_element:" << sizeof(T) << "\n";
//     return data;
// }

// int main() {
//     size_t test_number = 0, base_number = 0, test_gt_d = 0, vecdim = 0;
//     std::string data_path = "/anndata/";

//     // 加载数据（第二种代码保留）
//     float* test_query = LoadData<float>(data_path + "DEEP100K.query.fbin", test_number, vecdim);
//     int* test_gt = LoadData<int>(data_path + "DEEP100K.gt.query.100k.top100.bin", test_number, test_gt_d);
//     float* base = LoadData<float>(data_path + "DEEP100K.base.100k.fbin", base_number, vecdim);

//     const size_t k = 10;
//     std::vector<size_t> rerank_values = {10,20,30,40,50,60,70,80,90,100,150,200,300,400,500,600,700,800,1000,2000,3000,4000};

//     // 加载预处理文件（新增：预计算簇内积）
//     auto codebooks = read_codebooks("files/codebooks.bin");
//     auto pq_codes_pair = load_pq_codes("files/pq_codes.bin");
//     auto cluster_products = read_cluster_products("files/cluster_products.bin");  // 新增
//     const auto& pq_codes = pq_codes_pair.first;
//     const size_t M = codebooks.size();
//     const size_t sub_dim = codebooks[0][0].size();

//     for (size_t rerank : rerank_values) {
//         double total_recall = 0.0;
//         long long total_latency = 0;

//         for (size_t i = 0; i < test_number; ++i) {
//             struct timeval start, end;
//             gettimeofday(&start, nullptr);

//             // 调用修改后的 pq_search_rerank，传入 cluster_products
//             auto result = pq_search_rerank(
//                 test_query + i * vecdim,
//                 pq_codes,
//                 codebooks,
//                 cluster_products,  // 传入预计算内积表
//                 base,
//                 M,
//                 sub_dim,
//                 k,
//                 rerank,
//                 vecdim
//             );

//             gettimeofday(&end, nullptr);
//             long long latency = (end.tv_sec * 1000000 + end.tv_usec) - (start.tv_sec * 1000000 + start.tv_usec);
//             total_latency += latency;

//             // 计算召回率（第二种代码保留）
//             std::set<uint32_t> gtset;
//             for (size_t j = 0; j < k; ++j) {
//                 gtset.insert(static_cast<uint32_t>(test_gt[i * test_gt_d + j]));
//             }
//             size_t correct = 0;
//             while (!result.empty()) {
//                 if (gtset.count(result.top().second)) correct++;
//                 result.pop();
//             }
//             total_recall += static_cast<float>(correct) / k;
//         }

//         double average_recall = total_recall / test_number;
//         long long average_latency = total_latency / test_number;

//         std::cout << "Rerank: " << rerank << "\tAverage Recall: " << average_recall << "\tAverage Latency: " << average_latency << " us\n";
//     }

//     delete[] test_query;
//     delete[] test_gt;
//     delete[] base;
//     return 0;
// }

// //子向量内积向量化
// //实现了并行查找基向量
#include <iostream>
#include <fstream>
#include <vector>
#include <queue>
#include <algorithm>
#include <cmath>
#include <cassert>
#include <sys/time.h>
#include <set>
#include <arm_neon.h>

std::vector<std::vector<std::vector<float>>> read_codebooks(const std::string& filename) {
    std::ifstream file(filename, std::ios::binary);
    int M, sub_dim;
    file.read(reinterpret_cast<char*>(&M), sizeof(int));
    file.read(reinterpret_cast<char*>(&sub_dim), sizeof(int));

    std::vector<std::vector<std::vector<float>>> codebooks(M, std::vector<std::vector<float>>(256, std::vector<float>(sub_dim)));
    for (int m = 0; m < M; ++m) {
        for (int k = 0; k < 256; ++k) {
            for (int d = 0; d < sub_dim; ++d) {
                file.read(reinterpret_cast<char*>(&codebooks[m][k][d]), sizeof(float));
            }
        }
    }
    return codebooks;
}

std::pair<std::vector<std::vector<uint8_t>>, std::pair<int, int>> load_pq_codes(const std::string& filename) {
    std::ifstream file(filename, std::ios::binary);
    int n, M;
    file.read(reinterpret_cast<char*>(&n), sizeof(int));
    file.read(reinterpret_cast<char*>(&M), sizeof(int));

    std::vector<std::vector<uint8_t>> pq_codes(n, std::vector<uint8_t>(M));
    for (int i = 0; i < n; ++i) {
        for (int m = 0; m < M; ++m) {
            file.read(reinterpret_cast<char*>(&pq_codes[i][m]), sizeof(uint8_t));
        }
    }
    return std::make_pair(pq_codes, std::make_pair(n, M));
}

std::vector<std::vector<std::vector<float>>> read_cluster_products(const std::string& filename) {
    std::ifstream file(filename, std::ios::binary);
    int M;
    file.read(reinterpret_cast<char*>(&M), sizeof(int));

    std::vector<std::vector<std::vector<float>>> products(M, std::vector<std::vector<float>>(256, std::vector<float>(256, 0.0f)));
    for (int m = 0; m < M; ++m) {
        for (int i = 0; i < 256; ++i) {
            for (int j = i; j < 256; ++j) {
                float val;
                file.read(reinterpret_cast<char*>(&val), sizeof(float));
                products[m][i][j] = val;
                products[m][j][i] = val;
            }
        }
    }
    return products;
}

std::priority_queue<std::pair<float, uint32_t>> flat_search(float* base, float* query, size_t base_number, size_t vecdim, size_t k) {
    std::priority_queue<std::pair<float, uint32_t>> q;
    const size_t vecdim_aligned = vecdim & ~3;
    for (size_t i = 0; i < base_number; ++i) {
        float32x4_t sum_vec = vdupq_n_f32(0.0f);
        for (size_t d = 0; d < vecdim_aligned; d += 4) {
            float32x4_t base_vec = vld1q_f32(base + i * vecdim + d);
            float32x4_t query_vec = vld1q_f32(query + d);
            sum_vec = vmlaq_f32(sum_vec, base_vec, query_vec);
        }
        float sum = vaddvq_f32(sum_vec);
        for (size_t d = vecdim_aligned; d < vecdim; ++d) {
            sum += base[i * vecdim + d] * query[d];
        }
        float dis = 1 - sum;
        if (q.size() < k) {
            q.push({dis, static_cast<uint32_t>(i)});
        } else if (dis < q.top().first) {
            q.push({dis, static_cast<uint32_t>(i)});
            q.pop();
        }
    }
    return q;
}

std::vector<std::vector<uint8_t>> batch_quantize_query(float* queries, size_t query_num, size_t M, size_t sub_dim, const std::vector<std::vector<std::vector<float>>>& codebooks) {
    std::vector<std::vector<uint8_t>> q_codes(query_num, std::vector<uint8_t>(M));
    for (size_t q = 0; q < query_num; ++q) {
        for (size_t m = 0; m < M; ++m) {
            float* sub_query = queries + q * M * sub_dim + m * sub_dim;
            float max_inner = -1e9;
            uint8_t best_code = 0;
            const size_t sub_dim_aligned = sub_dim & ~3;
            
            for (int k_idx = 0; k_idx < 256; ++k_idx) {
                float32x4_t sum_vec = vdupq_n_f32(0.0f);
                for (size_t d = 0; d < sub_dim_aligned; d += 4) {
                    float32x4_t codebook_vec = vld1q_f32(&codebooks[m][k_idx][d]);
                    float32x4_t sub_query_vec = vld1q_f32(sub_query + d);
                    sum_vec = vmlaq_f32(sum_vec, codebook_vec, sub_query_vec);
                }
                float sum = vaddvq_f32(sum_vec);
                for (size_t d = sub_dim_aligned; d < sub_dim; ++d) {
                    sum += codebooks[m][k_idx][d] * sub_query[d];
                }
                if (sum > max_inner) {
                    max_inner = sum;
                    best_code = k_idx;
                }
            }
            q_codes[q][m] = best_code;
        }
    }
    return q_codes;
}

std::vector<std::vector<float>> batch_pq_score(const std::vector<std::vector<uint8_t>>& q_codes, const std::vector<std::vector<uint8_t>>& pq_codes, const std::vector<std::vector<std::vector<float>>>& cluster_products, size_t query_num, size_t n, size_t M) {
    std::vector<std::vector<float>> scores(query_num, std::vector<float>(n, 0.0f));
    
    // 向量化参数：NEON向量宽度（4个float）
    const size_t vec_width = 4;
    const size_t vec_size = vec_width; // 每次处理4个基向量
    
    // 按查询向量化（外层循环展开）
    for (size_t q = 0; q < query_num; ++q) {
        // 预取查询编码，提升局部性
        const std::vector<uint8_t>& q_code = q_codes[q];
        
        // 按子空间并行
        #pragma omp parallel for
        for (size_t m = 0; m < M; ++m) {
            uint8_t q_idx = q_code[m];
            const std::vector<std::vector<float>>& prod_table = cluster_products[m];
            
            // 基向量向量化分组
            for (size_t i = 0; i < n; i += vec_size) {
                float32x4_t vec_scores = vdupq_n_f32(0.0f);
                for (size_t j = 0; j < vec_size && i + j < n; ++j) {
                    uint8_t db_idx = pq_codes[i + j][m];
                    // 直接访问乘积表，利用NEON向量加载
                    vec_scores = vaddq_f32(vec_scores, vld1q_f32(&prod_table[q_idx][db_idx]));
                }
                // 对齐存储，避免内存别名问题
                vst1q_f32(&scores[q][i], vec_scores);
            }
        }
    }
    return scores;
}

std::vector<std::priority_queue<std::pair<float, uint32_t>>> batch_rerank(float* queries, float* base_vectors, const std::vector<std::vector<float>>& scores, size_t query_num, size_t base_number, size_t vecdim, size_t k, size_t rerank) {
    std::vector<std::priority_queue<std::pair<float, uint32_t>>> final_results(query_num);
    for (size_t q = 0; q < query_num; ++q) {
        std::vector<size_t> indices(base_number);
        for (size_t i = 0; i < base_number; ++i) indices[i] = i;
        std::nth_element(indices.begin(), indices.begin() + rerank, indices.end(), [&scores, q](size_t a, size_t b) { return scores[q][a] > scores[q][b]; });

        std::priority_queue<std::pair<float, uint32_t>>& final_result = final_results[q];
        const size_t vecdim_aligned = vecdim & ~3;
        for (size_t i = 0; i < rerank; ++i) {
            size_t idx = indices[i];
            float32x4_t sum_vec = vdupq_n_f32(0.0f);
            for (size_t d = 0; d < vecdim_aligned; d += 4) {
                float32x4_t candidate_vec = vld1q_f32(base_vectors + idx * vecdim + d);
                float32x4_t query_vec = vld1q_f32(queries + q * vecdim + d);
                sum_vec = vmlaq_f32(sum_vec, candidate_vec, query_vec);
            }
            float sum = vaddvq_f32(sum_vec);
            for (size_t d = vecdim_aligned; d < vecdim; ++d) {
                sum += base_vectors[idx * vecdim + d] * (queries + q * vecdim)[d];
            }
            float dis = 1 - sum;
            if (final_result.size() < k) {
                final_result.push({dis, static_cast<uint32_t>(idx)});
            } else if (dis < final_result.top().first) {
                final_result.pop();
                final_result.push({dis, static_cast<uint32_t>(idx)});
            }
        }
    }
    return final_results;
}
/// 加载数据函数
template<typename T>
T* LoadData(const std::string& data_path, size_t& n, size_t& d) {
    std::ifstream fin(data_path, std::ios::in | std::ios::binary);
    fin.read(reinterpret_cast<char*>(&n), sizeof(int));
    fin.read(reinterpret_cast<char*>(&d), sizeof(int));
    T* data = new T[n * d];
    for (size_t i = 0; i < n; ++i) {
        fin.read(reinterpret_cast<char*>(data + i * d), d * sizeof(T));
    }
    fin.close();
    std::cerr << "load data " << data_path << "\n";
    std::cerr << "dimension: " << d << "  number:" << n << "  size_per_element:" << sizeof(T) << "\n";
    return data;
}

int main() {
    size_t test_number = 0, base_number = 0, test_gt_d = 0, vecdim = 0;
    std::string data_path = "/anndata/";

    float* test_query = LoadData<float>(data_path + "DEEP100K.query.fbin", test_number, vecdim);
    int* test_gt = LoadData<int>(data_path + "DEEP100K.gt.query.100k.top100.bin", test_number, test_gt_d);
    float* base = LoadData<float>(data_path + "DEEP100K.base.100k.fbin", base_number, vecdim);

    const size_t k = 10;
    //rerank表
    std::vector<size_t> rerank_values = {3200};

    auto codebooks = read_codebooks("files/codebooks.bin");
    auto pq_codes_pair = load_pq_codes("files/pq_codes.bin");
    auto cluster_products = read_cluster_products("files/cluster_products.bin");
    const auto& pq_codes = pq_codes_pair.first;
    const size_t M = codebooks.size();
    const size_t sub_dim = codebooks[0][0].size();

    for (size_t rerank : rerank_values) {
        double total_recall = 0.0;
        long long total_latency = 0;

        for (size_t i = 0; i < test_number; ++i) {
            struct timeval start, end;
            gettimeofday(&start, nullptr);

            float* query = test_query + i * vecdim;
            std::vector<uint8_t> q_code(M);
            for (size_t m = 0; m < M; ++m) {
                float* sub_query = query + m * sub_dim;
                float max_inner = -1e9;
                uint8_t best_code = 0;
                const size_t sub_dim_aligned = sub_dim & ~3;
                for (int k_idx = 0; k_idx < 256; ++k_idx) {
                    float32x4_t sum_vec = vdupq_n_f32(0.0f);
                    for (size_t d = 0; d < sub_dim_aligned; d += 4) {
                        float32x4_t codebook_vec = vld1q_f32(&codebooks[m][k_idx][d]);
                        float32x4_t sub_query_vec = vld1q_f32(sub_query + d);
                        sum_vec = vmlaq_f32(sum_vec, codebook_vec, sub_query_vec);
                    }
                    float sum = vaddvq_f32(sum_vec);
                    for (size_t d = sub_dim_aligned; d < sub_dim; ++d) {
                        sum += codebooks[m][k_idx][d] * sub_query[d];
                    }
                    if (sum > max_inner) {
                        max_inner = sum;
                        best_code = k_idx;
                    }
                }
                q_code[m] = best_code;
            }
            
            std::vector<float> pq_scores(base_number, 0.0);
            for (size_t i_db = 0; i_db < base_number; ++i_db) {
                float score = 0.0;
                for (size_t m = 0; m < M; ++m) {
                    uint8_t q_idx = q_code[m];
                    uint8_t db_idx = pq_codes[i_db][m];
                    score += cluster_products[m][q_idx][db_idx];
                }
                pq_scores[i_db] = score;
            }

            std::vector<size_t> indices(base_number);
            for (size_t i_db = 0; i_db < base_number; ++i_db) indices[i_db] = i_db;
            std::nth_element(indices.begin(), indices.begin() + rerank, indices.end(),
                             [&pq_scores](size_t a, size_t b) { return pq_scores[a] > pq_scores[b]; });

            std::priority_queue<std::pair<float, uint32_t>> final_result;
            const size_t vecdim_aligned = vecdim & ~3;
            for (size_t i_db = 0; i_db < rerank; ++i_db) {
                size_t idx = indices[i_db];
                float32x4_t sum_vec = vdupq_n_f32(0.0f);
                for (size_t d = 0; d < vecdim_aligned; d += 4) {
                    float32x4_t candidate_vec = vld1q_f32(base + idx * vecdim + d);
                    float32x4_t query_vec = vld1q_f32(query + d);
                    sum_vec = vmlaq_f32(sum_vec, candidate_vec, query_vec);
                }
                float sum = vaddvq_f32(sum_vec);
                for (size_t d = vecdim_aligned; d < vecdim; ++d) {
                    sum += base[idx * vecdim + d] * query[d];
                }
                float dis = 1 - sum;
                if (final_result.size() < k) {
                    final_result.push({dis, static_cast<uint32_t>(idx)});
                } else if (dis < final_result.top().first) {
                    final_result.pop();
                    final_result.push({dis, static_cast<uint32_t>(idx)});
                }
            }

            gettimeofday(&end, nullptr);
            long long latency = (end.tv_sec * 1000000 + end.tv_usec) - (start.tv_sec * 1000000 + start.tv_usec);
            total_latency += latency;

            std::set<uint32_t> gtset;
            for (size_t j = 0; j < k; ++j) {
                gtset.insert(static_cast<uint32_t>(test_gt[i * test_gt_d + j]));
            }
            size_t correct = 0;
            while (!final_result.empty()) {
                if (gtset.count(final_result.top().second)) correct++;
                final_result.pop();
            }
            total_recall += static_cast<float>(correct) / k;
        }

        double average_recall = total_recall / test_number;
        long long average_latency = total_latency / test_number;
        std::cout << "Rerank: " << rerank 
                  << "\tAverage Recall: " << average_recall 
                  << "\tAverage Latency: " << average_latency << " us\n";
    }

    delete[] test_query;
    delete[] test_gt;
    delete[] base;
    return 0;
} 
