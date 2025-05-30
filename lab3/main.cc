//代码中有许多版本，最终使用的是pthread版本和openMP2版本作为最终的作业
//ivf+hnsw
// #include <iostream>
// #include <fstream>
// #include <vector>
// #include <pthread.h>
// #include <sys/time.h>
// #include "hnswlib/hnswlib/hnswlib.h"
// #include <cfloat>
// #include <cstdlib>
// #include <unordered_set>
// #include <cstdint>
// #include <algorithm>
// #include <queue>

// using namespace std;

// // --------------------- 基础数据结构 ---------------------
// struct IVFIndex {
//     int32_t nlist;       // 聚类数（质心数量）
//     int32_t dim;         // 向量维度
//     vector<float> centroids;  // 质心数据 [nlist * dim]
//     vector<vector<uint32_t>> inverted_lists; // 倒排表（使用uint32_t匹配Python）
// };

// struct SearchResult {
//     size_t id;          // 向量ID
//     float distance;     // 距离
// };

// // --------------------- 数据加载 ---------------------
// vector<float> load_fbin(const string& path, int32_t& num_vectors, int32_t& dim) {
//     ifstream file(path, ios::binary);
//     if (!file) {
//         cerr << "Error: Cannot open file " << path << endl;
//         exit(1);
//     }
//     file.read(reinterpret_cast<char*>(&num_vectors), sizeof(int32_t));
//     file.read(reinterpret_cast<char*>(&dim), sizeof(int32_t));
//     if (dim <= 0 || dim > 10000) {
//         cerr << "Error: Invalid dimension " << dim << " in file " << path << endl;
//         exit(1);
//     }
//     vector<float> data(num_vectors * dim);
//     file.read(reinterpret_cast<char*>(data.data()), num_vectors * dim * sizeof(float));
//     if (file.gcount() != static_cast<streamsize>(num_vectors * dim * sizeof(float))) {
//         cerr << "Error: Incomplete read from file " << path << endl;
//         exit(1);
//     }
//     return data;
// }

// // --------------------- IVF索引加载 ---------------------
// IVFIndex load_ivf_index(const string& path) {
//     IVFIndex index;
//     ifstream file(path, ios::binary);
//     if (!file) {
//         cerr << "Error: Cannot open file " << path << endl;
//         exit(1);
//     }
//     file.read(reinterpret_cast<char*>(&index.nlist), sizeof(int32_t));
//     file.read(reinterpret_cast<char*>(&index.dim), sizeof(int32_t));
//     index.centroids.resize(index.nlist * index.dim);
//     file.read(reinterpret_cast<char*>(index.centroids.data()), index.nlist * index.dim * sizeof(float));
    
//     for (int32_t i = 0; i < index.nlist; i++) {
//         int32_t size;
//         file.read(reinterpret_cast<char*>(&size), sizeof(int32_t));
//         vector<uint32_t> vec_ids(size); // 使用uint32_t匹配Python
//         file.read(reinterpret_cast<char*>(vec_ids.data()), size * sizeof(uint32_t));
//         index.inverted_lists.push_back(vec_ids);
//     }
//     return index;
// }

// // --------------------- 解析HNSW索引文件头获取维度 ---------------------
// size_t parse_hnsw_dimension(const string& path) {
//     ifstream file(path, ios::binary | ios::ate);
//     if (!file) {
//         cerr << "Error: Cannot open HNSW index file " << path << endl;
//         exit(1);
//     }
    
//     // HNSW文件头格式: 4字节magic number + 4字节维度 + ...
//     if (file.tellg() < 8) {
//         cerr << "Error: Invalid HNSW index file" << endl;
//         exit(1);
//     }
    
//     file.seekg(4, ios::beg); // 跳过magic number
//     uint32_t dim;
//     file.read(reinterpret_cast<char*>(&dim), sizeof(uint32_t));
//     return static_cast<size_t>(dim);
// }

// // --------------------- HNSW索引加载 ---------------------
// struct HNSWIndexWrapper {
//     hnswlib::HierarchicalNSW<float>* index;
//     size_t dim;
    
//     HNSWIndexWrapper(hnswlib::HierarchicalNSW<float>* idx, size_t d) 
//         : index(idx), dim(d) {}
    
//     ~HNSWIndexWrapper() {
//         delete index;
//     }
// };

// HNSWIndexWrapper* load_hnsw_index(const string& path, int expected_dim) {
//     // 从文件头解析实际维度
//     size_t actual_dim = parse_hnsw_dimension(path);
    
//     if (actual_dim != static_cast<size_t>(expected_dim)) {
//         cerr << "HNSW维度不匹配: 配置维度" << expected_dim 
//              << " 实际维度" << actual_dim << endl;
//         exit(EXIT_FAILURE);
//     }
    
//     hnswlib::L2Space space(expected_dim);
//     try {
//         hnswlib::HierarchicalNSW<float>* index = new hnswlib::HierarchicalNSW<float>(&space, path, false);
//         return new HNSWIndexWrapper(index, actual_dim);
//     } catch (const exception& e) {
//         cerr << "Error loading HNSW index from " << path << ": " << e.what() << endl;
//         exit(1);
//     }
// }

// // --------------------- IVF搜索任务 ---------------------
// struct IVFSearchTask {
//     const IVFIndex* ivf;
//     const float* query;
//     size_t nprobe;
//     size_t thread_id;
//     size_t num_threads;
//     vector<int32_t>* candidate_clusters;
//     pthread_mutex_t* mutex;
// };

// void* ivf_search_thread(void* arg) {
//     IVFSearchTask* task = (IVFSearchTask*)arg;
//     const IVFIndex& index = *task->ivf;
//     size_t per_thread = index.nlist / task->num_threads;
//     size_t start = task->thread_id * per_thread;
//     size_t end = (task->thread_id + 1) * per_thread;
//     if (task->thread_id == task->num_threads - 1) end = index.nlist;

//     vector<pair<float, int32_t>> local_distances;
//     for (int32_t cid = start; cid < end; cid++) {
//         const float* centroid = &index.centroids[cid * index.dim];
//         float dist = 0;
//         for (int32_t d = 0; d < index.dim; d++) {
//             dist += (task->query[d] - centroid[d]) * (task->query[d] - centroid[d]);
//         }
//         local_distances.emplace_back(dist, cid);
//     }

//     sort(local_distances.begin(), local_distances.end());
//     pthread_mutex_lock(task->mutex);
//     for (size_t i = 0; i < min(task->nprobe, local_distances.size()); i++) {
//         task->candidate_clusters->push_back(local_distances[i].second);
//     }
//     pthread_mutex_unlock(task->mutex);
//     return nullptr;
// }

// // --------------------- 计算向量距离 ---------------------
// float compute_distance(const float* a, const float* b, size_t dim) {
//     float dist = 0;
//     for (size_t i = 0; i < dim; i++) {
//         float diff = a[i] - b[i];
//         dist += diff * diff;
//     }
//     return dist;
// }

// // --------------------- 混合搜索（兼容所有HNSW版本） ---------------------
// vector<SearchResult> hybrid_search(
//     const IVFIndex& ivf_index,
//     HNSWIndexWrapper* hnsw_wrapper,
//     const float* query,
//     size_t k,
//     size_t nprobe,
//     size_t num_threads) {
    
//     size_t dim = ivf_index.dim;
    
//     if (hnsw_wrapper->dim != dim) {
//         cerr << "维度不匹配: IVF(" << dim 
//              << ") vs HNSW(" << hnsw_wrapper->dim << ")" << endl;
//         return {};
//     }

//     // 1. IVF部分：找出候选聚类
//     vector<int32_t> candidate_clusters;
//     pthread_mutex_t mutex;
//     pthread_mutex_init(&mutex, nullptr);

//     pthread_t* threads = new pthread_t[num_threads];
//     IVFSearchTask* tasks = new IVFSearchTask[num_threads];

//     for (size_t t = 0; t < num_threads; t++) {
//         tasks[t] = {&ivf_index, query, nprobe, t, num_threads, &candidate_clusters, &mutex};
//         pthread_create(&threads[t], nullptr, ivf_search_thread, &tasks[t]);
//     }

//     for (size_t t = 0; t < num_threads; t++) {
//         pthread_join(threads[t], nullptr);
//     }
//     pthread_mutex_destroy(&mutex);
//     delete[] threads;
//     delete[] tasks;

//     // 2. 收集候选向量ID
//     unordered_set<int32_t> unique_clusters(candidate_clusters.begin(), candidate_clusters.end());
//     vector<size_t> candidate_vectors;
//     for (int32_t cid : unique_clusters) {
//         // 添加聚类ID范围检查
//         if (cid < 0 || cid >= ivf_index.nlist) {
//             cerr << "无效聚类ID: " << cid << endl;
//             continue;
//         }
        
//         for (uint32_t vid : ivf_index.inverted_lists[cid]) {
//             // 添加向量ID范围检查（转换为size_t后比较）
//             if (static_cast<size_t>(vid) >= hnsw_wrapper->index->max_elements_) {
//                 cerr << "无效向量ID: " << vid << " (超出HNSW索引范围)" << endl;
//                 continue;
//             }
//             candidate_vectors.push_back(static_cast<size_t>(vid));
//         }
//     }
    
//     cout << "找到 " << candidate_vectors.size() << " 个候选向量" << endl;

//     // 3. 正确获取向量数据（兼容返回vector<float>的HNSW版本）
//     priority_queue<pair<float, size_t>> max_heap;
    
//     for (size_t vid : candidate_vectors) {
//         // 添加ID有效性检查
//         if (vid >= hnsw_wrapper->index->max_elements_) {
//             cerr << "无效ID: " << vid << endl;
//             continue;
//         }

//         // 获取向量数据
//         vector<float> candidate_vec = hnsw_wrapper->index->getDataByLabel<float>(vid);
        
//         // 检查向量维度是否正确
//         if (candidate_vec.size() != dim) {
//             cerr << "警告: 向量ID " << vid << " 维度错误（预期" << dim 
//                  << "，实际" << candidate_vec.size() << "）" << endl;
//             continue;
//         }

//         // 计算距离（使用vector.data()获取原始指针）
//         float distance = compute_distance(query, candidate_vec.data(), dim);
        
//         // 维护最大堆以保留top-k
//         if (max_heap.size() < k) {
//             max_heap.push({distance, vid});
//         } else if (distance < max_heap.top().first) {
//             max_heap.pop();
//             max_heap.push({distance, vid});
//         }
//     }

//     // 4. 提取结果并排序
//     vector<SearchResult> results;
//     while (!max_heap.empty()) {
//         pair<float, size_t> top = max_heap.top();
//         results.push_back({top.second, top.first});
//         max_heap.pop();
//     }
//     // 从小到大排序
//     sort(results.begin(), results.end(), [](const SearchResult& a, const SearchResult& b) {
//         return a.distance < b.distance;
//     });

//     return results;
// }

// // --------------------- 主函数 ---------------------
// int main() {
//     // 修改为相对路径或命令行参数
//     const string base_path = "/anndata/DEEP100K.base.100k.fbin";
//     const string query_path = "/anndata/DEEP100K.query.fbin";
//     const string ivf_path = "files/ivf.index";
//     const string hnsw_path = "files/hnsw.index";
    
//     int32_t base_num, base_dim;
//     vector<float> base_data = load_fbin(base_path, base_num, base_dim);
    
//     int32_t query_num, query_dim;
//     vector<float> query_data = load_fbin(query_path, query_num, query_dim);
    
//     if (base_dim != query_dim) {
//         cerr << "Error: Base dimension (" << base_dim 
//              << ") != Query dimension (" << query_dim << ")" << endl;
//         return 1;
//     }
    
//     IVFIndex ivf_index = load_ivf_index(ivf_path);
//     HNSWIndexWrapper* hnsw_wrapper = load_hnsw_index(hnsw_path, base_dim);
    
//     // 验证所有索引维度一致
//     if (ivf_index.dim != base_dim) {
//         cerr << "Error: IVF dimension (" << ivf_index.dim 
//              << ") != Base dimension (" << base_dim << ")" << endl;
//         return 1;
//     }
    
//     cout << "Index loaded successfully. Dimensions: " << base_dim << endl;
//     cout << "Base vectors: " << base_num << endl;
//     cout << "Query vectors: " << query_num << endl;
    
//     // 搜索参数
//     size_t k = 10;      // 返回结果数
//     size_t nprobe = 32; // 探查的聚类数
//     size_t num_threads = 8; // 线程数
    
//     // 测试前10条查询
//     for (size_t i = 0; i < min(static_cast<size_t>(10), static_cast<size_t>(query_num)); i++) {
//         const float* query = &query_data[i * query_dim];
//         struct timeval start, end;
//         gettimeofday(&start, nullptr);
        
//         vector<SearchResult> results = hybrid_search(
//             ivf_index, hnsw_wrapper, query, k, nprobe, num_threads);
        
//         gettimeofday(&end, nullptr);
//         double elapsed = (end.tv_sec - start.tv_sec) * 1000.0 + (end.tv_usec - start.tv_usec) / 1000.0;
        
//         cout << "Query " << i << " Time: " << elapsed << " ms" << endl;
//         for (size_t j = 0; j < results.size(); j++) {
//             cout << "  Top-" << (j+1) << ": ID=" << results[j].id 
//                  << ", Distance=" << results[j].distance << endl;
//         }
//     }
    
//     delete hnsw_wrapper; // 释放包装器，会自动释放内部索引
//     return 0;
// }

//ivf+pq
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

// using namespace std;

// // 前置声明数据加载函数（修复未声明错误）
// template<typename T>
// T* LoadData(const std::string& data_path, size_t& n, size_t& d);

// // ===================== IVF索引结构 =====================
// struct IVFIndex {
//     int nlist, dim, nprobe;
//     vector<vector<float>> centroids;
//     vector<vector<uint32_t>> inverted_lists;

//     void load() {
//         string filename = "./files/ivf.index";
//         ifstream file(filename, ios::binary);
//         if (!file.is_open()) {
//             cerr << "Error loading IVF index from " << filename << endl;
//             exit(1);
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
//             file.read(reinterpret_cast<char*>(inverted_lists[i].data()), size * sizeof(uint32_t));
//         }
//         file.close();
//         nprobe = 32;
//     }

//     vector<int> search_clusters(const float* query) {
//         vector<pair<float, int>> distances;
//         distances.reserve(nlist);
        
//         for (int cid = 0; cid < nlist; ++cid) {
//             float dist_sq = 0.0f;
//             for (int d = 0; d < dim; ++d) {
//                 float diff = centroids[cid][d] - query[d];
//                 dist_sq += diff * diff;
//             }
//             float dist = sqrtf(dist_sq);
//             distances.push_back(make_pair(dist, cid));
//         }

//         nth_element(distances.begin(), distances.begin() + nprobe, distances.end());
//         sort(distances.begin(), distances.begin() + nprobe);
        
//         vector<int> clusters(nprobe);
//         for (int i = 0; i < nprobe; ++i) {
//             clusters[i] = distances[i].second;
//         }
//         return clusters;
//     }
// };

// // ===================== PQ代码 =====================
// vector<vector<vector<float>>> read_codebooks(const string& filename) {
//     ifstream file(filename, ios::binary);
//     int M, sub_dim;
//     file.read(reinterpret_cast<char*>(&M), sizeof(int));
//     file.read(reinterpret_cast<char*>(&sub_dim), sizeof(int));

//     vector<vector<vector<float>>> codebooks(M, vector<vector<float>>(256, vector<float>(sub_dim)));
//     for (int m = 0; m < M; ++m) {
//         for (int k = 0; k < 256; ++k) {
//             for (int d = 0; d < sub_dim; ++d) {
//                 file.read(reinterpret_cast<char*>(&codebooks[m][k][d]), sizeof(float));
//             }
//         }
//     }
//     return codebooks;
// }

// pair<vector<vector<uint8_t>>, pair<int, int>> load_pq_codes(const string& filename) {
//     ifstream file(filename, ios::binary);
//     int n, M;
//     file.read(reinterpret_cast<char*>(&n), sizeof(int));
//     file.read(reinterpret_cast<char*>(&M), sizeof(int));

//     vector<vector<uint8_t>> pq_codes(n, vector<uint8_t>(M));
//     for (int i = 0; i < n; ++i) {
//         for (int m = 0; m < M; ++m) {
//             file.read(reinterpret_cast<char*>(&pq_codes[i][m]), sizeof(uint8_t));
//         }
//     }
//     return make_pair(pq_codes, make_pair(n, M));
// }

// vector<vector<vector<float>>> read_cluster_products(const string& filename) {
//     ifstream file(filename, ios::binary);
//     int M;
//     file.read(reinterpret_cast<char*>(&M), sizeof(int));

//     vector<vector<vector<float>>> products(M, vector<vector<float>>(256, vector<float>(256, 0.0f)));
//     for (int m = 0; m < M; ++m) {
//         for (int i = 0; i < 256; ++i) {
//             for (int j = i; j < 256; ++j) {
//                 float val;
//                 file.read(reinterpret_cast<char*>(&val), sizeof(float));
//                 products[m][i][j] = val;
//                 products[m][j][i] = val;
//             }
//         }
//     }
//     return products;
// }

// // ===================== 主函数 =====================
// int main() {
//     size_t test_number = 0, base_number = 0, test_gt_d = 0, vecdim = 0;
//     string data_path = "/anndata/";

//     float* test_query = LoadData<float>(data_path + "DEEP100K.query.fbin", test_number, vecdim);
//     int* test_gt = LoadData<int>(data_path + "DEEP100K.gt.query.100k.top100.bin", test_number, test_gt_d);
//     float* base = LoadData<float>(data_path + "DEEP100K.base.100k.fbin", base_number, vecdim);

//     const size_t k = 10;
//     vector<size_t> rerank_values = {20};

//     auto codebooks = read_codebooks("files/codebooks.bin");
//     auto pq_codes_pair = load_pq_codes("files/pq_codes.bin");
//     auto cluster_products = read_cluster_products("files/cluster_products.bin");
//     const auto& pq_codes = pq_codes_pair.first;
//     const size_t M = codebooks.size();
//     const size_t sub_dim = codebooks[0][0].size();

//     IVFIndex ivf;
//     ivf.load();

//     // 使用传统assert（修复多参数错误）
//     if (vecdim != ivf.dim) {
//         cerr << "Vector dimension mismatch between IVF and data" << endl;
//         return 1;
//     }

//     for (size_t rerank : rerank_values) {
//         double total_recall = 0.0;
//         long long total_latency = 0;

//         for (size_t i = 0; i < test_number; ++i) {
//             struct timeval start, end;
//             gettimeofday(&start, nullptr);

//             float* query = test_query + i * vecdim;

//             vector<int> clusters = ivf.search_clusters(query);
            
//             vector<uint32_t> candidates;
//             for (int cid : clusters) {
//                 candidates.insert(candidates.end(), 
//                                 ivf.inverted_lists[cid].begin(), 
//                                 ivf.inverted_lists[cid].end());
//             }
//             if (candidates.empty()) continue;

//             vector<uint8_t> q_code(M);
//             for (size_t m = 0; m < M; ++m) {
//                 float* sub_query = query + m * sub_dim;
//                 float max_inner = -1e9;
//                 uint8_t best_code = 0;
//                 const size_t sub_dim_aligned = sub_dim & ~3;
                
//                 for (int k_idx = 0; k_idx < 256; ++k_idx) {
//                     float32x4_t sum_vec = vdupq_n_f32(0.0f);
//                     for (size_t d = 0; d < sub_dim_aligned; d += 4) {
//                         float32x4_t codebook_vec = vld1q_f32(&codebooks[m][k_idx][d]);
//                         float32x4_t sub_query_vec = vld1q_f32(sub_query + d);
//                         sum_vec = vmlaq_f32(sum_vec, codebook_vec, sub_query_vec);
//                     }
//                     float sum = vaddvq_f32(sum_vec);
//                     for (size_t d = sub_dim_aligned; d < sub_dim; ++d) {
//                         sum += codebooks[m][k_idx][d] * sub_query[d];
//                     }
//                     if (sum > max_inner) {
//                         max_inner = sum;
//                         best_code = k_idx;
//                     }
//                 }
//                 q_code[m] = best_code;
//             }

//             vector<float> pq_scores(candidates.size(), 0.0f);
//             for (size_t i_db = 0; i_db < candidates.size(); ++i_db) {
//                 uint32_t vec_id = candidates[i_db];
//                 float score = 0.0f;
//                 for (size_t m = 0; m < M; ++m) {
//                     uint8_t q_idx = q_code[m];
//                     uint8_t db_idx = pq_codes[vec_id][m];
//                     score += cluster_products[m][q_idx][db_idx];
//                 }
//                 pq_scores[i_db] = score;
//             }

//             vector<size_t> indices(candidates.size());
//             // 使用传统循环替代iota（修复C++11兼容性）
//             for (size_t i = 0; i < candidates.size(); ++i) {
//                 indices[i] = i;
//             }
            
//             nth_element(indices.begin(), indices.begin() + rerank, indices.end(),
//                        [&pq_scores](size_t a, size_t b) { return pq_scores[a] > pq_scores[b]; });

//             priority_queue<pair<float, uint32_t>> final_result;
//             const size_t vecdim_aligned = vecdim & ~3;
//             for (size_t idx : indices) {
//                 if (idx >= candidates.size()) continue;
//                 uint32_t vec_id = candidates[idx];
//                 float32x4_t sum_vec = vdupq_n_f32(0.0f);
                
//                 for (size_t d = 0; d < vecdim_aligned; d += 4) {
//                     float32x4_t candidate_vec = vld1q_f32(base + vec_id * vecdim + d);
//                     float32x4_t query_vec = vld1q_f32(query + d);
//                     sum_vec = vmlaq_f32(sum_vec, candidate_vec - query_vec, candidate_vec - query_vec);
//                 }
//                 float dist = vaddvq_f32(sum_vec);
//                 for (size_t d = vecdim_aligned; d < vecdim; ++d) {
//                     float diff = base[vec_id * vecdim + d] - query[d];
//                     dist += diff * diff;
//                 }
//                 dist = sqrtf(dist);

//                 if (final_result.size() < k || dist < final_result.top().first) {
//                     if (final_result.size() == k) final_result.pop();
//                     final_result.push(make_pair(dist, vec_id));
//                 }
//             }

//             gettimeofday(&end, nullptr);
//             long long latency = (end.tv_sec * 1000000 + end.tv_usec) - (start.tv_sec * 1000000 + start.tv_usec);
//             total_latency += latency;

//             set<uint32_t> gtset;
//             for (size_t j = 0; j < k; ++j) {
//                 gtset.insert(static_cast<uint32_t>(test_gt[i * test_gt_d + j]));
//             }
//             size_t correct = 0;
//             while (!final_result.empty()) {
//                 if (gtset.count(final_result.top().second)) correct++;
//                 final_result.pop();
//             }
//             total_recall += static_cast<float>(correct) / k;
//         }

//         double average_recall = total_recall / test_number;
//         long long average_latency = total_latency / test_number;
//         cout << "Rerank: " << rerank 
//              << "\tIVF nprobe: " << ivf.nprobe
//              << "\tRecall@" << k << ": " << average_recall 
//              << "\tLatency: " << average_latency << " us\n";
//     }

//     delete[] test_query;
//     delete[] test_gt;
//     delete[] base;
//     return 0;
// }

// // 数据加载函数
// template<typename T>
// T* LoadData(const std::string& data_path, size_t& n, size_t& d) {
//     ifstream fin(data_path, ios::in | ios::binary);
//     fin.read(reinterpret_cast<char*>(&n), sizeof(int));
//     fin.read(reinterpret_cast<char*>(&d), sizeof(int));
//     T* data = new T[n * d];
//     for (size_t i = 0; i < n; ++i) {
//         fin.read(reinterpret_cast<char*>(data + i * d), d * sizeof(T));
//     }
//     fin.close();
//     cerr << "load data " << data_path << "\n";
//     cerr << "dimension: " << d << "  number:" << n << "  size_per_element:" << sizeof(T) << "\n";
//     return data;
// }


//pthread
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
// #include <pthread.h>
// #include <mutex>
// #include <thread>
// #include <atomic>
// #include <condition_variable>
// #include <functional>
// #include <future>

// using namespace std;

// // 线程安全累加器
// struct ThreadSafeAccumulator {
//     double total_recall = 0.0;
//     long long total_latency = 0;
//     mutex mtx;

//     void add_recall(double recall) {
//         lock_guard<mutex> lock(mtx);
//         total_recall += recall;
//     }

//     void add_latency(long long latency) {
//         lock_guard<mutex> lock(mtx);
//         total_latency += latency;
//     }
// };

// // 线程池实现
// class ThreadPool {
// public:
//     ThreadPool(size_t num_threads) : stop(false) {
//         for (size_t i = 0; i < num_threads; ++i) {
//             workers.emplace_back([this] {
//                 while (true) {
//                     function<void()> task;
//                     {
//                         unique_lock<mutex> lock(this->queue_mutex);
//                         this->condition.wait(lock, [this] { return this->stop || !this->tasks.empty(); });
//                         if (this->stop && this->tasks.empty())
//                             return;
//                         task = move(this->tasks.front());
//                         this->tasks.pop();
//                     }
//                     task();
//                 }
//             });
//         }
//     }

//     ~ThreadPool() {
//         {
//             unique_lock<mutex> lock(queue_mutex);
//             stop = true;
//         }
//         condition.notify_all();
//         for (thread &worker : workers) {
//             worker.join();
//         }
//     }

//     template<class F, class... Args>
//     auto enqueue(F&& f, Args&&... args) 
//         -> future<typename result_of<F(Args...)>::type> {
//         using return_type = typename result_of<F(Args...)>::type;

//         auto task = make_shared<packaged_task<return_type()>>(
//             bind(forward<F>(f), forward<Args>(args)...)
//         );
        
//         future<return_type> res = task->get_future();
//         {
//             unique_lock<mutex> lock(queue_mutex);
//             if (stop)
//                 throw runtime_error("enqueue on stopped ThreadPool");
//             tasks.emplace([task]() { (*task)(); });
//         }
//         condition.notify_one();
//         return res;
//     }

// private:
//     vector<thread> workers;
//     queue<function<void()>> tasks;
//     mutex queue_mutex;
//     condition_variable condition;
//     atomic<bool> stop;
// };

// // IVF索引结构
// struct IVFIndex {
//     int nlist, dim, nprobe;
//     vector<vector<float>> centroids;
//     vector<vector<uint32_t>> inverted_lists;
//     mutex cluster_mutex; // 候选收集互斥锁
//     ThreadPool* thread_pool; // 线程池指针

//     IVFIndex(int num_threads) : thread_pool(new ThreadPool(num_threads)) {}

//     ~IVFIndex() {
//         delete thread_pool;
//     }

//     // 质心距离计算任务（按nprobe分组）
//     struct CentroidDistanceTask {
//         const float* query;
//         const vector<vector<float>>* centroids;
//         vector<pair<float, int>>* distances;
//         int start_cid, end_cid;

//         void operator()() {
//             for (int cid = start_cid; cid < end_cid; ++cid) {
//                 float dist_sq = 0.0f;
//                 for (int d = 0; d < centroids->at(cid).size(); ++d) {
//                     float diff = centroids->at(cid)[d] - query[d];
//                     dist_sq += diff * diff;
//                 }
//                 float dist = sqrtf(dist_sq);
//                 (*distances)[cid] = make_pair(dist, cid);
//             }
//         }
//     };

//     // 并行计算质心距离（按nprobe分组）
//     void parallel_compute_distances(const float* query, vector<pair<float, int>>& distances, int num_threads) {
//         if (num_threads <= 1 || nlist < 100) {
//             distances.resize(nlist);
//             for (int cid = 0; cid < nlist; ++cid) {
//                 float dist_sq = 0.0f;
//                 for (int d = 0; d < dim; ++d) {
//                     float diff = centroids[cid][d] - query[d];
//                     dist_sq += diff * diff;
//                 }
//                 distances[cid] = {sqrtf(dist_sq), cid};
//             }
//             return;
//         }

//         distances.resize(nlist);
//         vector<future<void>> futures;
//         int per_thread = nlist / num_threads;
//         int remainder = nlist % num_threads;

//         // 按线程数划分任务
//         for (int t = 0; t < num_threads; ++t) {
//             int start = t * per_thread + min(t, remainder);
//             int end = start + per_thread + (t < remainder ? 1 : 0);
            
//             futures.emplace_back(
//                 thread_pool->enqueue(
//                     CentroidDistanceTask{query, &centroids, &distances, start, end}
//                 )
//             );
//         }

//         // 等待所有任务完成
//         for (auto& future : futures) {
//             future.wait();
//         }
//     }

//     // 候选收集任务（按nprobe分组）
//     struct CandidateGatherTask {
//         const vector<int>* clusters; // 选中的nprobe个簇
//         const vector<vector<uint32_t>>* inverted_lists;
//         vector<uint32_t>* candidates;
//         mutex* mtx;
//         int start_idx, end_idx; // 处理clusters的索引范围

//         void operator()() {
//             vector<uint32_t> local_candidates;
//             for (int i = start_idx; i < end_idx; ++i) {
//                 int cid = (*clusters)[i];
//                 local_candidates.insert(local_candidates.end(), 
//                                     inverted_lists->at(cid).begin(), 
//                                     inverted_lists->at(cid).end());
//             }
//             lock_guard<mutex> lock(*mtx);
//             candidates->insert(candidates->end(), local_candidates.begin(), local_candidates.end());
//         }
//     };

//     // 并行收集候选向量（按nprobe分组）
//     void parallel_gather_candidates(const vector<int>& clusters, vector<uint32_t>& candidates, int num_threads) {
//         if (num_threads <= 1 || clusters.size() < 10) {
//             candidates.clear();
//             for (int cid : clusters) {
//                 candidates.insert(candidates.end(), inverted_lists[cid].begin(), inverted_lists[cid].end());
//             }
//             return;
//         }

//         candidates.clear();
//         vector<future<void>> futures;
//         int per_thread = clusters.size() / num_threads;
//         int remainder = clusters.size() % num_threads;

//         // 按线程数划分任务
//         for (int t = 0; t < num_threads; ++t) {
//             int start = t * per_thread + min(t, remainder);
//             int end = start + per_thread + (t < remainder ? 1 : 0);
            
//             futures.emplace_back(
//                 thread_pool->enqueue(
//                     CandidateGatherTask{&clusters, &inverted_lists, &candidates, &cluster_mutex, start, end}
//                 )
//             );
//         }

//         // 等待所有任务完成
//         for (auto& future : futures) {
//             future.wait();
//         }
//     }

//     void load() {
//         string filename = "./files/ivf.index";
//         ifstream file(filename, ios::binary);
//         if (!file.is_open()) {
//             cerr << "Error loading IVF index from " << filename << endl;
//             exit(1);
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
//             file.read(reinterpret_cast<char*>(inverted_lists[i].data()), size * sizeof(uint32_t));
//         }
//         file.close();
//         nprobe = 16;
//     }

//     vector<int> search_clusters(const float* query, int num_threads) {
//         vector<pair<float, int>> distances;
//         parallel_compute_distances(query, distances, num_threads);
        
//         nth_element(distances.begin(), distances.begin() + nprobe, distances.end());
//         sort(distances.begin(), distances.begin() + nprobe);
        
//         vector<int> clusters(nprobe);
//         for (int i = 0; i < nprobe; ++i) clusters[i] = distances[i].second;
//         return clusters;
//     }
// };

// // PQ相关函数
// vector<vector<vector<float>>> read_codebooks(const string& filename) {
//     ifstream file(filename, ios::binary);
//     int M, sub_dim;
//     file.read(reinterpret_cast<char*>(&M), sizeof(int));
//     file.read(reinterpret_cast<char*>(&sub_dim), sizeof(int));

//     vector<vector<vector<float>>> codebooks(M, vector<vector<float>>(256, vector<float>(sub_dim)));
//     for (int m = 0; m < M; ++m) {
//         for (int k = 0; k < 256; ++k) {
//             for (int d = 0; d < sub_dim; ++d) {
//                 file.read(reinterpret_cast<char*>(&codebooks[m][k][d]), sizeof(float));
//             }
//         }
//     }
//     return codebooks;
// }

// pair<vector<vector<uint8_t>>, pair<int, int>> load_pq_codes(const string& filename) {
//     ifstream file(filename, ios::binary);
//     int n, M;
//     file.read(reinterpret_cast<char*>(&n), sizeof(int));
//     file.read(reinterpret_cast<char*>(&M), sizeof(int));

//     vector<vector<uint8_t>> pq_codes(n, vector<uint8_t>(M));
//     for (int i = 0; i < n; ++i) {
//         for (int m = 0; m < M; ++m) {
//             file.read(reinterpret_cast<char*>(&pq_codes[i][m]), sizeof(uint8_t));
//         }
//     }
//     return make_pair(pq_codes, make_pair(n, M));
// }

// vector<vector<vector<float>>> read_cluster_products(const string& filename) {
//     ifstream file(filename, ios::binary);
//     int M;
//     file.read(reinterpret_cast<char*>(&M), sizeof(int));

//     vector<vector<vector<float>>> products(M, vector<vector<float>>(256, vector<float>(256, 0.0f)));
//     for (int m = 0; m < M; ++m) {
//         for (int i = 0; i < 256; ++i) {
//             for (int j = i; j < 256; ++j) {
//                 float val;
//                 file.read(reinterpret_cast<char*>(&val), sizeof(float));
//                 products[m][i][j] = val;
//                 products[m][j][i] = val;
//             }
//         }
//     }
//     return products;
// }

// // 数据加载函数
// template<typename T>
// T* LoadData(const std::string& data_path, size_t& n, size_t& d) {
//     ifstream fin(data_path, ios::in | ios::binary);
//     fin.read(reinterpret_cast<char*>(&n), sizeof(int));
//     fin.read(reinterpret_cast<char*>(&d), sizeof(int));
//     T* data = new T[n * d];
//     for (size_t i = 0; i < n; ++i) {
//         fin.read(reinterpret_cast<char*>(data + i * d), d * sizeof(T));
//     }
//     fin.close();
//     cerr << "load data " << data_path << "\n";
//     cerr << "dimension: " << d << "  number:" << n << "  size_per_element:" << sizeof(T) << "\n";
//     return data;
// }

// int main() {
//     size_t test_number = 0, base_number = 0, test_gt_d = 0, vecdim = 0;
//     string data_path = "/anndata/";

//     float* test_query = LoadData<float>(data_path + "DEEP100K.query.fbin", test_number, vecdim);
//     int* test_gt = LoadData<int>(data_path + "DEEP100K.gt.query.100k.top100.bin", test_number, test_gt_d);
//     float* base = LoadData<float>(data_path + "DEEP100K.base.100k.fbin", base_number, vecdim);

//     const size_t k = 10;
//     vector<size_t> rerank_values = {3200};
    
//     int num_threads = 8;  // 直接修改这个值即可调整线程数
    
//     cout << "Using " << num_threads << " threads" << endl;

//     auto codebooks = read_codebooks("files/codebooks.bin");
//     auto pq_codes_pair = load_pq_codes("files/pq_codes.bin");
//     auto cluster_products = read_cluster_products("files/cluster_products.bin");
//     const auto& pq_codes = pq_codes_pair.first;
//     const size_t M = codebooks.size();
//     const size_t sub_dim = codebooks[0][0].size();

//     IVFIndex ivf(num_threads);
//     ivf.load();

//     if (vecdim != ivf.dim) {
//         cerr << "Vector dimension mismatch" << endl;
//         return 1;
//     }

//     for (size_t rerank : rerank_values) {
//         double total_recall = 0.0;
//         long long total_latency = 0;

//         const size_t MAX_QUERIES = 2000;
//         const size_t actual_queries = min(MAX_QUERIES, test_number);

//         for (size_t i = 0; i < actual_queries; ++i) {
//             struct timeval start, end;
//             gettimeofday(&start, nullptr);

//             float* query = test_query + i * vecdim;
//             vector<int> clusters = ivf.search_clusters(query, num_threads);
            
//             vector<uint32_t> candidates;
//             ivf.parallel_gather_candidates(clusters, candidates, num_threads);
            
//             if (candidates.empty()) continue;

//             // PQ编码
//             vector<uint8_t> q_code(M);
//             for (size_t m = 0; m < M; ++m) {
//                 float* sub_query = query + m * sub_dim;
//                 float max_inner = -1e9;
//                 uint8_t best_code = 0;
//                 const size_t sub_dim_aligned = sub_dim & ~3;
                
//                 for (int k_idx = 0; k_idx < 256; ++k_idx) {
//                     float32x4_t sum_vec = vdupq_n_f32(0.0f);
//                     for (size_t d = 0; d < sub_dim_aligned; d += 4) {
//                         float32x4_t codebook_vec = vld1q_f32(&codebooks[m][k_idx][d]);
//                         float32x4_t sub_query_vec = vld1q_f32(sub_query + d);
//                         sum_vec = vmlaq_f32(sum_vec, codebook_vec, sub_query_vec);
//                     }
//                     float sum = vaddvq_f32(sum_vec);
//                     for (size_t d = sub_dim_aligned; d < sub_dim; ++d) {
//                         sum += codebooks[m][k_idx][d] * sub_query[d];
//                     }
//                     if (sum > max_inner) {
//                         max_inner = sum;
//                         best_code = k_idx;
//                     }
//                 }
//                 q_code[m] = best_code;
//             }

//             // PQ分数计算
//             vector<float> pq_scores(candidates.size(), 0.0f);
//             for (size_t i_db = 0; i_db < candidates.size(); ++i_db) {
//                 uint32_t vec_id = candidates[i_db];
//                 float score = 0.0f;
//                 for (size_t m = 0; m < M; ++m) {
//                     uint8_t q_idx = q_code[m];
//                     uint8_t db_idx = pq_codes[vec_id][m];
//                     score += cluster_products[m][q_idx][db_idx];
//                 }
//                 pq_scores[i_db] = score;
//             }

//             // 精排
//             vector<size_t> indices(candidates.size());
//             for (size_t i = 0; i < candidates.size(); ++i) indices[i] = i;
            
//             nth_element(indices.begin(), indices.begin() + rerank, indices.end(),
//                        [&pq_scores](size_t a, size_t b) { return pq_scores[a] > pq_scores[b]; });

//             priority_queue<pair<float, uint32_t>> final_result;
//             const size_t vecdim_aligned = vecdim & ~3;
//             for (size_t idx : indices) {
//                 if (idx >= candidates.size()) continue;
//                 uint32_t vec_id = candidates[idx];
//                 float32x4_t sum_vec = vdupq_n_f32(0.0f);
                
//                 for (size_t d = 0; d < vecdim_aligned; d += 4) {
//                     float32x4_t candidate_vec = vld1q_f32(base + vec_id * vecdim + d);
//                     float32x4_t query_vec = vld1q_f32(query + d);
//                     sum_vec = vmlaq_f32(sum_vec, candidate_vec - query_vec, candidate_vec - query_vec);
//                 }
//                 float dist = vaddvq_f32(sum_vec);
//                 for (size_t d = vecdim_aligned; d < vecdim; ++d) {
//                     float diff = base[vec_id * vecdim + d] - query[d];
//                     dist += diff * diff;
//                 }
//                 dist = sqrtf(dist);

//                 if (final_result.size() < k || dist < final_result.top().first) {
//                     if (final_result.size() == k) final_result.pop();
//                     final_result.push(make_pair(dist, vec_id));
//                 }
//             }

//             gettimeofday(&end, nullptr);
//             long long latency = (end.tv_sec * 1000000 + end.tv_usec) - (start.tv_sec * 1000000 + start.tv_usec);
//             total_latency += latency;

//             set<uint32_t> gtset;
//             for (size_t j = 0; j < k; ++j) {
//                 gtset.insert(static_cast<uint32_t>(test_gt[i * test_gt_d + j]));
//             }
//             size_t correct = 0;
//             while (!final_result.empty()) {
//                 if (gtset.count(final_result.top().second)) correct++;
//                 final_result.pop();
//             }
//             total_recall += static_cast<float>(correct) / k;
//         }

//         double average_recall = total_recall / actual_queries;
//         long long average_latency = total_latency / actual_queries;
//         cout << "Rerank: " << rerank 
//              << "\tIVF nprobe: " << ivf.nprobe
//              << "\tRecall@" << k << ": " << average_recall 
//              << "\tLatency: " << average_latency << " us\n";
//     }

//     delete[] test_query;
//     delete[] test_gt;
//     delete[] base;
//     return 0;
// }

// template float* LoadData<float>(const std::string&, size_t&, size_t&);
// template int* LoadData<int>(const std::string&, size_t&, size_t&);



//openMP1(非最终版本)
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
// #include <omp.h>

// using namespace std;

// struct ThreadSafeAccumulator {
//     double total_recall = 0.0;
//     long long total_latency = 0;
//     omp_lock_t mtx;

//     ThreadSafeAccumulator() { omp_init_lock(&mtx); }
//     ~ThreadSafeAccumulator() { omp_destroy_lock(&mtx); }
//     void add_recall(double recall) {
//         omp_set_lock(&mtx);
//         total_recall += recall;
//         omp_unset_lock(&mtx);
//     }
//     void add_latency(long long latency) {
//         omp_set_lock(&mtx);
//         total_latency += latency;
//         omp_unset_lock(&mtx);
//     }
// };

// struct IVFIndex {
//     int nlist, dim, nprobe;
//     vector<vector<float>> centroids;
//     vector<vector<uint32_t>> inverted_lists;

//     void load() {
//         string filename = "./files/ivf.index";
//         ifstream file(filename, ios::binary);
//         if (!file.is_open()) {
//             cerr << "Error loading IVF index from " << filename << endl;
//             exit(1);
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
//             file.read(reinterpret_cast<char*>(inverted_lists[i].data()), size * sizeof(uint32_t));
//         }
//         file.close();
//         nprobe = 32;
//     }

//     void parallel_compute_distances(const float* query, vector<pair<float, int>>& distances, int num_threads) {
//         distances.resize(nlist);
//         #pragma omp parallel for num_threads(num_threads) schedule(static)
//         for (int cid = 0; cid < nlist; ++cid) {
//             float dist_sq = 0.0f;
//             for (int d = 0; d < dim; ++d) {
//                 float diff = centroids[cid][d] - query[d];
//                 dist_sq += diff * diff;
//             }
//             distances[cid] = {sqrtf(dist_sq), cid};
//         }
//     }

//     void parallel_gather_candidates(const vector<int>& clusters, vector<uint32_t>& candidates, int num_threads) {
//         candidates.clear();
//         vector<vector<uint32_t>> local_candidates(num_threads);
        
//         #pragma omp parallel num_threads(num_threads)
//         {
//             int thread_id = omp_get_thread_num();
//             #pragma omp for schedule(static)
//             for (size_t i = 0; i < clusters.size(); ++i) {
//                 int cid = clusters[i];
//                 local_candidates[thread_id].insert(local_candidates[thread_id].end(), 
//                                                 inverted_lists[cid].begin(), 
//                                                 inverted_lists[cid].end());
//             }
//         }

//         for (const auto& lc : local_candidates) {
//             candidates.insert(candidates.end(), lc.begin(), lc.end());
//         }
//     }

//     vector<int> search_clusters(const float* query, int num_threads) {
//         vector<pair<float, int>> distances;
//         parallel_compute_distances(query, distances, num_threads);
        
//         nth_element(distances.begin(), distances.begin() + nprobe, distances.end());
//         sort(distances.begin(), distances.begin() + nprobe);
        
//         vector<int> clusters(nprobe);
//         for (int i = 0; i < nprobe; ++i) clusters[i] = distances[i].second;
//         return clusters;
//     }
// };

// vector<vector<vector<float>>> read_codebooks(const string& filename) {
//     ifstream file(filename, ios::binary);
//     int M, sub_dim;
//     file.read(reinterpret_cast<char*>(&M), sizeof(int));
//     file.read(reinterpret_cast<char*>(&sub_dim), sizeof(int));

//     vector<vector<vector<float>>> codebooks(M, vector<vector<float>>(256, vector<float>(sub_dim)));
//     for (int m = 0; m < M; ++m) {
//         for (int k = 0; k < 256; ++k) {
//             for (int d = 0; d < sub_dim; ++d) {
//                 file.read(reinterpret_cast<char*>(&codebooks[m][k][d]), sizeof(float));
//             }
//         }
//     }
//     return codebooks;
// }

// pair<vector<vector<uint8_t>>, pair<int, int>> load_pq_codes(const string& filename) {
//     ifstream file(filename, ios::binary);
//     int n, M;
//     file.read(reinterpret_cast<char*>(&n), sizeof(int));
//     file.read(reinterpret_cast<char*>(&M), sizeof(int));

//     vector<vector<uint8_t>> pq_codes(n, vector<uint8_t>(M));
//     for (int i = 0; i < n; ++i) {
//         for (int m = 0; m < M; ++m) {
//             file.read(reinterpret_cast<char*>(&pq_codes[i][m]), sizeof(uint8_t));
//         }
//     }
//     return make_pair(pq_codes, make_pair(n, M));
// }

// vector<vector<vector<float>>> read_cluster_products(const string& filename) {
//     ifstream file(filename, ios::binary);
//     int M;
//     file.read(reinterpret_cast<char*>(&M), sizeof(int));

//     vector<vector<vector<float>>> products(M, vector<vector<float>>(256, vector<float>(256, 0.0f)));
//     for (int m = 0; m < M; ++m) {
//         for (int i = 0; i < 256; ++i) {
//             for (int j = i; j < 256; ++j) {
//                 float val;
//                 file.read(reinterpret_cast<char*>(&val), sizeof(float));
//                 products[m][i][j] = val;
//                 products[m][j][i] = val;
//             }
//         }
//     }
//     return products;
// }

// template<typename T>
// T* LoadData(const std::string& data_path, size_t& n, size_t& d) {
//     ifstream fin(data_path, ios::in | ios::binary);
//     fin.read(reinterpret_cast<char*>(&n), sizeof(int));
//     fin.read(reinterpret_cast<char*>(&d), sizeof(int));
//     T* data = new T[n * d];
//     for (size_t i = 0; i < n; ++i) {
//         fin.read(reinterpret_cast<char*>(data + i * d), d * sizeof(T));
//     }
//     fin.close();
//     cerr << "load data " << data_path << "\n";
//     cerr << "dimension: " << d << "  number:" << n << "  size_per_element:" << sizeof(T) << "\n";
//     return data;
// }

// int main() {
//     size_t test_number = 0, base_number = 0, test_gt_d = 0, vecdim = 0;
//     string data_path = "/anndata/";

//     float* test_query = LoadData<float>(data_path + "DEEP100K.query.fbin", test_number, vecdim);
//     int* test_gt = LoadData<int>(data_path + "DEEP100K.gt.query.100k.top100.bin", test_number, test_gt_d);
//     float* base = LoadData<float>(data_path + "DEEP100K.base.100k.fbin", base_number, vecdim);

//     const size_t k = 10;
//     vector<size_t> rerank_values = {20, 50, 100,200, 500, 1000 };
    
//     int num_threads = omp_get_max_threads();
//     num_threads = min(num_threads, 16);
//     cout << "Using " << num_threads << " threads" << endl;

//     auto codebooks = read_codebooks("files/codebooks.bin");
//     auto pq_codes_pair = load_pq_codes("files/pq_codes.bin");
//     auto cluster_products = read_cluster_products("files/cluster_products.bin");
//     const auto& pq_codes = pq_codes_pair.first;
//     const size_t M = codebooks.size();
//     const size_t sub_dim = codebooks[0][0].size();

//     IVFIndex ivf;
//     ivf.load();

//     if (vecdim != ivf.dim) {
//         cerr << "Vector dimension mismatch" << endl;
//         return 1;
//     }

//     ThreadSafeAccumulator accumulator;
//     const size_t MAX_QUERIES = 2000; 
//     const size_t actual_queries = min(MAX_QUERIES, test_number);

//     for (size_t rerank : rerank_values) {
//         accumulator.total_recall = 0.0;
//         accumulator.total_latency = 0;

//         #pragma omp parallel for num_threads(num_threads) schedule(dynamic, 1) \
//             shared(accumulator, rerank, actual_queries, test_query, test_gt, test_gt_d, vecdim, base, ivf, codebooks, pq_codes, cluster_products, k, M, sub_dim)
//         for (size_t query_idx = 0; query_idx < actual_queries; ++query_idx) {
//             struct timeval start, end;
//             gettimeofday(&start, nullptr);

//             float* query = test_query + query_idx * vecdim;
//             vector<int> clusters = ivf.search_clusters(query, num_threads);
            
//             vector<uint32_t> candidates;
//             ivf.parallel_gather_candidates(clusters, candidates, num_threads);
            
//             if (candidates.empty()) {
//                 gettimeofday(&end, nullptr);
//                 long long latency = (end.tv_sec * 1000000 + end.tv_usec) - (start.tv_sec * 1000000 + start.tv_usec);
//                 accumulator.add_latency(latency);
//                 continue;
//             }

//             vector<uint8_t> q_code(M);
//             for (size_t m = 0; m < M; ++m) {
//                 float* sub_query = query + m * sub_dim;
//                 float max_inner = -1e9;
//                 uint8_t best_code = 0;
//                 const size_t sub_dim_aligned = sub_dim & ~3;
                
//                 for (int k_idx = 0; k_idx < 256; ++k_idx) {
//                     float32x4_t sum_vec = vdupq_n_f32(0.0f);
//                     for (size_t d = 0; d < sub_dim_aligned; d += 4) {
//                         sum_vec = vmlaq_f32(sum_vec, 
//                             vld1q_f32(&codebooks[m][k_idx][d]), 
//                             vld1q_f32(sub_query + d));
//                     }
//                     float sum = vaddvq_f32(sum_vec);
//                     for (size_t d = sub_dim_aligned; d < sub_dim; ++d) {
//                         sum += codebooks[m][k_idx][d] * sub_query[d];
//                     }
//                     if (sum > max_inner) {
//                         max_inner = sum;
//                         best_code = k_idx;
//                     }
//                 }
//                 q_code[m] = best_code;
//             }

//             vector<float> pq_scores(candidates.size());
//             for (size_t cand_idx = 0; cand_idx < candidates.size(); ++cand_idx) {
//                 uint32_t vec_id = candidates[cand_idx];
//                 float score = 0.0f;
//                 for (size_t m = 0; m < M; ++m) {
//                     uint8_t q_idx = q_code[m];
//                     uint8_t db_idx = pq_codes[vec_id][m];
//                     score += cluster_products[m][q_idx][db_idx];
//                 }
//                 pq_scores[cand_idx] = score;
//             }

//             vector<size_t> indices(candidates.size());
//             for (size_t idx = 0; idx < candidates.size(); ++idx) indices[idx] = idx;
            
//             size_t rerank_size = min(rerank, candidates.size());
//             nth_element(indices.begin(), indices.begin() + rerank_size, indices.end(),
//                        [&](size_t a, size_t b) { return pq_scores[a] > pq_scores[b]; });
            
//             // CORRECTED: Use max-heap for smallest distances
//             priority_queue<pair<float, uint32_t>> max_heap; // (dist, id) max-heap
            
//             const size_t vecdim_aligned = vecdim & ~3;
//             for (size_t j = 0; j < rerank_size; ++j) {
//                 uint32_t vec_id = candidates[indices[j]];
//                 float dist_sq = 0.0f;
//                 float32x4_t sum_vec = vdupq_n_f32(0.0f);
                
//                 for (size_t d = 0; d < vecdim_aligned; d += 4) {
//                     float32x4_t candidate_vec = vld1q_f32(base + vec_id * vecdim + d);
//                     float32x4_t query_vec = vld1q_f32(query + d);
//                     float32x4_t diff = vsubq_f32(candidate_vec, query_vec);
//                     sum_vec = vmlaq_f32(sum_vec, diff, diff);
//                 }
//                 dist_sq = vaddvq_f32(sum_vec);
//                 // Handle remaining elements
//                 for (size_t d = vecdim_aligned; d < vecdim; ++d) {
//                     float diff = base[vec_id * vecdim + d] - query[d];
//                     dist_sq += diff * diff;
//                 }
//                 float dist = sqrtf(dist_sq);
                
//                 if (max_heap.size() < k) {
//                     max_heap.push(make_pair(dist, vec_id));
//                 } else if (dist < max_heap.top().first) {
//                     max_heap.pop();
//                     max_heap.push(make_pair(dist, vec_id));
//                 }
//             }

//             // Extract results from heap
//             vector<uint32_t> result_ids;
//             while (!max_heap.empty()) {
//                 result_ids.push_back(max_heap.top().second);
//                 max_heap.pop();
//             }

//             gettimeofday(&end, nullptr);
//             long long latency = (end.tv_sec * 1000000 + end.tv_usec) - (start.tv_sec * 1000000 + start.tv_usec);
            
//             set<uint32_t> gtset;
//             for (size_t j = 0; j < k; ++j) {
//                 gtset.insert(static_cast<uint32_t>(test_gt[query_idx * test_gt_d + j]));
//             }
            
//             size_t correct = 0;
//             for (uint32_t id : result_ids) {
//                 if (gtset.find(id) != gtset.end()) {
//                     correct++;
//                 }
//             }
            
//             double recall = static_cast<double>(correct) / k;
//             accumulator.add_recall(recall);
//             accumulator.add_latency(latency);
//         }

//         double average_recall = accumulator.total_recall / actual_queries;
//         long long average_latency = accumulator.total_latency / actual_queries;
//         cout << "Rerank: " << rerank 
//              << "\tRecall@" << k << ": " << average_recall 
//              << "\tLatency: " << average_latency << " us\n";
//     }

//     delete[] test_query;
//     delete[] test_gt;
//     delete[] base;
//     return 0;
// }

// template float* LoadData<float>(const std::string&, size_t&, size_t&);
// template int* LoadData<int>(const std::string&, size_t&, size_t&);
//openMP2
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
#include <omp.h>
#include <cstdlib> // 添加头文件用于命令行参数解析

using namespace std;

// 使用线程本地存储替代全局锁
struct ThreadLocalAccumulator {
    double recall = 0.0;
    long long latency = 0;
    int count = 0;
};

struct ThreadSafeAccumulator {
    vector<ThreadLocalAccumulator> locals;
    omp_lock_t mtx;

    ThreadSafeAccumulator(int num_threads) : locals(num_threads) {
        omp_init_lock(&mtx);
    }

    ~ThreadSafeAccumulator() {
        omp_destroy_lock(&mtx);
    }

    void add_result(int thread_id, double recall, long long latency) {
        locals[thread_id].recall += recall;
        locals[thread_id].latency += latency;
        locals[thread_id].count++;
    }

    pair<double, long long> get_average() {
        double total_recall = 0.0;
        long long total_latency = 0;
        int total_count = 0;
        
        for (const auto& local : locals) {
            total_recall += local.recall;
            total_latency += local.latency;
            total_count += local.count;
        }
        
        return make_pair(total_recall / total_count, total_latency / total_count);
    }
};

struct IVFIndex {
    int nlist, dim, nprobe;
    vector<vector<float>> centroids;
    vector<vector<uint32_t>> inverted_lists;

    void load() {
        string filename = "./files/ivf.index";
        ifstream file(filename, ios::binary);
        if (!file.is_open()) {
            cerr << "Error loading IVF index from " << filename << endl;
            exit(1);
        }

        file.read(reinterpret_cast<char*>(&nlist), sizeof(int));
        file.read(reinterpret_cast<char*>(&dim), sizeof(int));
        
        centroids.resize(nlist, vector<float>(dim));
        for (int i = 0; i < nlist; ++i) {
            file.read(reinterpret_cast<char*>(centroids[i].data()), dim * sizeof(float));
        }

        inverted_lists.resize(nlist);
        for (int i = 0; i < nlist; ++i) {
            int size;
            file.read(reinterpret_cast<char*>(&size), sizeof(int));
            inverted_lists[i].resize(size);
            file.read(reinterpret_cast<char*>(inverted_lists[i].data()), size * sizeof(uint32_t));
        }
        file.close();
        nprobe = 16;
    }

    // 优化：增大任务粒度，使用动态调度
    void parallel_process_clusters(const float* query, vector<int>& selected_clusters, int num_threads) {
        vector<pair<float, int>> all_distances(nlist);
        
        #pragma omp parallel for schedule(dynamic, 100)
        for (int cid = 0; cid < nlist; ++cid) {
            float dist_sq = 0.0f;
            for (int d = 0; d < dim; ++d) {
                float diff = centroids[cid][d] - query[d];
                dist_sq += diff * diff;
            }
            all_distances[cid] = make_pair(sqrtf(dist_sq), cid);
        }

        nth_element(all_distances.begin(), all_distances.begin() + nprobe, all_distances.end());
        sort(all_distances.begin(), all_distances.begin() + nprobe);
        
        selected_clusters.resize(nprobe);
        for (int i = 0; i < nprobe; ++i) {
            selected_clusters[i] = all_distances[i].second;
        }
    }

    // 缓存优化：按簇顺序访问，预分配内存
    void parallel_gather_cluster_candidates(const vector<int>& clusters, vector<uint32_t>& candidates, int num_threads) {
        size_t total_size = 0;
        for (int cid : clusters) total_size += inverted_lists[cid].size();
        candidates.reserve(total_size);

        vector<vector<uint32_t>> local_candidates(num_threads);
        #pragma omp parallel for schedule(static)
        for (size_t i = 0; i < clusters.size(); ++i) {
            int thread_id = omp_get_thread_num();
            int cid = clusters[i];
            local_candidates[thread_id].insert(local_candidates[thread_id].end(),
                inverted_lists[cid].begin(), inverted_lists[cid].end());
        }

        for (const auto& lc : local_candidates) {
            candidates.insert(candidates.end(), lc.begin(), lc.end());
        }
    }
};

vector<vector<vector<float>>> read_codebooks(const string& filename) {
    ifstream file(filename, ios::binary);
    int M, sub_dim;
    file.read(reinterpret_cast<char*>(&M), sizeof(int));
    file.read(reinterpret_cast<char*>(&sub_dim), sizeof(int));

    vector<vector<vector<float>>> codebooks(M, vector<vector<float>>(256, vector<float>(sub_dim)));
    for (int m = 0; m < M; ++m) {
        for (int k = 0; k < 256; ++k) {
            for (int d = 0; d < sub_dim; ++d) {
                file.read(reinterpret_cast<char*>(&codebooks[m][k][d]), sizeof(float));
            }
        }
    }
    return codebooks;
}

pair<vector<vector<uint8_t>>, pair<int, int>> load_pq_codes(const string& filename) {
    ifstream file(filename, ios::binary);
    int n, M;
    file.read(reinterpret_cast<char*>(&n), sizeof(int));
    file.read(reinterpret_cast<char*>(&M), sizeof(int));

    vector<vector<uint8_t>> pq_codes(n, vector<uint8_t>(M));
    for (int i = 0; i < n; ++i) {
        for (int m = 0; m < M; ++m) {
            file.read(reinterpret_cast<char*>(&pq_codes[i][m]), sizeof(uint8_t));
        }
    }
    return make_pair(pq_codes, make_pair(n, M));
}

vector<vector<vector<float>>> read_cluster_products(const string& filename) {
    ifstream file(filename, ios::binary);
    int M;
    file.read(reinterpret_cast<char*>(&M), sizeof(int));

    vector<vector<vector<float>>> products(M, vector<vector<float>>(256, vector<float>(256, 0.0f)));
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

template<typename T>
T* LoadData(const std::string& data_path, size_t& n, size_t& d) {
    ifstream fin(data_path, ios::in | ios::binary);
    fin.read(reinterpret_cast<char*>(&n), sizeof(int));
    fin.read(reinterpret_cast<char*>(&d), sizeof(int));
    T* data = new T[n * d];
    for (size_t i = 0; i < n; ++i) {
        fin.read(reinterpret_cast<char*>(data + i * d), d * sizeof(T));
    }
    fin.close();
    cerr << "load data " << data_path << "\n";
    cerr << "dimension: " << d << "  number:" << n << "  size_per_element:" << sizeof(T) << "\n";
    return data;
}

int main(int argc, char* argv[]) {
    // 默认线程数
    // ===================== 在这里直接设置线程数 =====================
    int num_threads = 8;  // 直接修改这个值即可调整线程数
    
    // 设置OpenMP线程数
    omp_set_num_threads(num_threads);
    cout << "Using " << num_threads << " threads" << endl;
    
    // 解析命令行参数
    for (int i = 1; i < argc; ++i) {
        string arg = argv[i];
        if ((arg == "-t" || arg == "--threads") && i + 1 < argc) {
            num_threads = atoi(argv[++i]);
            if (num_threads <= 0) {
                cerr << "Invalid thread count: " << num_threads << ". Using default." << endl;
                num_threads = min(omp_get_max_threads(), 16);
            }
        }
        else if (arg == "-h" || arg == "--help") {
            cout << "Usage: " << argv[0] << " [options]\n"
                 << "Options:\n"
                 << "  -t, --threads N   Set number of threads (default: min(cores, 16))\n"
                 << "  -h, --help        Show this help message\n";
            return 0;
        }
    }

    // 设置OpenMP线程数
    omp_set_num_threads(num_threads);
    cout << "Using " << num_threads << " threads" << endl;

    size_t test_number = 0, base_number = 0, test_gt_d = 0, vecdim = 0;
    string data_path = "/anndata/";

    float* test_query = LoadData<float>(data_path + "DEEP100K.query.fbin", test_number, vecdim);
    int* test_gt = LoadData<int>(data_path + "DEEP100K.gt.query.100k.top100.bin", test_number, test_gt_d);
    float* base = LoadData<float>(data_path + "DEEP100K.base.100k.fbin", base_number, vecdim);

    const size_t k = 10;
    vector<size_t> rerank_values = {1100};
    
    auto codebooks = read_codebooks("files/codebooks.bin");
    auto pq_codes_pair = load_pq_codes("files/pq_codes.bin");
    auto cluster_products = read_cluster_products("files/cluster_products.bin");
    const auto& pq_codes = pq_codes_pair.first;
    const size_t M = codebooks.size();
    const size_t sub_dim = codebooks[0][0].size();

    IVFIndex ivf;
    ivf.load();

    if (vecdim != ivf.dim) {
        cerr << "Vector dimension mismatch" << endl;
        return 1;
    }

    for (size_t rerank : rerank_values) {
        ThreadSafeAccumulator accumulator(num_threads);
        const size_t MAX_QUERIES = 2000;
        const size_t actual_queries = min(MAX_QUERIES, test_number);

        #pragma omp parallel for schedule(dynamic, 50)
        for (size_t query_idx = 0; query_idx < actual_queries; ++query_idx) {
            int thread_id = omp_get_thread_num();
            struct timeval start, end;
            gettimeofday(&start, nullptr);

            float* query = test_query + query_idx * vecdim;
            vector<int> clusters;
            ivf.parallel_process_clusters(query, clusters, num_threads);
            
            vector<uint32_t> candidates;
            ivf.parallel_gather_cluster_candidates(clusters, candidates, num_threads);
            
            if (candidates.empty()) {
                gettimeofday(&end, nullptr);
                long long latency = (end.tv_sec * 1000000 + end.tv_usec) - (start.tv_sec * 1000000 + start.tv_usec);
                accumulator.add_result(thread_id, 0.0, latency);
                continue;
            }

            vector<uint8_t> q_code(M);
            for (size_t m = 0; m < M; ++m) {
                float* sub_query = query + m * sub_dim;
                float max_inner = -1e9;
                uint8_t best_code = 0;
                const size_t sub_dim_aligned = sub_dim & ~3;
                
                for (int k_idx = 0; k_idx < 256; ++k_idx) {
                    float32x4_t sum_vec = vdupq_n_f32(0.0f);
                    for (size_t d = 0; d < sub_dim_aligned; d += 4) {
                        sum_vec = vmlaq_f32(sum_vec, 
                            vld1q_f32(&codebooks[m][k_idx][d]), 
                            vld1q_f32(sub_query + d));
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

            vector<float> pq_scores(candidates.size());
            #pragma omp parallel for num_threads(min(num_threads, 4))
            for (size_t cand_idx = 0; cand_idx < candidates.size(); ++cand_idx) {
                uint32_t vec_id = candidates[cand_idx];
                float score = 0.0f;
                for (size_t m = 0; m < M; ++m) {
                    uint8_t q_idx = q_code[m];
                    uint8_t db_idx = pq_codes[vec_id][m];
                    score += cluster_products[m][q_idx][db_idx];
                }
                pq_scores[cand_idx] = score;
            }

            vector<size_t> indices(candidates.size());
            for (size_t idx = 0; idx < candidates.size(); ++idx) indices[idx] = idx;
            
            size_t rerank_size = min(rerank, candidates.size());
            nth_element(indices.begin(), indices.begin() + rerank_size, indices.end(),
                       [&](size_t a, size_t b) { return pq_scores[a] > pq_scores[b]; });
            
            priority_queue<pair<float, uint32_t>> max_heap;
            
            const size_t vecdim_aligned = vecdim & ~3;
            for (size_t j = 0; j < rerank_size; ++j) {
                uint32_t vec_id = candidates[indices[j]];
                float dist_sq = 0.0f;
                float32x4_t sum_vec = vdupq_n_f32(0.0f);
                
                for (size_t d = 0; d < vecdim_aligned; d += 4) {
                    float32x4_t candidate_vec = vld1q_f32(base + vec_id * vecdim + d);
                    float32x4_t query_vec = vld1q_f32(query + d);
                    float32x4_t diff = vsubq_f32(candidate_vec, query_vec);
                    sum_vec = vmlaq_f32(sum_vec, diff, diff);
                }
                dist_sq = vaddvq_f32(sum_vec);
                for (size_t d = vecdim_aligned; d < vecdim; ++d) {
                    float diff = base[vec_id * vecdim + d] - query[d];
                    dist_sq += diff * diff;
                }
                float dist = sqrtf(dist_sq);
                
                if (max_heap.size() < k) {
                    max_heap.push(make_pair(dist, vec_id));
                } else if (dist < max_heap.top().first) {
                    max_heap.pop();
                    max_heap.push(make_pair(dist, vec_id));
                }
            }

            vector<uint32_t> result_ids;
            while (!max_heap.empty()) {
                result_ids.push_back(max_heap.top().second);
                max_heap.pop();
            }

            gettimeofday(&end, nullptr);
            long long latency = (end.tv_sec * 1000000 + end.tv_usec) - (start.tv_sec * 1000000 + start.tv_usec);
            
            set<uint32_t> gtset;
            for (size_t j = 0; j < k; ++j) {
                gtset.insert(static_cast<uint32_t>(test_gt[query_idx * test_gt_d + j]));
            }
            
            size_t correct = 0;
            for (uint32_t id : result_ids) {
                if (gtset.find(id) != gtset.end()) {
                    correct++;
                }
            }
            
            double recall = static_cast<double>(correct) / k;
            accumulator.add_result(thread_id, recall, latency);
        }

        // 使用C++11兼容方式获取结果
        pair<double, long long> result = accumulator.get_average();
        double average_recall = result.first;
        long long average_latency = result.second;
        
        cout << "Rerank: " << rerank 
             << "\tRecall@" << k << ": " << average_recall 
             << "\tLatency: " << average_latency << " us\n";
    }

    delete[] test_query;
    delete[] test_gt;
    delete[] base;
    return 0;
}

template float* LoadData<float>(const std::string&, size_t&, size_t&);
template int* LoadData<int>(const std::string&, size_t&, size_t&);
