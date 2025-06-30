//初始优化
// #include <iostream>
// #include <fstream>
// #include <vector>
// #include <queue>
// #include <algorithm>
// #include <cmath>
// #include <cassert>
// #include <set>
// #include <arm_neon.h>
// #include <mpi.h>
// #include <cstdlib>
// #include <omp.h>

// using namespace std;

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
//         nprobe = 24;
//     }

//     void process_clusters(const float* query, vector<int>& selected_clusters) {
//         vector<pair<float, int>> all_distances(nlist);
        
//         #pragma omp parallel for
//         for (int cid = 0; cid < nlist; ++cid) {
//             float dist_sq = 0.0f;
//             for (int d = 0; d < dim; ++d) {
//                 float diff = centroids[cid][d] - query[d];
//                 dist_sq += diff * diff;
//             }
//             all_distances[cid] = make_pair(dist_sq, cid); // 使用平方距离
//         }

//         nth_element(all_distances.begin(), all_distances.begin() + nprobe, all_distances.end());
//         sort(all_distances.begin(), all_distances.begin() + nprobe);
        
//         selected_clusters.resize(nprobe);
//         for (int i = 0; i < nprobe; ++i) {
//             selected_clusters[i] = all_distances[i].second;
//         }
//     }

//     void gather_cluster_candidates(const vector<int>& clusters, vector<uint32_t>& candidates) {
//         size_t total_size = 0;
//         for (int cid : clusters) total_size += inverted_lists[cid].size();
//         candidates.reserve(total_size);

//         for (int cid : clusters) {
//             candidates.insert(candidates.end(), inverted_lists[cid].begin(), inverted_lists[cid].end());
//         }
//     }
// };

// // Load codebooks, PQ codes, and cluster products as before
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

// int main(int argc, char* argv[]) {
//     MPI_Init(&argc, &argv);
//     int rank, size;
//     MPI_Comm_rank(MPI_COMM_WORLD, &rank);
//     MPI_Comm_size(MPI_COMM_WORLD, &size);

//     size_t test_number = 0, base_number = 0, test_gt_d = 0, vecdim = 0;
//     string data_path = "/anndata/";

//     float* test_query = nullptr;
//     int* test_gt = nullptr;
//     float* base = nullptr;

//     // 在 rank 0 加载数据并广播
//     if (rank == 0) {
//         test_query = LoadData<float>(data_path + "DEEP100K.query.fbin", test_number, vecdim);
//         test_gt = LoadData<int>(data_path + "DEEP100K.gt.query.100k.top100.bin", test_number, test_gt_d);
//         base = LoadData<float>(data_path + "DEEP100K.base.100k.fbin", base_number, vecdim);
//     }
//     MPI_Bcast(&test_number, 1, MPI_UNSIGNED_LONG, 0, MPI_COMM_WORLD);
//     MPI_Bcast(&base_number, 1, MPI_UNSIGNED_LONG, 0, MPI_COMM_WORLD);
//     MPI_Bcast(&test_gt_d, 1, MPI_UNSIGNED_LONG, 0, MPI_COMM_WORLD);
//     MPI_Bcast(&vecdim, 1, MPI_UNSIGNED_LONG, 0, MPI_COMM_WORLD);

//     if (rank != 0) {
//         test_query = new float[test_number * vecdim];
//         test_gt = new int[test_number * test_gt_d];
//         base = new float[base_number * vecdim];
//     }
//     MPI_Bcast(test_query, test_number * vecdim, MPI_FLOAT, 0, MPI_COMM_WORLD);
//     MPI_Bcast(test_gt, test_number * test_gt_d, MPI_INT, 0, MPI_COMM_WORLD);
//     MPI_Bcast(base, base_number * vecdim, MPI_FLOAT, 0, MPI_COMM_WORLD);

//     const size_t k = 10;
//     vector<size_t> rerank_values = {900};

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
//         MPI_Finalize();
//         return 1;
//     }

//     // 使用MPI并行处理查询，簇内使用OpenMP并行化
//     MPI_Barrier(MPI_COMM_WORLD);
//     double total_start_time = MPI_Wtime(); // 记录总开始时间
//     #pragma omp parallel for
//     for (size_t rerank : rerank_values) {
//         double local_recall_sum = 0.0;
//         long long local_latency_sum = 0;
//         int local_count = 0;
//         long long local_query_count = 0; // 新增：用于统计每个进程处理的查询数量

//         const size_t MAX_QUERIES = 2000;
//         const size_t total_queries = min(MAX_QUERIES, test_number);
        
//         size_t remainder = total_queries % size;
//         size_t queries_per_process = total_queries / size;
//         size_t start_query = rank * queries_per_process + min(static_cast<size_t>(rank), remainder);
//         size_t end_query = start_query + queries_per_process + (rank < remainder ? 1 : 0);

//         for (size_t query_idx = start_query; query_idx < end_query; ++query_idx) {
//             double start_time = MPI_Wtime();

//             float* query = test_query + query_idx * vecdim;
//             vector<int> clusters;
//             ivf.process_clusters(query, clusters);
            
//             vector<uint32_t> candidates;
//             ivf.gather_cluster_candidates(clusters, candidates);
            
//             if (candidates.empty()) {
//                 double end_time = MPI_Wtime();
//                 long long latency = static_cast<long long>((end_time - start_time) * 1e6);
//                 local_recall_sum += 0.0;
//                 local_latency_sum += latency;
//                 local_query_count++; // 增加处理的查询数量
//                 local_count++;
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
            
//             priority_queue<pair<float, uint32_t>> max_heap;
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

//             vector<uint32_t> result_ids;
//             while (!max_heap.empty()) {
//                 result_ids.push_back(max_heap.top().second);
//                 max_heap.pop();
//             }

//             double end_time = MPI_Wtime();
//             long long latency = static_cast<long long>((end_time - start_time) * 1e6);
            
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
//             local_recall_sum += recall;
//             local_latency_sum += latency;
//             local_query_count++; // 增加处理的查询数量
//             local_count++;
//         }

//         double total_recall_sum;
//         long long total_latency_sum;
//         int total_count;
//         long long total_query_count;

//         MPI_Reduce(&local_recall_sum, &total_recall_sum, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
//         MPI_Reduce(&local_latency_sum, &total_latency_sum, 1, MPI_LONG_LONG, MPI_SUM, 0, MPI_COMM_WORLD);
//         MPI_Reduce(&local_count, &total_count, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
//         MPI_Reduce(&local_query_count, &total_query_count, 1, MPI_LONG_LONG, MPI_SUM, 0, MPI_COMM_WORLD);

//         // 在 rank 0 进行平均延迟计算
//         if (rank == 0) {
//             double average_recall = total_recall_sum / total_count;
//             long long average_latency = total_latency_sum / total_query_count; // 除以处理的查询总数
//             cout << "Rerank: " << rerank 
//                  << "\tRecall@" << k << ": " << average_recall<< endl;
//         }
//     }
//     MPI_Barrier(MPI_COMM_WORLD); // 确保所有进程都完成
//     double total_end_time = MPI_Wtime(); // 记录总结束时间
//     // 在 rank 0 进行平均延迟计算
//     if (rank == 0) {
//         double total_time = total_end_time - total_start_time;
//         cout << "Total time for all queries: " << total_time << " seconds" << endl;
//         cout << "Average latency per query: " << (total_end_time - total_start_time) / 2000 * 1e6 << " us" << endl;

//     }

//     delete[] test_query;
//     delete[] test_gt;
//     delete[] base;
//     MPI_Finalize();
//     return 0;
// }

// template float* LoadData<float>(const std::string&, size_t&, size_t&);
// template int* LoadData<int>(const std::string&, size_t&, size_t&);

//二次优化
#include <iostream>
#include <fstream>
#include <vector>
#include <queue>
#include <algorithm>
#include <cmath>
#include <cassert>
#include <set>
#include <arm_neon.h>
#include <mpi.h>
#include <cstdlib>
#include <omp.h>

using namespace std;

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
        nprobe = 24;
    }

    void process_clusters(const float* query, vector<int>& selected_clusters) {
        vector<pair<float, int>> all_distances(nlist);

        #pragma omp parallel for
        for (int cid = 0; cid < nlist; ++cid) {
            float dist_sq = 0.0f;
            for (int d = 0; d < dim; ++d) {
                float diff = centroids[cid][d] - query[d];
                dist_sq += diff * diff;
            }
            all_distances[cid] = make_pair(dist_sq, cid);
        }

        nth_element(all_distances.begin(), all_distances.begin() + nprobe, all_distances.end());
        sort(all_distances.begin(), all_distances.begin() + nprobe);
        
        selected_clusters.resize(nprobe);
        for (int i = 0; i < nprobe; ++i) {
            selected_clusters[i] = all_distances[i].second;
        }
    }

    void gather_cluster_candidates(const vector<int>& clusters, vector<uint32_t>& candidates) {
        size_t total_size = 0;
        for (int cid : clusters) total_size += inverted_lists[cid].size();
        candidates.reserve(total_size);

        for (int cid : clusters) {
            candidates.insert(candidates.end(), inverted_lists[cid].begin(), inverted_lists[cid].end());
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
    const size_t MAX_QUERIES = 2000;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, nullptr); // Enable multi-threading support
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    size_t test_number = 0, base_number = 0, test_gt_d = 0, vecdim = 0;
    string data_path = "/anndata/";

    float* test_query = nullptr;
    int* test_gt = nullptr;
    float* base = nullptr;

    // Load data on rank 0 and broadcast
    if (rank == 0) {
        test_query = LoadData<float>(data_path + "DEEP100K.query.fbin", test_number, vecdim);
        test_gt = LoadData<int>(data_path + "DEEP100K.gt.query.100k.top100.bin", test_number, test_gt_d);
        base = LoadData<float>(data_path + "DEEP100K.base.100k.fbin", base_number, vecdim);
    }
    MPI_Bcast(&test_number, 1, MPI_UNSIGNED_LONG, 0, MPI_COMM_WORLD);
    MPI_Bcast(&base_number, 1, MPI_UNSIGNED_LONG, 0, MPI_COMM_WORLD);
    MPI_Bcast(&test_gt_d, 1, MPI_UNSIGNED_LONG, 0, MPI_COMM_WORLD);
    MPI_Bcast(&vecdim, 1, MPI_UNSIGNED_LONG, 0, MPI_COMM_WORLD);

    if (rank != 0) {
        test_query = new float[test_number * vecdim];
        test_gt = new int[test_number * test_gt_d];
        base = new float[base_number * vecdim];
    }
    MPI_Bcast(test_query, test_number * vecdim, MPI_FLOAT, 0, MPI_COMM_WORLD);
    MPI_Bcast(test_gt, test_number * test_gt_d, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(base, base_number * vecdim, MPI_FLOAT, 0, MPI_COMM_WORLD);

    const size_t k = 10;
    vector<size_t> rerank_values = {900};

    auto codebooks = read_codebooks("files/codebooks.bin");
    auto pq_codes_pair = load_pq_codes("files/pq_codes.bin");
    auto cluster_products = read_cluster_products("files/cluster_products.bin");
    const auto& pq_codes = pq_codes_pair.first;
    const size_t M = codebooks.size();
    const size_t sub_dim = codebooks[0][0].size();

    IVFIndex ivf;
    ivf.load();

    MPI_Barrier(MPI_COMM_WORLD);
    double total_start_time = MPI_Wtime();

    // Dynamic task scheduling
    #pragma omp parallel for schedule(dynamic)
    for (size_t rerank : rerank_values) {
        double local_recall_sum = 0.0;
        long long local_latency_sum = 0;
        long long local_query_count = 0;

        const size_t total_queries = min(MAX_QUERIES, test_number);
        
        size_t remainder = total_queries % size;
        size_t queries_per_process = total_queries / size;
        size_t start_query = rank * queries_per_process + min(static_cast<size_t>(rank), remainder);
        size_t end_query = start_query + queries_per_process + (rank < remainder ? 1 : 0);

        for (size_t query_idx = start_query; query_idx < end_query; ++query_idx) {
            double start_time = MPI_Wtime();

            float* query = test_query + query_idx * vecdim;
            vector<int> clusters;
            ivf.process_clusters(query, clusters);
            
            vector<uint32_t> candidates;
            ivf.gather_cluster_candidates(clusters, candidates);
            
            if (candidates.empty()) {
                double end_time = MPI_Wtime();
                long long latency = static_cast<long long>((end_time - start_time) * 1e6);
                local_recall_sum += 0.0;
                local_latency_sum += latency;
                local_query_count++;
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

            double end_time = MPI_Wtime();
            long long latency = static_cast<long long>((end_time - start_time) * 1e6);

            // Compute recall
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
            local_recall_sum += recall;
            local_latency_sum += latency;
            local_query_count++;
        }

        // Aggregate recall and latency across processes
        double total_recall_sum;
        long long total_latency_sum;
        long long total_query_count;

        MPI_Reduce(&local_recall_sum, &total_recall_sum, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
        MPI_Reduce(&local_latency_sum, &total_latency_sum, 1, MPI_LONG_LONG, MPI_SUM, 0, MPI_COMM_WORLD);
        MPI_Reduce(&local_query_count, &total_query_count, 1, MPI_LONG_LONG, MPI_SUM, 0, MPI_COMM_WORLD);

        // Output results for this rerank value on rank 0
        if (rank == 0) {
            double average_recall = total_recall_sum / total_query_count;
            long long average_latency = total_latency_sum / total_query_count;
            cout << "Rerank: " << rerank 
                 << "\tRecall@" << k << ": " << average_recall
                 <<endl;
        }
    }

    MPI_Barrier(MPI_COMM_WORLD);
    double total_end_time = MPI_Wtime();
    if (rank == 0) {
        double total_time = total_end_time - total_start_time;
        cout << "Total time for all queries: " << total_time << " seconds" << endl;
        cout << "Average latency per query: " << total_time / MAX_QUERIES * 1e6 << " us" << endl;
    }

    delete[] test_query;
    delete[] test_gt;
    delete[] base;
    MPI_Finalize();
    return 0;
}

template float* LoadData<float>(const std::string&, size_t&, size_t&);
template int* LoadData<int>(const std::string&, size_t&, size_t&);
