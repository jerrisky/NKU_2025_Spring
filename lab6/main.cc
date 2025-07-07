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

二次优化
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
//修正的多线程
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
// #include <cstdlib>

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

//     // 查询内并行：并行计算簇距离
//     void parallel_process_clusters(const float* query, vector<int>& selected_clusters, int num_threads) {
//         vector<pair<float, int>> all_distances(nlist);
        
//         #pragma omp parallel for num_threads(num_threads) schedule(dynamic, 100)
//         for (int cid = 0; cid < nlist; ++cid) {
//             float dist_sq = 0.0f;
//             #pragma omp simd reduction(+:dist_sq)
//             for (int d = 0; d < dim; ++d) {
//                 float diff = centroids[cid][d] - query[d];
//                 dist_sq += diff * diff;
//             }
//             all_distances[cid] = make_pair(sqrtf(dist_sq), cid);
//         }

//         nth_element(all_distances.begin(), all_distances.begin() + nprobe, all_distances.end());
//         sort(all_distances.begin(), all_distances.begin() + nprobe);
        
//         selected_clusters.resize(nprobe);
//         for (int i = 0; i < nprobe; ++i) {
//             selected_clusters[i] = all_distances[i].second;
//         }
//     }

//     // 查询内并行：并行聚集候选向量
//     void parallel_gather_cluster_candidates(const vector<int>& clusters, vector<uint32_t>& candidates, int num_threads) {
//         size_t total_size = 0;
//         for (int cid : clusters) total_size += inverted_lists[cid].size();
//         candidates.clear();
//         candidates.reserve(total_size);

//         #pragma omp parallel num_threads(num_threads)
//         {
//             vector<uint32_t> local_candidates;
//             local_candidates.reserve(total_size / omp_get_num_threads() + 100);
            
//             #pragma omp for schedule(static)
//             for (size_t i = 0; i < clusters.size(); ++i) {
//                 int cid = clusters[i];
//                 local_candidates.insert(local_candidates.end(),
//                     inverted_lists[cid].begin(), inverted_lists[cid].end());
//             }
            
//             #pragma omp critical
//             candidates.insert(candidates.end(), local_candidates.begin(), local_candidates.end());
//         }
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

// // 精确计时函数
// inline double get_time_us() {
//     struct timeval tv;
//     gettimeofday(&tv, nullptr);
//     return tv.tv_sec * 1000000.0 + tv.tv_usec;
// }

// int main(int argc, char* argv[]) {
//     // ===================== 设置线程数 =====================
//     int num_threads = 8;
//     if (argc > 1) {
//         num_threads = atoi(argv[1]);
//     }
//     omp_set_num_threads(num_threads);
//     cout << "Using " << num_threads << " threads for intra-query parallelism" << endl;

//     size_t test_number = 0, base_number = 0, test_gt_d = 0, vecdim = 0;
//     string data_path = "/anndata/";

//     float* test_query = LoadData<float>(data_path + "DEEP100K.query.fbin", test_number, vecdim);
//     int* test_gt = LoadData<int>(data_path + "DEEP100K.gt.query.100k.top100.bin", test_number, test_gt_d);
//     float* base = LoadData<float>(data_path + "DEEP100K.base.100k.fbin", base_number, vecdim);

//     const size_t k = 10;
//     vector<size_t> rerank_values = {1100};
    
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

//     // 允许OpenMP嵌套并行
//     omp_set_nested(1);
//     omp_set_max_active_levels(2);

//     for (size_t rerank : rerank_values) {
//         const size_t MAX_QUERIES = 2000;
//         const size_t actual_queries = min(MAX_QUERIES, test_number);
        
//         double total_recall = 0.0;
//         double total_query_time_us = 0.0;
//         double min_query_time_us = 1e20;
//         double max_query_time_us = 0.0;
        
//         double total_start_wall = get_time_us();

//         // 串行处理每个查询，但在查询内部使用并行
//         for (size_t query_idx = 0; query_idx < actual_queries; ++query_idx) {
//             double query_start = get_time_us();
            
//             float* query = test_query + query_idx * vecdim;
//             vector<int> clusters;
//             ivf.parallel_process_clusters(query, clusters, num_threads);
            
//             vector<uint32_t> candidates;
//             ivf.parallel_gather_cluster_candidates(clusters, candidates, num_threads);
            
//             if (candidates.empty()) {
//                 double query_end = get_time_us();
//                 double latency = query_end - query_start;
//                 total_query_time_us += latency;
//                 min_query_time_us = min(min_query_time_us, latency);
//                 max_query_time_us = max(max_query_time_us, latency);
//                 continue;
//             }

//             // 并行计算查询的PQ编码
//             vector<uint8_t> q_code(M);
//             #pragma omp parallel for num_threads(num_threads) schedule(static)
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

//             // 并行计算PQ分数
//             vector<float> pq_scores(candidates.size());
//             #pragma omp parallel for num_threads(num_threads) schedule(static)
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
//             // 并行计算重排向量的距离
//             #pragma omp parallel for num_threads(num_threads) schedule(static)
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
                
//                 #pragma omp critical
//                 {
//                     if (max_heap.size() < k) {
//                         max_heap.push(make_pair(dist, vec_id));
//                     } else if (dist < max_heap.top().first) {
//                         max_heap.pop();
//                         max_heap.push(make_pair(dist, vec_id));
//                     }
//                 }
//             }

//             vector<uint32_t> result_ids;
//             while (!max_heap.empty()) {
//                 result_ids.push_back(max_heap.top().second);
//                 max_heap.pop();
//             }

//             double query_end = get_time_us();
//             double latency = query_end - query_start;
//             total_query_time_us += latency;
//             min_query_time_us = min(min_query_time_us, latency);
//             max_query_time_us = max(max_query_time_us, latency);
            
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
//             total_recall += recall;
//         }

//         double total_end_wall = get_time_us();
//         double total_wall_time_us = total_end_wall - total_start_wall;
        
//         double avg_recall = total_recall / actual_queries;
//         double avg_query_time_us = total_query_time_us / actual_queries;
//         double avg_query_time_ms = avg_query_time_us / 1000.0;
//         double qps = actual_queries / (total_wall_time_us / 1000000.0);
        
//         cout << "Rerank: " << rerank 
//              << "\tRecall@" << k << ": " << avg_recall 
//              << "\tAvg Query Time: " << avg_query_time_ms << " ms"
//              << "\tMin: " << min_query_time_us/1000.0 << " ms"
//              << "\tMax: " << max_query_time_us/1000.0 << " ms"
//              << "\tQPS: " << qps << endl;
//     }

//     delete[] test_query;
//     delete[] test_gt;
//     delete[] base;
//     return 0;
// }

// template float* LoadData<float>(const std::string&, size_t&, size_t&);
// template int* LoadData<int>(const std::string&, size_t&, size_t&);
//补充查询内MPI
// #include <iostream>
// #include <fstream>
// #include <vector>
// #include <queue>
// #include <algorithm>
// #include <cmath>
// #include <cassert>
// #include <set>
// #include <numeric>
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
//             all_distances[cid] = make_pair(dist_sq, cid);
//         }

//         nth_element(all_distances.begin(), all_distances.begin() + nprobe, all_distances.end());
//         sort(all_distances.begin(), all_distances.begin() + nprobe);
        
//         selected_clusters.resize(nprobe);
//         for (int i = 0; i < nprobe; ++i) {
//             selected_clusters[i] = all_distances[i].second;
//         }
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

// int main(int argc, char* argv[]) {
//     const size_t MAX_QUERIES = 2000;
//     MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, nullptr);
//     int rank, size;
//     MPI_Comm_rank(MPI_COMM_WORLD, &rank);
//     MPI_Comm_size(MPI_COMM_WORLD, &size);

//     size_t test_number = 0, base_number = 0, test_gt_d = 0, vecdim = 0;
//     string data_path = "/anndata/";

//     float* test_query = nullptr;
//     int* test_gt = nullptr;
//     float* base = nullptr;

//     // 仅rank 0加载数据
//     if (rank == 0) {
//         test_query = LoadData<float>(data_path + "DEEP100K.query.fbin", test_number, vecdim);
//         test_gt = LoadData<int>(data_path + "DEEP100K.gt.query.100k.top100.bin", test_number, test_gt_d);
//         base = LoadData<float>(data_path + "DEEP100K.base.100k.fbin", base_number, vecdim);
//     }
    
//     // 广播元数据
//     MPI_Bcast(&test_number, 1, MPI_UNSIGNED_LONG, 0, MPI_COMM_WORLD);
//     MPI_Bcast(&base_number, 1, MPI_UNSIGNED_LONG, 0, MPI_COMM_WORLD);
//     MPI_Bcast(&test_gt_d, 1, MPI_UNSIGNED_LONG, 0, MPI_COMM_WORLD);
//     MPI_Bcast(&vecdim, 1, MPI_UNSIGNED_LONG, 0, MPI_COMM_WORLD);

//     // 分配内存并广播数据
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

//     // 所有进程加载索引文件
//     auto codebooks = read_codebooks("files/codebooks.bin");
//     auto pq_codes_pair = load_pq_codes("files/pq_codes.bin");
//     auto cluster_products = read_cluster_products("files/cluster_products.bin");
//     const auto& pq_codes = pq_codes_pair.first;
//     const size_t M = codebooks.size();
//     const size_t sub_dim = codebooks[0][0].size();

//     IVFIndex ivf;
//     ivf.load();

//     MPI_Barrier(MPI_COMM_WORLD);
//     double total_start_time = MPI_Wtime();

//     for (size_t rerank : rerank_values) {
//         double recall_sum = 0.0;
//         long long latency_sum = 0;
//         const size_t total_queries = min(MAX_QUERIES, test_number);

//         for (size_t query_idx = 0; query_idx < total_queries; query_idx++) {
//             double start_time = MPI_Wtime();
//             float* query = test_query + query_idx * vecdim;

//             // 步骤1: 计算选中的簇
//             vector<int> selected_clusters;
//             if (rank == 0) {
//                 ivf.process_clusters(query, selected_clusters);
//             }

//             int nprobe = ivf.nprobe;
//             selected_clusters.resize(nprobe);
//             MPI_Bcast(selected_clusters.data(), nprobe, MPI_INT, 0, MPI_COMM_WORLD);

//             // 步骤2: 计算每个簇的候选向量数量
//             vector<int> cluster_sizes(nprobe);
//             for (int i = 0; i < nprobe; i++) {
//                 cluster_sizes[i] = ivf.inverted_lists[selected_clusters[i]].size();
//             }

//             // 计算总候选向量数
//             int total_candidates = 0;
//             for (int size : cluster_sizes) {
//                 total_candidates += size;
//             }

//             // 如果没有候选向量，跳过
//             if (total_candidates == 0) {
//                 if (rank == 0) {
//                     double end_time = MPI_Wtime();
//                     latency_sum += static_cast<long long>((end_time - start_time) * 1e6);
//                     recall_sum += 0.0; // 召回率为0
//                 }
//                 continue;
//             }

//             // 步骤3: 基于候选向量数量分配任务（负载均衡）
//             vector<int> displs(size, 0);
//             vector<int> counts(size, 0);
//             int avg_candidates = total_candidates / size;
//             int remaining = total_candidates % size;
            
//             int current_idx = 0;
//             for (int i = 0; i < size; i++) {
//                 int target_count = avg_candidates + (i < remaining ? 1 : 0);
//                 int accumulated = 0;
                
//                 while (accumulated < target_count && current_idx < nprobe) {
//                     int take = min(target_count - accumulated, cluster_sizes[current_idx]);
//                     accumulated += take;
//                     counts[i] += take;
                    
//                     // 移动到下一个簇
//                     cluster_sizes[current_idx] -= take;
//                     if (cluster_sizes[current_idx] == 0) {
//                         current_idx++;
//                     }
//                 }
                
//                 if (i < size - 1) {
//                     displs[i+1] = displs[i] + counts[i];
//                 }
//             }

//             // 步骤4: 分配候选向量
//             vector<uint32_t> my_candidates;
//             my_candidates.reserve(counts[rank]);
            
//             int current_cluster = 0;
//             int current_offset = 0;
//             int accumulated = 0;
            
//             // 定位到当前进程的起始位置
//             for (int i = 0; i < rank; i++) {
//                 accumulated += counts[i];
//                 while (accumulated > 0) {
//                     int take = min(accumulated, (int)ivf.inverted_lists[selected_clusters[current_cluster]].size() - current_offset);
//                     accumulated -= take;
//                     current_offset += take;
                    
//                     if (current_offset >= (int)ivf.inverted_lists[selected_clusters[current_cluster]].size()) {
//                         current_cluster++;
//                         current_offset = 0;
//                     }
//                 }
//             }
            
//             // 收集当前进程的候选向量
//             int to_collect = counts[rank];
//             while (to_collect > 0 && current_cluster < nprobe) {
//                 const auto& list = ivf.inverted_lists[selected_clusters[current_cluster]];
//                 int available = list.size() - current_offset;
//                 int take = min(to_collect, available);
                
//                 my_candidates.insert(my_candidates.end(), 
//                                    list.begin() + current_offset, 
//                                    list.begin() + current_offset + take);
                
//                 to_collect -= take;
//                 current_offset += take;
                
//                 if (current_offset >= (int)list.size()) {
//                     current_cluster++;
//                     current_offset = 0;
//                 }
//             }

//             // 步骤5: 计算查询的PQ编码
//             vector<uint8_t> q_code(M);
//             if (rank == 0) {
//                 for (size_t m = 0; m < M; ++m) {
//                     float* sub_query = query + m * sub_dim;
//                     float max_inner = -1e9;
//                     uint8_t best_code = 0;
//                     const size_t sub_dim_aligned = sub_dim & ~3;

//                     for (int k_idx = 0; k_idx < 256; ++k_idx) {
//                         float32x4_t sum_vec = vdupq_n_f32(0.0f);
//                         for (size_t d = 0; d < sub_dim_aligned; d += 4) {
//                             sum_vec = vmlaq_f32(sum_vec, 
//                                 vld1q_f32(&codebooks[m][k_idx][d]), 
//                                 vld1q_f32(sub_query + d));
//                         }
//                         float sum = vaddvq_f32(sum_vec);
//                         for (size_t d = sub_dim_aligned; d < sub_dim; ++d) {
//                             sum += codebooks[m][k_idx][d] * sub_query[d];
//                         }
//                         if (sum > max_inner) {
//                             max_inner = sum;
//                             best_code = k_idx;
//                         }
//                     }
//                     q_code[m] = best_code;
//                 }
//             }
//             MPI_Bcast(q_code.data(), M, MPI_UNSIGNED_CHAR, 0, MPI_COMM_WORLD);

//             // 步骤6: 计算候选向量的PQ分数
//             vector<pair<uint32_t, float>> my_candidate_scores;
//             my_candidate_scores.reserve(my_candidates.size());
            
//             for (uint32_t vec_id : my_candidates) {
//                 float score = 0.0f;
//                 for (size_t m = 0; m < M; ++m) {
//                     uint8_t q_idx = q_code[m];
//                     uint8_t db_idx = pq_codes[vec_id][m];
//                     score += cluster_products[m][q_idx][db_idx];
//                 }
//                 my_candidate_scores.push_back(make_pair(vec_id, score));
//             }

//             // 步骤7: 收集所有候选向量和分数
//             vector<int> all_counts(size);
//             MPI_Gather(&counts[rank], 1, MPI_INT, all_counts.data(), 1, MPI_INT, 0, MPI_COMM_WORLD);
            
//             vector<int> displs_candidates(size, 0);
//             if (rank == 0) {
//                 for (int i = 1; i < size; i++) {
//                     displs_candidates[i] = displs_candidates[i-1] + all_counts[i-1];
//                 }
//             }

//             vector<uint32_t> all_candidates(total_candidates);
//             vector<float> all_scores(total_candidates);
            
//             // 准备发送缓冲
//             vector<uint32_t> my_candidate_ids;
//             vector<float> my_scores;
//             for (const auto& p : my_candidate_scores) {
//                 my_candidate_ids.push_back(p.first);
//                 my_scores.push_back(p.second);
//             }
            
//             MPI_Gatherv(my_candidate_ids.data(), my_candidate_ids.size(), MPI_UNSIGNED, 
//                        rank == 0 ? all_candidates.data() : nullptr, 
//                        all_counts.data(), displs_candidates.data(), MPI_UNSIGNED, 0, MPI_COMM_WORLD);
            
//             MPI_Gatherv(my_scores.data(), my_scores.size(), MPI_FLOAT, 
//                        rank == 0 ? all_scores.data() : nullptr, 
//                        all_counts.data(), displs_candidates.data(), MPI_FLOAT, 0, MPI_COMM_WORLD);

//             // 步骤8: 在rank 0上执行重排和精确计算
//             if (rank == 0) {
//                 // 选择top rerank个候选
//                 vector<size_t> indices(total_candidates);
//                 iota(indices.begin(), indices.end(), 0);
//                 size_t rerank_size = min(rerank, (size_t)total_candidates);
//                 partial_sort(indices.begin(), indices.begin() + rerank_size, indices.end(),
//                            [&](size_t a, size_t b) { return all_scores[a] > all_scores[b]; });

//                 // 精确距离计算
//                 priority_queue<pair<float, uint32_t>> max_heap;
//                 const size_t vecdim_aligned = vecdim & ~3;
//                 for (size_t j = 0; j < rerank_size; ++j) {
//                     uint32_t vec_id = all_candidates[indices[j]];
//                     float dist_sq = 0.0f;
//                     float32x4_t sum_vec = vdupq_n_f32(0.0f);

//                     for (size_t d = 0; d < vecdim_aligned; d += 4) {
//                         float32x4_t candidate_vec = vld1q_f32(base + vec_id * vecdim + d);
//                         float32x4_t query_vec = vld1q_f32(query + d);
//                         float32x4_t diff = vsubq_f32(candidate_vec, query_vec);
//                         sum_vec = vmlaq_f32(sum_vec, diff, diff);
//                     }
//                     dist_sq = vaddvq_f32(sum_vec);
//                     for (size_t d = vecdim_aligned; d < vecdim; ++d) {
//                         float diff = base[vec_id * vecdim + d] - query[d];
//                         dist_sq += diff * diff;
//                     }
//                     float dist = sqrtf(dist_sq);

//                     if (max_heap.size() < k) {
//                         max_heap.push(make_pair(dist, vec_id));
//                     } else if (dist < max_heap.top().first) {
//                         max_heap.pop();
//                         max_heap.push(make_pair(dist, vec_id));
//                     }
//                 }

//                 // 收集结果
//                 vector<uint32_t> result_ids;
//                 while (!max_heap.empty()) {
//                     result_ids.push_back(max_heap.top().second);
//                     max_heap.pop();
//                 }
//                 reverse(result_ids.begin(), result_ids.end());

//                 // 计算召回率
//                 set<uint32_t> gtset;
//                 for (size_t j = 0; j < k; ++j) {
//                     gtset.insert(static_cast<uint32_t>(test_gt[query_idx * test_gt_d + j]));
//                 }
//                 size_t correct = 0;
//                 for (uint32_t id : result_ids) {
//                     if (gtset.find(id) != gtset.end()) {
//                         correct++;
//                     }
//                 }
//                 double recall = static_cast<double>(correct) / k;
//                 recall_sum += recall;
//                 double end_time = MPI_Wtime();
//                 latency_sum += static_cast<long long>((end_time - start_time) * 1e6);
//             }
//         }

//         // 输出结果
//         if (rank == 0) {
//             double average_recall = recall_sum / total_queries;
//             long long average_latency = latency_sum / total_queries;
//             cout << "Rerank: " << rerank 
//                  << "\tRecall@" << k << ": " << average_recall
//                  << "\tLatency: " << average_latency << " us"
//                  << endl;
//         }
//     }

//     MPI_Barrier(MPI_COMM_WORLD);
//     double total_end_time = MPI_Wtime();
//     if (rank == 0) {
//         double total_time = total_end_time - total_start_time;
//         cout << "Total time for all queries: " << total_time << " seconds" << endl;
//         cout << "Average latency per query: " << total_time / MAX_QUERIES * 1e6 << " us" << endl;
//     }

//     delete[] test_query;
//     delete[] test_gt;
//     delete[] base;
//     MPI_Finalize();
//     return 0;
// }

// template float* LoadData<float>(const std::string&, size_t&, size_t&);
// template int* LoadData<int>(const std::string&, size_t&, size_t&);