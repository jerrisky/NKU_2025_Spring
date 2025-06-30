import numpy as np
import struct
import time
import os
from sklearn.cluster import KMeans
from collections import defaultdict
from typing import List, Tuple, Dict


class IVFIndex:
    def __init__(self, nlist: int = 1024, nprobe: int = 10):
        """初始化IVF索引"""
        self.nlist = nlist  # 聚类中心数量
        self.nprobe = nprobe  # 搜索时探查的聚类数
        self.dim = 0  # 向量维度
        self.centroids = None  # 聚类中心向量
        self.inverted_lists = None  # 倒排表: {聚类ID: [向量ID列表]}

    def load_data(self, data_path: str) -> np.ndarray:
        """加载二进制向量数据"""
        try:
            with open(data_path, 'rb') as f:
                n = np.fromfile(f, dtype=np.int32, count=1)[0]
                d = np.fromfile(f, dtype=np.int32, count=1)[0]
                data = np.fromfile(f, dtype=np.float32).reshape(n, d)
            self.dim = d
            print(f"成功加载数据: {n} 个向量，维度 {d}")
            return data
        except Exception as e:
            print(f"加载数据失败: {e}")
            raise

    def build(self, data: np.ndarray, n_init: int = 5, max_iter: int = 100) -> None:
        """构建IVF索引"""
        print(f"开始构建IVF索引 (nlist={self.nlist}, nprobe={self.nprobe})...")

        # 1. K-means聚类生成质心
        start_time = time.time()
        print("运行K-means聚类...")
        kmeans = KMeans(n_clusters=self.nlist, n_init=n_init, max_iter=max_iter, random_state=42)
        cluster_labels = kmeans.fit_predict(data)
        self.centroids = kmeans.cluster_centers_

        # 2. 构建倒排表（使用无符号32位整数存储向量ID）
        print("构建倒排表...")
        self.inverted_lists = defaultdict(list)
        for vector_id, label in enumerate(cluster_labels):
            # 关键修改：使用无符号32位整数存储向量ID，确保与C++兼容
            self.inverted_lists[label].append(np.uint32(vector_id))

        # 验证倒排表ID类型
        sample_cluster = next(iter(self.inverted_lists))
        if self.inverted_lists:
            sample_id = self.inverted_lists[sample_cluster][0]
            print(f"倒排表ID类型验证: {type(sample_id)}, 值: {sample_id}")

        print(f"IVF索引构建完成，耗时{time.time() - start_time:.2f}秒")
        print(f"质心形状: {self.centroids.shape}, 平均每个聚类{len(data) / self.nlist:.2f}个向量")

    def save(self, filename: str = "ivf.index") -> None:
        """保存IVF索引到二进制文件"""
        if not os.path.exists('files'):
            os.makedirs('files')
        full_filename = os.path.join('files', filename)

        try:
            with open(full_filename, 'wb') as f:
                # 写入基本信息
                f.write(struct.pack('ii', self.nlist, self.dim))  # nlist, 向量维度

                # 写入质心数据
                for centroid in self.centroids:
                    f.write(centroid.astype(np.float32).tobytes())

                # 写入倒排表（使用无符号32位整数存储向量ID）
                for centroid_id in range(self.nlist):
                    vector_ids = self.inverted_lists.get(centroid_id, [])
                    f.write(struct.pack('i', len(vector_ids)))  # 该聚类中的向量数量
                    # 关键修改：使用无符号32位整数写入向量ID
                    f.write(np.array(vector_ids, dtype=np.uint32).tobytes())

            print(f"IVF索引已保存至: {full_filename}")
        except Exception as e:
            print(f"保存索引失败: {e}")
            raise

    def load(self, filename: str = "ivf.index") -> None:
        """从二进制文件加载IVF索引"""
        full_filename = os.path.join('files', filename)

        try:
            with open(full_filename, 'rb') as f:
                # 读取基本信息
                self.nlist, self.dim = struct.unpack('ii', f.read(8))

                # 读取质心数据
                self.centroids = np.fromfile(f, dtype=np.float32, count=self.nlist * self.dim)
                self.centroids = self.centroids.reshape(self.nlist, self.dim)

                # 读取倒排表
                self.inverted_lists = defaultdict(list)
                for centroid_id in range(self.nlist):
                    list_size = struct.unpack('i', f.read(4))[0]
                    # 关键修改：使用无符号32位整数读取向量ID
                    vector_ids = np.fromfile(f, dtype=np.uint32, count=list_size)
                    self.inverted_lists[centroid_id] = vector_ids.tolist()

            print(f"IVF索引已从 {full_filename} 加载")

            # 验证加载的ID类型
            if self.inverted_lists:
                sample_cluster = next(iter(self.inverted_lists))
                if self.inverted_lists[sample_cluster]:
                    sample_id = self.inverted_lists[sample_cluster][0]
                    print(f"加载的倒排表ID类型验证: {type(sample_id)}, 值: {sample_id}")
        except Exception as e:
            print(f"加载索引失败: {e}")
            raise

    def search(self, query: np.ndarray, base_data: np.ndarray, k: int = 10) -> List[Tuple[int, float]]:
        """搜索最近邻向量"""
        # 1. 计算查询向量与所有质心的距离，找出最近的nprobe个聚类
        distances = np.linalg.norm(self.centroids - query, axis=1)
        nearest_clusters = np.argsort(distances)[:self.nprobe]

        # 2. 在这些聚类中搜索最近邻
        candidates = []
        for cluster_id in nearest_clusters:
            vector_ids = self.inverted_lists[cluster_id]
            for vec_id in vector_ids:
                # 计算向量与查询的L2距离
                dist = np.linalg.norm(base_data[vec_id] - query)
                candidates.append((int(vec_id), dist))  # 转换为Python整数用于输出

        # 3. 按距离排序，返回前k个
        candidates.sort(key=lambda x: x[1])
        return candidates[:k]


# ====================== 主流程 ======================
if __name__ == "__main__":
    # 示例数据路径（请根据实际情况修改）
    data_path = r"D:/PyCode/pythonProject7/anndata/DEEP100K.base.100k.fbin"
    query_path = r"D:/PyCode/pythonProject7/anndata/DEEP100K.query.fbin"

    # 参数配置
    nlist = 2048
    # 聚类中心数量
    nprobe = 10  # 搜索时探查的聚类数
    k = 10  # 返回的最近邻数量

    try:
        # 1. 创建并构建IVF索引
        ivf = IVFIndex(nlist=nlist, nprobe=nprobe)
        base_data = ivf.load_data(data_path)
        ivf.build(base_data)
        ivf.save()

        # 2. 加载查询向量并测试搜索
        query_data = ivf.load_data(query_path)
        print(f"加载查询向量: {len(query_data)} 条")

        # 测试单条查询
        query_idx = 0  # 测试第0条查询
        query = query_data[query_idx]

        start_time = time.time()
        results = ivf.search(query, base_data, k=k)
        search_time = time.time() - start_time

        print(f"查询 {query_idx} 耗时: {search_time * 1000:.2f} ms")
        print(f"最近邻结果 (ID, 距离):")
        for i, (vec_id, dist) in enumerate(results):
            print(f"Top-{i + 1}: ID={vec_id}, 距离={dist:.4f}")

    except Exception as e:
        print(f"执行过程中发生错误: {e}")