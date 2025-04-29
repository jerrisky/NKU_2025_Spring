import numpy as np
from sklearn.cluster import KMeans
import struct
import time
import os

def load_data(data_path):
    """加载二进制向量数据"""
    with open(data_path, 'rb') as f:
        n = np.fromfile(f, dtype=np.int32, count=1)[0]
        d = np.fromfile(f, dtype=np.int32, count=1)[0]
        data = np.fromfile(f, dtype=np.float32).reshape(n, d)
    return n, d, data

def preprocess(data, M):
    """子空间划分与 K-means 聚类，生成码本"""
    D = data.shape[1]
    sub_dim = D // M
    codebooks = []
    for m in range(M):
        sub_data = data[:, m * sub_dim : (m + 1) * sub_dim]
        kmeans = KMeans(n_clusters=256, random_state=42, n_init=10).fit(sub_data)
        codebooks.append(kmeans.cluster_centers_)
    return codebooks, sub_dim

def save_codebooks(codebooks, filename):
    """保存码本到二进制文件"""
    # 创建 files 文件夹
    if not os.path.exists('files'):
        os.makedirs('files')
    full_filename = os.path.join('files', filename)
    M = len(codebooks)
    sub_dim = codebooks[0].shape[1]
    with open(full_filename, 'wb') as f:
        f.write(struct.pack('ii', M, sub_dim))  # 写入子空间数和子空间维度
        for sub_codebook in codebooks:
            for center in sub_codebook:
                f.write(center.astype(np.float32).tobytes())  # 写入每个簇中心

def save_cluster_products(codebooks, filename):
    """预计算并保存簇中心内积（上三角矩阵）"""
    # 创建 files 文件夹
    if not os.path.exists('files'):
        os.makedirs('files')
    full_filename = os.path.join('files', filename)
    M = len(codebooks)
    with open(full_filename, 'wb') as f:
        f.write(struct.pack('i', M))  # 写入子空间数
        for m in range(M):
            sub_codebook = codebooks[m]
            for i in range(256):
                for j in range(i, 256):
                    product = np.dot(sub_codebook[i], sub_codebook[j])
                    f.write(struct.pack('f', product))  # 写入内积值

def pq_quantize(vec, codebooks):
    """对单个向量进行 PQ 量化，返回各子空间的簇索引"""
    M = len(codebooks)
    sub_dim = codebooks[0].shape[1]
    codes = []
    for m in range(M):
        sub_vec = vec[m * sub_dim : (m + 1) * sub_dim]
        # 计算子向量与所有簇中心的内积，取最大值索引
        inner_products = np.dot(codebooks[m], sub_vec)
        code = np.argmax(inner_products)
        codes.append(np.uint8(code))
    return codes

def build_pq_index(data, codebooks):
    """批量生成所有向量的 PQ 编码（索引）"""
    M = len(codebooks)
    pq_codes = []
    start_time = time.time()
    for i, vec in enumerate(data):
        code = pq_quantize(vec, codebooks)
        pq_codes.append(code)
        if (i + 1) % 10000 == 0:
            print(f"已处理 {i+1} / {len(data)} 条向量，耗时 {time.time()-start_time:.2f} 秒")
    return np.array(pq_codes, dtype=np.uint8)

def save_pq_codes(pq_codes, filename):
    """保存 PQ 编码到二进制文件"""
    # 创建 files 文件夹
    if not os.path.exists('files'):
        os.makedirs('files')
    full_filename = os.path.join('files', filename)
    n, M = pq_codes.shape
    with open(full_filename, 'wb') as f:
        f.write(struct.pack('ii', n, M))  # 写入向量数和子空间数
        f.write(pq_codes.tobytes())  # 直接写入所有编码

# ====================== 主流程 ======================
if __name__ == "__main__":
    data_path = r"D:/PyCode/pythonProject7/anndata/DEEP100K.base.100k.fbin"
    M = 8  # 子空间数量，需与向量维度整除（96 ÷ 8 = 12）

    # 1. 加载数据
    base_number, vecdim, base_data = load_data(data_path)
    print(f"加载数据：{base_number} 条向量，维度 {vecdim}")

    # 2. 生成码本
    print(f"开始聚类，M={M} 个子空间...")
    codebooks, sub_dim = preprocess(base_data, M)
    print(f"码本生成完成，每个子空间 {sub_dim} 维，256 个簇中心")

    # 3. 保存码本和簇中心内积
    save_codebooks(codebooks, "codebooks.bin")
    save_cluster_products(codebooks, "cluster_products.bin")
    print("码本和预计算内积已保存")

    # 4. 生成 PQ 索引（PQ 编码）
    print("开始构建 PQ 索引...")
    pq_codes = build_pq_index(base_data, codebooks)
    print(f"PQ 索引构建完成，编码形状：{pq_codes.shape}（向量数 × 子空间数）")

    # 5. 保存 PQ 编码
    save_pq_codes(pq_codes, "pq_codes.bin")
    print("PQ 编码已保存到 files/pq_codes.bin")