import numpy as np
import hnswlib
import struct
import os
import time
import tempfile


def load_fbin(path):
    """加载FBIN格式的向量数据"""
    print(f"Loading vectors from {path}...")

    with open(path, 'rb') as f:
        n, d = np.fromfile(f, dtype=np.int32, count=2)
        data = np.fromfile(f, dtype=np.float32).reshape(n, d)

    print(f"Loaded {n} vectors of dimension {d}")
    return data, d


def build_hnsw_index(data, dim, m=16, ef_construction=200):
    """构建HNSW索引"""
    print(f"Building HNSW index (M={m}, ef_construction={ef_construction})...")

    index = hnswlib.Index(space='l2', dim=dim)
    index.init_index(max_elements=data.shape[0], ef_construction=ef_construction, M=m)

    # 添加向量
    labels = np.arange(data.shape[0], dtype=np.uint32)
    index.add_items(data, labels)

    print(f"HNSW index built with {index.get_current_count()} vectors")
    return index


def manually_save_hnsw_index(index, path, dim, data, m):
    """完全手动保存HNSW索引，绕过hnswlib的save_index()"""
    print(f"Manually saving HNSW index to {path}...")

    # 创建目录（如果不存在）
    os.makedirs(os.path.dirname(path), exist_ok=True)

    # 获取索引参数
    max_elements = index.max_elements
    ef_construction = index.ef_construction
    enterpoint_node = index.get_enterpoint()
    max_level = index.get_max_level()

    # 获取所有节点的连接信息
    nodes = {}
    for label in range(max_elements):
        if index.contains(label):
            level = index.get_level(label)
            connections = []
            for l in range(level + 1):
                conns = index.get_connections(label, l)
                connections.append(conns)
            nodes[label] = {
                'level': level,
                'connections': connections
            }

    # 手动写入HNSW文件
    with open(path, 'wb') as f:
        # 写入文件头
        f.write(struct.pack('I', 0x1234))  # Magic number
        f.write(struct.pack('I', dim))  # Dimension
        f.write(struct.pack('I', max_elements))  # Max elements
        f.write(struct.pack('I', m))  # M
        f.write(struct.pack('I', ef_construction))  # ef_construction
        f.write(struct.pack('I', enterpoint_node))  # Enterpoint node
        f.write(struct.pack('I', max_level))  # Max level

        # 写入节点数量
        f.write(struct.pack('I', len(nodes)))

        # 写入每个节点
        for label, node_info in nodes.items():
            level = node_info['level']
            f.write(struct.pack('I', label))
            f.write(struct.pack('I', level))

            # 写入每个层级的连接
            for l in range(level + 1):
                conns = node_info['connections'][l]
                f.write(struct.pack('I', len(conns)))
                for conn in conns:
                    f.write(struct.pack('I', conn))

        # 写入向量数据
        for label in range(data.shape[0]):
            if label in nodes:  # 只写入存在的节点
                vector = data[label].astype(np.float32)
                f.write(vector.tobytes())

    print(f"Manual save complete. Verifying index...")

    # 验证手动保存的索引
    with open(path, 'rb') as f:
        magic = struct.unpack('I', f.read(4))[0]
        file_dim = struct.unpack('I', f.read(4))[0]

        if magic != 0x1234:
            raise ValueError(f"Invalid magic number: 0x{magic:08X} (expected 0x1234)")

        if file_dim != dim:
            raise ValueError(f"Dimension mismatch: {file_dim} (expected {dim})")

    print(f"HNSW index manually saved and verified at {path}")


def verify_hnsw_index(path, dim):
    """验证HNSW索引文件的有效性"""
    print(f"Verifying HNSW index file {path}...")

    if not os.path.exists(path):
        raise FileNotFoundError(f"Index file not found: {path}")

    with open(path, 'rb') as f:
        # 验证文件头
        magic = struct.unpack('I', f.read(4))[0]
        file_dim = struct.unpack('I', f.read(4))[0]

        if magic != 0x1234:
            raise ValueError(f"Invalid magic number: 0x{magic:08X} (expected 0x1234)")

        if file_dim != dim:
            raise ValueError(f"Dimension mismatch: {file_dim} (expected {dim})")

    # 尝试加载索引进行进一步验证
    try:
        test_index = hnswlib.Index(space='l2', dim=dim)
        test_index.load_index(path)
        print(f"Index loaded successfully with {test_index.get_current_count()} vectors")
    except Exception as e:
        raise ValueError(f"Failed to load index: {str(e)}") from e

    print(f"HNSW index file verified successfully")


def main():
    # 配置参数
    DATA_PATH = r"D:/PyCode/pythonProject7/anndata/DEEP100K.base.100k.fbin"
    INDEX_PATH = "files/hnsw.index"
    HNSW_M = 16
    HNSW_EF_CONSTRUCTION = 200

    try:
        print("==== Starting HNSW Index Construction ====")

        # 1. 加载数据
        data, dim = load_fbin(DATA_PATH)

        # 2. 构建HNSW索引
        index = build_hnsw_index(
            data=data,
            dim=dim,
            m=HNSW_M,
            ef_construction=HNSW_EF_CONSTRUCTION
        )

        # 3. 手动保存索引
        manually_save_hnsw_index(index, INDEX_PATH, dim, data, HNSW_M)

        # 4. 验证索引
        verify_hnsw_index(INDEX_PATH, dim)

        print("\n==== HNSW Index Successfully Built ====")
        print(f"Index file: {INDEX_PATH}")
        print(f"Parameters: M={HNSW_M}, ef_construction={HNSW_EF_CONSTRUCTION}, dim={dim}")

    except Exception as e:
        print(f"Error: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()