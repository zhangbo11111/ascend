
# 三、TorchRec 与 DynamicEmb（学习笔记）

> 搞懂"超大 embedding 表怎么在分布式环境里存取"，这是读 GR 适配代码的地基。

## 1.定位：谁管什么

TorchRec = PyTorch 生态里的超大 embedding 表管理系统（Meta 开源，2019），目标是让推荐模型能在多卡/多机上训练。

| 管                                   | 不管                             |
| ------------------------------------ | -------------------------------- |
| 表的声明（有哪些表、多大）           | 模型结构（HSTU）                 |
| 表的切分（哪张表怎么切、切到哪张卡） | 训练循环与优化器（Megatron）     |
| 切分后查表的通信编排（all-to-all）   | 表的内容与生命周期（DynamicEmb） |

**两个正交维度，可叠加使用**：

* **TorchRec 管「表放在哪里」**——空间维度：切成几份、放在哪几张卡
* **DynamicEmb 管「表里放什么」**——时间维度：谁进来、谁被淘汰、谁下沉

所以一张表可以既"被按行切成 8 份"，又"每份内部按 LRU 淘汰"。

## 2.五个角色

| 角色                | 是什么             | 干什么                             | 本项目对应                                     |
| ------------------- | ------------------ | ---------------------------------- | ---------------------------------------------- |
| `EmbeddingConfig` | 表的"身份证"       | 声明表名、行数、维度、dtype        | `HASH_SIZE`、`item_vocab_size_or_capacity` |
| `EBC` / `EC`    | 表的**容器** | 声明"用哪几张表、喂哪些特征"       | `EmbeddingCollection`（EC）                  |
| Planner             | **决策者**   | 算"哪张表用哪种切法、放哪些卡"     | `get_planner`                                |
| Sharder             | **执行者**   | 真正把表切片、分配到各卡           | `DynamicEmbeddingCollectionSharder`          |
| DMP                 | **装配者**   | 把 EBC/EC 换成分片版，接管 forward | `apply_dmp`                                  |

## 3.EBC 与 EC

都是 `torchrec` 的 `nn.Module`，职责都是"持有若干张表 + 实现前向查表"。

**唯一区别：查完表之后，要不要把同一特征的多个向量合并掉。**

* **EBC**（`EmbeddingBagCollection`）：**合并**（池化 sum/mean/max），输入 N 个 id 出来 1 个向量
* **EC**（`EmbeddingCollection`）：**不合并**，输入 N 个 id 出来 N 个向量，序列结构保留

```python
# 1) 声明表（两者配置类不同）
from torchrec.modules.embedding_configs import EmbeddingBagConfig, EmbeddingConfig

bag_cfg = EmbeddingBagConfig(name="user_id", num_embeddings=1_000_000,
                             embedding_dim=64, feature_names=["user_id"])
seq_cfg = EmbeddingConfig(name="movie_id", num_embeddings=1_000_000,
                          embedding_dim=64, feature_names=["movie_id"])

# 2) 装成容器
from torchrec.modules.embedding_modules import EmbeddingBagCollection, EmbeddingCollection
ebc = EmbeddingBagCollection(tables=[bag_cfg], device=torch.device("npu"))
ec  = EmbeddingCollection(tables=[seq_cfg],   device=torch.device("npu"))

# 3) 造输入：2 个样本各 3 个 id
from torchrec.sparse.jagged_tensor import KeyedJaggedTensor
kjt = KeyedJaggedTensor.from_lengths_sync(
    keys=["movie_id"],
    values=torch.tensor([10, 20, 30,  40, 50, 60]),
    lengths=torch.tensor([3, 3]),
)

# 4) 输出差异（全部关键）
out_ec = ec(kjt)      # Dict[str, JaggedTensor]
#   values() 形状 [6, 64]   ← 6 个 id → 6 个向量，一个不少
#   offsets() = [0, 3, 6]   ← 边界保留

out_ebc = ebc(kjt)    # KeyedTensor
#   values() 形状 [2, 64]   ← 6 个向量被池化成 2 个，序列信息丢了
```

**结论：HSTU 必须用 EC**——它要对"每一个历史 item"做 attention，用 EBC 池化后序列就塌了。

本项目真实代码（`modules/embedding.py` 的 `ShardedEmbedding.forward`）：

```python
def forward(self, kjt: KeyedJaggedTensor) -> Dict[str, JaggedTensor]:
    mp_embeddings_awaitables = self._model_parallel_embedding_collection(kjt)
    if self._data_parallel_embedding_collection is not None:
        with torch_npu.npu.stream(self._side_stream):
            dp_embeddings = self._data_parallel_embedding_collection(kjt)
        torch_npu.npu.current_stream().wait_stream(self._side_stream)
        embeddings = {**mp_embeddings_awaitables.wait(), **dp_embeddings}
    else:
        embeddings = mp_embeddings_awaitables.wait()
    return embeddings
```

三个观察：

1. **返回 `Dict[str, JaggedTensor]`** → 证明用的是 EC 而非 EBC；这正是"稀疏/稠密分界线"：KJT（稀疏侧）进，JaggedTensor（稠密侧）出
2. **两个 collection 并存**：`_model_parallel_embedding_collection`（要跨卡查的表）与 `_data_parallel_embedding_collection`（每卡一份副本、不跨卡的表）→ "同一模型混用分片策略"落到代码了
3. **data-parallel 走独立 stream**：它不需要等 all-to-all，可丢到旁路 stream 并发执行，最后 `wait_stream` 汇合，白捡的并行度

## 4.DMP

`DistributedModelParallel`：把已建好的普通模型**就地改造成分片版**，并在 forward 时自动完成跨卡查询。

做三件事：

1. **识别**稀疏模块（靠 `TORCHREC_TYPES = {EmbeddingBagCollection, EmbeddingCollection}`）
2. **替换**成分片版（用 sharders + plan，表的切片真正发生在这里）
3. **接管** forward（内部自动发起 all-to-all）

本项目真实用法（`distributed/sharding.py` 的 `apply_dmp`）：

```python
model = DistributedModelParallel(
    module=model,               # 原始模型，内部还是完整的 EC
    device=device,              # torch.device("npu", 0)
    sharders=sharders,          # [DynamicEmbeddingCollectionSharder(...)]
    plan=plan,                  # get_planner 产出的分片方案
    init_data_parallel=True,
)
```

**关键是调用方式完全不变**：

```python
logits = model(batch)              # 分片前 / 分片后写法一样
raw = model._dmp_wrapped_module    # 需要时能取出原始模型
```

不用 DMP 就得自己写这些（这就是 DMP 替你干的活）：

```python
owner = ids_to_rank_range(ids, world_size)      # 1. 算归属
recv_ids = all_to_all(buckets(ids, owner), pg)  # 2. 分桶 + 去程 all-to-all
vecs = my_local_table[recv_ids]                 # 3. 查本地表
back = all_to_all(vecs, pg, reverse=True)       # 4. 回程 all-to-all
result = unbucketize(back)                      # 5. 重排
```

（DMP 还要额外处理缓冲区复用、prefetch 流水线、前后向通信顺序、数据并行表的分流。）

## 5.完整流程：单卡模型 → 分布式模型

```
Step 1  声明表         EmbeddingConfig(name="movie_id", num_embeddings=HASH_SIZE, embedding_dim=64)
Step 2  组成单卡模型    ec = EmbeddingCollection(tables=[...]); model = MyModel(ec, hstu_block, head)
Step 3  Planner 出方案  plan = planner.collective_plan(model, sharders, pg)   ← 只产方案，不动机器
Step 4  Sharder 执行    按 plan 把每张表切成 N 份，各卡拿到自己那份
Step 5  DMP 装配       model = DistributedModelParallel(module=model, sharders=..., plan=...)
                       内部把 ec 换成 ShardedEmbeddingCollection，forward 被接管
```

**Step 5 之后使用方式完全不变**：`model(batch)` 照旧，跨卡查表自动发生。这也解释了为什么适配时 `apply_dmp` 只改了几行。

## 6.四种分片策略

| 策略              | 怎么切               | 每卡存多少                         | 查表通信         | 用在哪                       | 代价                                      |
| ----------------- | -------------------- | ---------------------------------- | ---------------- | ---------------------------- | ----------------------------------------- |
| `TABLE_WISE`    | 按**表**切     | 整张表放某一张卡                   | 要（all-to-all） | 表**多**但每张不大     | 通信；负载可能不均                        |
| `ROW_WISE`      | 按**行**切     | `num_embeddings / world_size` 行 | 要（all-to-all） | 单表**极大**（最常用） | 通信 + 需重排结果                         |
| `COLUMN_WISE`   | 按**列**切     | `embedding_dim / world_size` 列  | 要且更复杂       | 行数不多、**维度极大** | 查完要 all-gather 拼维度，池化还要 reduce |
| `DATA_PARALLEL` | 不切，**复制** | 完整一份副本                       | **不需要** | 表**很小**             | 费显存（N 倍），但零通信                  |

要点：

* **优先选零通信的**：小表复制 N 份也就几 KB/MB，换查表零通信，很划算
* **同一模型可混用**：本项目就是（11 行的表走 `DATA_PARALLEL`，百万行的表走 `ROW_WISE`）
* `COLUMN_WISE` 少用：通信步骤多（all-gather 拼维度 + 可能的 reduce），延迟高
* 进阶：**表融合**——多张小 row-wise 表打包成一张大表统一切，一次 all-to-all 处理所有表。本项目未用到

## 7.all-to-all 完整机制（7 步）

设 `world_size=4`，按行切，本 batch 要查 **100 个 id**：

```
Step 1  索引换算        batch 的 id → 行号 → 归属哪张卡
Step 2  分桶            按归属卡分成 4 个桶（卡0:30、卡1:25、卡2:20、卡3:25）
Step 3  去程 all-to-all 每卡把"桶i"发给卡i，同时收下归自己的 id
Step 4  本地查表        各卡查自己存的那段行
Step 5  回程 all-to-all 把结果发回原来要它的卡
Step 6  重排            还原成 batch 原始顺序
Step 7  输出            100 个向量，与输入一一对应
```

**为什么便宜**：通信量 ∝ **本 batch 命中的 id 数**（100），不是表的总大小（几亿行）。

|          | 稀疏侧（查表）       | 稠密侧（数据并行）                |
| -------- | -------------------- | --------------------------------- |
| 通信原语 | **all-to-all** | **all-reduce / all-gather** |
| 通信量   | ∝ 命中的 id 数      | ∝**全部参数量 / 数据量**   |

**这就是"稀疏/稠密必须两套并行策略"的根本原因——通信模式完全不同。**

**prefetch（流水线）**：把这一批拆成若干小批，第 1 批在算的时候第 2 批在传，让通信和计算重叠。

## 8.planner 如何决策

四步：前三步"枚举 + 打分"，第四步才是决策。

### Step 0 约束先过滤候选（本项目里这步几乎锁死了决策权）

`get_planner` 按表名分三类，注意 `sharding_types=[...]` **只有一个元素**——不是建议，是候选集合只有一项：

| 表的分类                                     | 被允许的候选                                                               |
| -------------------------------------------- | -------------------------------------------------------------------------- |
| 在`data_parallel_embedding_table_names` 里 | 只有`DATA_PARALLEL` + `use_dynamicemb=False`                           |
| 在`dynamicemb_options_dict` 里             | 只有`ROW_WISE` + 动态表配置                                              |
| 其他                                         | 只有`ROW_WISE` + `use_dynamicemb=False`（save/load 不支持 table-wise） |

**洞察：本项目里 planner 已退化成"执行器"**，决策权被人工约束拿走——说明自动决策在工程上不够可靠，生产上常人工钉死。

### Step 1 枚举候选

每个候选 = `(切法 sharding_type) × (查表内核 compute_kernel) × (分配到哪些 rank)`

干这事的是 **enumerator**：

```python
enumerator = DynamicEmbeddingEnumerator(topology=topology, constraints=constraints)
```

### Step 2 打分：两项成本

**A. storage（内存）**

```
原表大小 = num_embeddings × embedding_dim × dtype_size
row_wise      → 每卡 = 原表 / world_size
data_parallel → 每卡 = 原表（每卡一份完整的）
```

**B. perf（通信）**——与硬件拓扑强相关，代码里带着带宽参数：

```python
ddr_cap: int = 512 * 1024 * 1024 * 1024,  # 假设一个节点 512GB 内存
intra_host_bw: int = 450e9,               # 节点内 450 GB/s（NVLink）
inter_host_bw: int = 25e9,                # 节点间 25 GB/s（网卡）
```

节点内 450 GB/s vs 节点间 25 GB/s，**差 18 倍**。同样传 1 GB：节点内约 2.2ms，节点间要 40ms。所以 planner 会尽量把"频繁查表的大表"放在同一节点内，并评估表下沉主机内存的容量上限（`ddr_cap`）。

### Step 3 全局选择（真正的决策）

> 给每张表挑一个候选 + 决定放哪些卡，使**每张卡内存 ≤ HBM 上限**，且**峰值通信代价最小**。

这是**装箱问题**（NP-hard），TorchRec 用**动态规划 / 贪心**求近似解：逐表决策、记录各卡占用、内存不超限是硬约束、在可行解里挑通信最省的。产出 `plan`（哪张表 / 哪种切法 / 放哪些卡），再喂给 sharder 和 DMP。

### 经验规则：什么样的表用什么切法

| 表的特征                          | 建议切法                       | 账怎么算                               |
| --------------------------------- | ------------------------------ | -------------------------------------- |
| **小表**（复制 N 份放得下） | `DATA_PARALLEL`              | 复制的代价可忽略，换零通信             |
| **行数多、维度正常**        | `ROW_WISE`                   | 内存摊薄 N 倍；通信量只 ∝ 命中 id 数  |
| 行数不多、**维度极大**      | `COLUMN_WISE`                | 按行切每卡仍很大，按列切才摊得开       |
| 表多、每张都不大                  | `TABLE_WISE`（或融合后切行） | 一张表放一张卡，避免每张都切           |
| 持续增长、极稀疏                  | `ROW_WISE` + 动态表          | 静态表装不下新 id（DynamicEmb 的领域） |

### 用本项目真实数字走一遍（假设 dim=64、fp32）

**`movie_id` 表**：`num_embeddings = HASH_SIZE`，8 卡

```
1000 万行时：单表 = 1000万 × 64 × 4 = 2.56 GB
   data_parallel → 每卡 2.56 GB，8 卡共 20 GB   ← 爆显存
   row_wise      → 每卡 320 MB
100 万行时：单表 = 100万 × 64 × 4 = 256 MB
   row_wise      → 每卡 32 MB
→ 选 ROW_WISE：内存摊薄 8 倍，代价是稀疏通信
```

**`action_weights` 表**：`num_embeddings = 11`（电影评分只有 11 种）

```
单表 = 11 × 64 × 4 = 2816 字节 ≈ 2.8 KB
   data_parallel → 每卡 2.8 KB，零通信         ← 完胜
   row_wise      → 每卡 0.35 KB，但每次查表走一次 all-to-all
→ 为省 2.5 KB 花一次通信，不划算
```

**结论：不是看内存绝对值，而是看"复制一份的代价" vs "通信一次的代价"哪个更小。**

## 9.KJT 与 JaggedTensor

**问题**：一个 batch 里 32 个用户，A 看了 3 个 item、B 看了 100 个、C 看了 7 个，怎么塞进一个张量？

* 笨办法：补齐到 100（padding）——费显存、白算、还要记 mask
* 聪明办法：把所有人的行为**首尾相接**排成一条长序列，再记每个人的**边界**

**Jagged Tensor（锯齿张量）= values + offsets**：

```
values  = [v1, v2, v3, v4, v5, ..., v110]     ← 所有人行为首尾相接
offsets = [0,   3,   103,  110]               ← 每个人的起点

用户A = values[0:3]、用户B = values[3:103]、用户C = values[103:110]
```

**KJT = Keyed Jagged Tensor = 带特征名的 Jagged Tensor**。因为模型一次不止一条序列（item 序列 / action 序列 / contextual 特征），要多带两层信息：

```
keys    = ["item", "action", "contextual"]   ← 特征名
values  = [ 所有序列的值拼在一起 ]
lengths = [ 每个特征、每个样本各有多长 ]     ← 一维展平
offsets = [ lengths 累加出来的起点 ]
```

**小例子（batch = 2，3 个特征）**：

| 特征       | 用户0 的 id 序列 | 用户1 的 id 序列 |
| ---------- | ---------------- | ---------------- |
| item       | `[11, 25]`     | `[7]`          |
| action     | `[3, 3]`       | `[1, 5, 9]`    |
| contextual | `[99]`         | `[42]`         |

拼成的 KJT（注意**先按特征分组，特征内再按 batch 排**）：

```python
keys    = ["item", "action", "contextual"]

values  = [ 11, 25,  7,  |  3, 3,  1, 5, 9,  |  99, 42 ]
#           └─ item ─┘     └──── action ────┘    └ ctx ┘

lengths = [ 2, 1,  |  2, 3,  |  1, 1 ]
#          item0 item1  act0 act1  ctx0 ctx1

offsets = [0, 2, 3, 5, 8, 9, 10]   # lengths 前缀和，多一个哨兵位
```

取用时的对应关系：

```python
batch = 2   # 特征 k、样本 b 的区间 = values[offsets[b + k*batch] : offsets[b + 1 + k*batch]]

# item(第0个特征) 用户1(b=1)：values[offsets[1]:offsets[2]] = values[2:3] = [7]
# action(第1个特征) 用户0(b=0)：values[offsets[2]:offsets[3]] = values[3:5] = [3, 3]
# action(第1个特征) 用户1(b=1)：values[offsets[3]:offsets[4]] = values[5:8] = [1, 5, 9]
```

这就是为什么要 `keys + lengths + offsets` 三件套：**keys 定位是哪个特征，lengths 定位每个样本多长，offsets 定位在 values 里的绝对区间**。

**为什么 offsets 是核心**：切片、拼接、转稠密（padded dense）全靠它划边界。这就是为什么 patch 里要专门新建 `pt_jagged_tensors.py`（246 行，单文件最长）。

|              | **KJT**        | **JaggedTensor**           |
| ------------ | -------------------- | -------------------------------- |
| 装什么       | **id**（整数） | **embedding 向量**（浮点） |
| 谁产生       | 数据加载 / 预处理    | 查完表之后                       |
| 属于哪个世界 | **稀疏侧**     | **稠密侧**                 |

```
KJT ──[查 embedding 表]──> JaggedTensor
```

## 10.本项目真实配置

`training/utils.py` 的 `get_dataset_and_embedding_args()`（ml-20m 分支）：

| 表名               | 特征         | 容量          | 分片类型          | 改动                                                                                      |
| ------------------ | ------------ | ------------- | ----------------- | ----------------------------------------------------------------------------------------- |
| `action_weights` | `rating`   | **11**  | `data_parallel` | `EmbeddingArgs` → `DynamicEmbeddingArgs`；新增 `item_vocab_gpu_capacity_ratio=0.5` |
| `movie_id`       | `movie_id` | `HASH_SIZE` | 未显式指定        | `caching` True→**False**                                                         |
| `user_id`        | `user_id`  | `HASH_SIZE` | 未显式指定        | `caching` True→**False**                                                         |

四条观察：

1. `action_weights` 完美印证第 8 节规律——`rating` 只有 11 种取值，表小到可忽略，走 `data_parallel` 零通信
2. `movie_id` / `user_id` 是真正的动态大表，走 `ROW_WISE` + DynamicEmb 淘汰；**未指定 sharding_type** → 交给 planner 决策
3. `caching` 被强制关闭，注释写明"NPU 暂不支持 caching" → **能力边界，不是性能选择**
4. `HASH_SIZE` 从 1000 万改到 100 万，缩小 10 倍 → 更容易触发淘汰/哈希冲突，**这类改动只靠精度对比才能验证**

动态表参数（`create_dynamic_optitons_dict`）：

```python
global_hbm_for_values=embedding_args.global_hbm_for_values,
evict_strategy=DynamicEmbEvictStrategy.LRU if embedding_args.evict_strategy == "lru" ...,
safe_check_mode=DynamicEmbCheckMode.IGNORE,
bucket_capacity=128,
training=training,
caching=False,   # NPU 暂不支持 caching
```

**本项目已知的 NPU 能力边界（都写在代码注释里）**：

* `caching=False` —— 暂不支持 caching
* `pipeline_type="none"` —— 暂不支持流水线（上游是 `"prefetch"`）
* 只保留 `DynamicEmbeddingCollectionSharder` —— 不支持 BagCollection 那条路
* table-wise 被禁用 —— save/load 不支持 table-wise 分片
* 只有 `DynamicEmbeddingArgs` —— NPU 暂仅支持这一类表参数

## 11.DynamicEmb 深挖

> 第 1–10 节讲的是 TorchRec 怎么把表**切开放**；这一节讲 DynamicEmb 怎么决定表里**放什么**。

### 11.1 一句话定位

> **TorchRec 解决"一张表装不下单卡显存"，DynamicEmb 解决"一堆表加起来装不下所有卡显存"。**

前者是**空间**问题（切开、分散），后者是**时间**问题（谁进表、谁被挤出去）。两者正交，可叠加。

### 11.2 静态表的三个死结

TorchRec 的表是 `num_embeddings × dim` 一块固定内存，全程常驻 HBM，有三个致命伤：

| 死结                       | 说明                                                                                                                    |
| -------------------------- | ----------------------------------------------------------------------------------------------------------------------- |
| **容量是"拍"出来的** | `HASH_SIZE` 得预估：估小了哈希冲突、新 id 插不进去；估大了白占显存。第 10 节那次 `10_000_000 → 10_000_00` 就是典型 |
| **幂律分布被浪费**   | 90% 的请求打在前 1% 的热 id 上，冷 id 占 99% 空间却几乎不被查                                                           |
| **多表叠加爆炸**     | 单表能切，但 user_id + movie_id + item_vocab + … 加起来超了，切开也没用                                                |

DynamicEmb 的解法：**表不再是"一块连续显存"，而是一个可增删的哈希表**——热留在 HBM、冷下沉到 DDR、太冷的直接删。

### 11.3 表结构：一张表 = 一堆桶

理解一切的起点。底层是 HKV 的哈希表，**不是** `nn.Embedding` 那种 `[num_embeddings, dim]` 矩阵：

```
表 = 哈希桶数组
     ┌──────────┬──────────┬──────────┬─────
     │  bucket  │  bucket  │  bucket  │ ...
     │ (128 槽) │ (128 槽) │ (128 槽) │
     └──────────┴──────────┴──────────┴─────
      每个槽 = { key, emb[dim], optstate, score }
```

* **`bucket_capacity=128`**：每桶 128 个槽位（本项目值），查表以"桶"为单位找 key
* **key 是哈希值**，不是原始 id → 任意 id 都能进表，只要哈希不撞
* **槽是懒分配的**：桶初始为空，容量随插入增长（`init_capacity → max_capacity`）

**为什么按桶组织**：NPU 上要并行访存，同桶的槽放一起才能一次读入批量比对 key（HKV 用 digest 做乐观并发）。

### 11.4 每个槽里的三样东西

| 数组         | 内容             | 作用                                       |
| ------------ | ---------------- | ------------------------------------------ |
| `emb`      | `dim` 个浮点数 | 查表结果，即嵌入向量                       |
| `optstate` | Adam 的 m / v 等 | **优化器状态也存在表里**，跟着槽走   |
| `score`    | 一个整数         | 淘汰打分依据（LRU 记访问时间，LFU 记频次） |

`optstate` 在表里这点很关键：**Adam 的动量必须和嵌入同生共死**。向量被淘汰了状态还在，下次 id 回来会带着陈旧动量训练，精度会飘。所以 `DynamicEmbDump/Load` 都带 `optim=True`。

`score` 是淘汰算法的输入——**没有 score 就没法淘汰**，这是 DynamicEmb 比静态表多出来的那部分开销。

### 11.5 分层存储：`global_hbm_for_values`

```
      HBM  ┌─────────────────────────┐
   快但小  │  key index + 热 value   │  ← global_hbm_for_values 划这条线
           ├─────────────────────────┤
     DDR  │       温 value          │
   慢但大  ├─────────────────────────┤
     磁盘  │       冷 value          │
           └─────────────────────────┘
```

* **key index 常驻 HBM** —— 查找路径要快，key 不能下沉
* **value 的 HBM 预算由 `global_hbm_for_values` 控制**，超出部分下沉
* 本项目用 `item_vocab_gpu_capacity_ratio=0.5` 表达：**表容量的 50% 放 HBM**，剩下一半留给 DDR

对照第 8 节：`data_parallel` vs `row_wise` 是**跨卡**取舍，这里是**跨存储层级**取舍，又多了一个维度。

### 11.6 淘汰：表满之后怎么办

表有 `max_capacity`。满了要插新 key，就得先踢一个旧的，策略由 `evict_strategy` 定：

| 策略          | 依据           | 适合                           |
| ------------- | -------------- | ------------------------------ |
| **LRU** | 最久没被访问的 | 兴趣随时间漂移（本项目的选择） |
| **LFU** | 访问次数最少的 | 长尾稳定、热点固定             |

淘汰粒度是**槽**，腾出来的槽给新 key 用。**这就是"动态"的含义**：表的大小和内容都在变。

⚠️ 淘汰是**有损的**。被淘汰的 id 下次再出现，嵌入从初始值重来（本项目用 `DynamicEmbInitializerMode.UNIFORM`）。所以：

* 淘汰太频繁 → 模型反复"遗忘"，**精度劣化但不报错**
* 这正是第 10 节"HASH_SIZE 缩小 10 倍**只能靠精度对比验证**"的根因

`safe_check_mode=DynamicEmbCheckMode.IGNORE` 意味着**不因为插不进去而报错**——静默容忍，性能优先。这也解释了为什么容量改错不会崩。

### 11.7 一次 forward 的完整路径

输入仍是 TorchRec 的 KJT（这就是"KJT 在稀疏侧"的意义），但查表被换掉了：

```
KJT(ids)
  │
  ├─ 1. unique 去重          ← use_index_dedup=True
  │      batch 里重复 id 只查一次
  ├─ 2. bucketize            按哈希分到各桶
  ├─ 3. find + insert missing
  │      命中的直接取；没命中的新 id 现场建槽
  │      表满 → evict 一个旧槽
  ├─ 4. gather emb           从 emb 数组取向量
  └─ 5. expand 回原顺序       去重的逆操作
       │
       ▼
   Dict[str, JaggedTensor]    ← 稠密侧，序列结构保留（EC 不池化）
```

反向是 `dynamic_emb_update`：按槽**原地**更新 `emb` 和 `optstate`，同时刷 `score`——表内原地更新，不走传统那套全量梯度同步。

### 11.8 与 TorchRec 的四个接口

DynamicEmb 不重造轮子，**继承 TorchRec 的抽象、只换实现**。即约束/枚举器/规划器/执行器等。`torch_rec_v2_examples/little_demo/model.py` 最清楚：

| TorchRec 抽象                  | DynamicEmb 版本                       | 换成什么了                                                    |
| ------------------------------ | ------------------------------------- | ------------------------------------------------------------- |
| `ParameterConstraints`       | `DynamicEmbParameterConstraints`    | 多了`dynamicemb_options`；**约束必须含 `ROW_WISE`** |
| `EmbeddingEnumerator`        | `DynamicEmbeddingEnumerator`        | 候选枚举只留动态表能走的分片                                  |
| `EmbeddingShardingPlanner`   | `DynamicEmbeddingShardingPlanner`   | 打分时把动态表当不可切                                        |
| `EmbeddingCollectionSharder` | `DynamicEmbeddingCollectionSharder` | 建表时创建哈希表而非 Tensor                                   |

```python
DynamicEmbParameterConstraints(
    sharding_types=[ShardingType.ROW_WISE.value],   # 动态表只能按行切
    compute_kernels=["fused"],                       # 用融合算子
    dynamicemb_options=DynamicEmbTableOptions(
        initializer_args=DynamicEmbInitializerArgs(mode=...),  # 新 id 怎么初始化
        training=training,                                     # 决定要不要 optstate
    ),
)
```

另有 `use_index_dedup=True` —— 决定 11.7 第 1 步要不要去重，省的是 gather 的量。

**容器仍是原生 `EmbeddingCollection`**（`EmbeddingCollection(tables=eb_configs)`），只有 Sharder 是动态的——印证第 9 节"KJT 进、JaggedTensor 出"的边界没变。

### 11.9 训练 / 推理 / Checkpoint

`DynamicEmbTableOptions.training` 是个**语义开关**：

|                  | `training=True` | `training=False` |
| ---------------- | ----------------- | ------------------ |
| 分配`optstate` | 是                | 否                 |
| 支持反向更新     | 是                | 否（只读）         |
| 显存占用         | 高（约 3×）      | 低（只存 emb）     |

**Checkpoint 是 DynamicEmb 最容易踩的坑**（`little_demo/main.py` 的注释）：

```python
# ShardedDynamicEmbeddingCollection.state_dict() will return a dummy tensor.
torch.save({...})                                          # ← 存不到嵌入！只是占位
DynamicEmbDump(save_dir/"dynamicemb", model, optim=True)   # ← 嵌入从这条路径走
```

|                                    | 存什么                                             | 怎么存               |
| ---------------------------------- | -------------------------------------------------- | -------------------- |
| `torch.save(model.state_dict())` | 稠密层权重                                         | 常规                 |
| `DynamicEmbDump`                 | **动态表的全部槽**（emb + optstate + score） | 每 rank 存自己那部分 |

加载必须 `DynamicEmbLoad(..., optim=True)`，且 `model.load_state_dict(..., strict=False)`（因为 state_dict 里没有嵌入）。**忘记 Dump 就会静默丢失整个嵌入表**——`strict=False` 不报错，直接从头训。

### 11.10 本项目（RecSDK v2）的适配点

Python 层沿用，**C++ 算子层全部用 Ascend C 重写**：

| 改动                                                         | 位置                             | 性质                                           |
| ------------------------------------------------------------ | -------------------------------- | ---------------------------------------------- |
| 导入路径`dynamicemb.*` → `dynamic_emb.*`                | `sharding.py` / `utils.py`   | 工程迁移                                       |
| 新增`initializer_args`（UNIFORM 显式指定）                 | `create_dynamic_optitons_dict` | **精度对齐**：默认初始化不确定，必须钉死 |
| `caching` True → **False**                          | 同上                             | **能力边界**，NPU 未实现                 |
| `DynamicEmbeddingArgs` + `item_vocab_gpu_capacity_ratio` | `utils.py`                     | 新增分层容量语义                               |
| `dynamic_emb_extensions`：17 个基于 HKV 的 Ascend C 算子   | 独立算子库                       | **重写**，不走 CUDA                      |

**全下沉的脉络**：v1 里哈希 / 去重 / 分桶在 CPU 上（`torchrec_embcache` / `hybrid_torchrec`），v2 把 11.7 那一整条链路搬到 NPU。这也是 v2 只能跑 A5 的根本原因——**不是软件挑硬件，是算子按 A5 编的**。

## 12.记忆锚点

1. **TorchRec 管"表放哪里"（空间），DynamicEmb 管"表里放什么"（时间）**——正交，可叠加
2. 五个角色：Config 声明 → Planner 决策 → Sharder 执行 → DMP 装配；EBC/EC 是容器
3. EBC 会池化掉序列，**HSTU 必须用 EC**
4. 四种切法：**小表复制（零通信）、大表按行切（all-to-all）、多表按表切、大维度按列切**
5. all-to-all = **分桶 → 去程 → 本地查 → 回程 → 重排**，通信量只跟"命中的 id 数"成正比
6. planner = **枚举候选 + 打分（内存 + 通信÷带宽）+ 装箱求解**；本项目因约束把候选钉死，planner 退化为执行器
7. 变长序列 = **values + offsets**；KJT 是"带特征名的变长稀疏张量"，在稀疏侧
8. 动态表 = **哈希桶**，桶里 128 槽，槽 = `{key, emb, optstate, score}`
9. **optstate 跟着槽走** → 淘汰即丢状态，`Dump/Load` 必须 `optim=True`
10. `global_hbm_for_values` 划 **HBM/DDR 分界线**，`evict_strategy` 决定**谁出局**
11. 动态表查表 = **去重 → 分桶 → 查+补 → gather → 展开**，约束必须是 `ROW_WISE`
12. **淘汰是有损的、静默的**——容量类改动只能靠精度对比验证

# 四、GR 适配源码精读（模块二 · 第 3 课：`pt_ops` 变长算子重写）

> 位置：`examples/hstu/ops/pt_ops/`
> 一句话：NPU 不支持 Triton / CUTLASS，把「变长序列」相关的算子用**纯 PyTorch** 重写一遍接进主流程。

## 1. 地基：什么是变长张量（jagged tensor）

要读懂这一课，先把"变长"这件事本身搞清楚。

**推荐系统的数据天生是变长的**：一个用户看过的商品可能是 3 个，另一个可能是 200 个。序列长度是行为决定的，不可能整齐划一。

于是有两种表示办法：

| 表示                   | 形状                                       | 说明                                                                                       |
| ---------------------- | ------------------------------------------ | ------------------------------------------------------------------------------------------ |
| 规整（dense / padded） | `[B, L, D]`                              | 每条序列都 pad（补 0）到统一的`L`。浪费显存，还要额外记一个 mask 来区分"真数据 / 补的 0" |
| 变长（jagged）         | `values [总行数, D]` + `offsets [B+1]` | 把所有样本的行**首尾相接压成一条**，只用一个 `offsets` 记边界                      |

**看一个具体例子**：`B = 3` 条样本，长度分别 `[2, 3, 1]`（第一条 2 个行为、第二条 3 个、第三条 1 个），维度 `D = 4`。

```
values（共 6 行，就是 2 + 3 + 1）：

  行0  ← 样本0 的第 1 个行为        ┐
  行1  ← 样本0 的第 2 个行为        ┘ 样本0
  行2  ← 样本1 的第 1 个行为        ┐
  行3  ← 样本1 的第 2 个行为        │ 样本1
  行4  ← 样本1 的第 3 个行为        ┘
  行5  ← 样本2 的第 1 个行为          样本2

offsets = [0, 2, 5, 6]
            │  │  │  └── 样本2：取行 [5, 6)，1 行
            │  │  └───── 样本1：取行 [2, 5)，3 行
            │  └──────── 样本0：取行 [0, 2)，2 行
            └─────────── 起点
```

**三条铁律**（后面每一行代码都建立在这三条上）：

1. **左闭右开**：样本 `i` 的数据 = `values[offsets[i] : offsets[i+1]]`
2. **首元素必须是 0、末元素必须等于总行数**，这叫 **complete cumsum**（完整前缀和，即"每项都是前面所有项累加的结果"）
3. **每段实际长度要现算**：`lengths = offsets[1:] - offsets[:-1]`（相邻两项相减）。上例 → `[2-0, 5-2, 6-5] = [2, 3, 1]` ✓

**为什么不能像普通张量那样直接切片？** 三个原因：

- `values[0:2]` 只能切出**某一个**样本。要批量切就必须写 `for` 循环逐样本 `.item()`，而每次 `.item()` 都要把数据从设备同步回 CPU，训练时会形成"同步等待"，是典型的性能杀手。（这一课后面那三个被淘汰的"草稿版"就是这种写法。）
- 长度不是常数 → 算子的静态 shape 假设失效，编译器 / 算子库没法提前规划。
- pad 成规整张量虽然能批量算，但会掺进大量无效的 0，而这些 0 一旦参与 attention、LayerNorm 这类计算就会污染结果（比如求均值时被 0 拉低），所以必须靠 mask 把它们排除掉。

## 2. 为什么会有这一课：NPU 上没有 Triton / CUTLASS

上游为变长序列准备了三套算子后端，NPU 上只有第三套能用：

| 后端    | 目录                | 实现方式               | NPU 能否用           |
| ------- | ------------------- | ---------------------- | -------------------- |
| cutlass | `ops/cuda_ops/`   | CUDA 核函数（`.cu`） | ❌ NVIDIA 专有       |
| triton  | `ops/triton_ops/` | Triton kernel          | ❌ NPU 不支持 Triton |
| pytorch | `ops/pt_ops/`     | 纯 PyTorch 算子拼装    | ✅ 唯一可用          |

**为什么前两套不能用**：CUTLASS 是 NVIDIA 的 CUDA 模板库；Triton 是专为 GPU 设计的 kernel DSL（写出来要编译到 GPU 的指令）。两者都绑死在 NVIDIA 平台上，NPU 都跑不了。所以只能走第三条路——**用 torch 原生的 `pad` / `cat` / `index_select` / 布尔 mask 这些算子，把原来的融合 kernel 一步步"拼"出来**。

代价要说清：融合 kernel 是一步算完、中间结果不落显存；纯 PyTorch 拼装则**每一步都会产生真实的中间张量**（显存占用和带宽都要付出代价）。这也解释了为什么代码里对"哪一步能不能省一次同尺寸张量"抠得那么细（§6 里 N 元版与二元版的对比就是例子）。

接入证据（改法是「换 import + 把原调用注释保留」，方便对照回滚）：

| 文件                                                                                                                                | 行    | 改法                                                                                                                 |
| ----------------------------------------------------------------------------------------------------------------------------------- | ----- | -------------------------------------------------------------------------------------------------------------------- |
| [hstu_processor.py](file:///c:/Users/z50066242/Desktop/zb/recsys-examples-25.09/examples/hstu/modules/hstu_processor.py#L31-L36)     | 31-36 | 注释掉`triton_split_2D_jagged` / `triton_concat_2D_jagged` 的 import，改 import `pt_jagged_tensors` 的两个函数 |
| [position_encoder.py](file:///c:/Users/z50066242/Desktop/zb/recsys-examples-25.09/examples/hstu/modules/position_encoder.py#L40-L43) | 40-43 | 改 import`pt_position` 的两个函数                                                                                  |
| [native_hstu_layer.py](file:///c:/Users/z50066242/Desktop/zb/recsys-examples-25.09/examples/hstu/modules/native_hstu_layer.py#L42)   | 42    | 换用`pt_norm_mul_dropout.pytorch_norm_mul_dropout`                                                                 |

调用点也是这个套路：原来的 triton 调用**注释保留、紧跟着写一行新调用**，比如：

```python
# _, sequence_embeddings = triton_split_2D_jagged(
#     jd.values, jd.max_seqlen,
#     offsets_a=jd.seqlen_offsets - jd.num_candidates_offsets,
#     offsets_b=seqlen_offsets,
# )
# 注释原算子实现，改用 PyTorch/NPU 兼容算子
_, sequence_embeddings = pytorch_split_2D_jagged(
    jd.max_seqlen, jd.values,
    max_len_left=None, max_len_right=None,
    offsets_left=jd.seqlen_offsets - jd.num_candidates_offsets,
    offsets_right=seqlen_offsets,
)
```

**重要澄清（别搞错归属）**：`pt_ops` 目录**不是华为新建的**。上游本来就有这个目录——`pt_norm_mul_dropout.py`、`torch_addmm.py`、`pt_hstu_attention.py` 都是 NVIDIA 原文件（带 SPDX NVIDIA 版权头），而且**根本不在 patch 里**。华为做的是**往里补 5 个新文件 + 改现有文件 3 行**。

## 3. 那 5 个文件分别做什么（逐个讲）

先给总表，再逐个展开：

| 文件                              | 行数 | 一句话功能                                         | 是否接入主流程              |
| --------------------------------- | ---- | -------------------------------------------------- | --------------------------- |
| `pt_jagged_tensors.py`          | 246  | 拼接 / 切分的**向量化实现**（没有 for 循环） | ✅ concat 1 处 + split 2 处 |
| `pt_position.py`                | 158  | **位置编码 + 时间编码**的向量化实现          | ✅ 3 处                     |
| `pt_add_position_embeddings.py` | 34   | 位置编码的**for 循环草稿版**                 | ❌ 全仓库无调用点           |
| `pt_concat_2d_jagged.py`        | 42   | 拼接的**for 循环草稿版**                     | ❌ 全仓库无调用点           |
| `pt_split_2d_jagged.py`         | 43   | 切分的**for 循环草稿版**                     | ❌ 全仓库无调用点           |

`gr_npu.patch` 在 `pt_ops/` 下**新增 5 个文件、共 523 行**。逐个说：

### 3.1 `pt_jagged_tensors.py`（246 行）：拼接 / 切分的向量化实现

**要解决什么问题**：HSTU 的一条样本，输入序列是"上下文特征 + 历史行为 + 候选物品"三段**拼**起来的（原因见 §6）。所以训练全程都在反复做两件事——

- **拼**：把"上下文段"和"历史段"接成一整条，送进 HSTU；
- **切**：HSTU 算完之后，把"候选段"切出来去打分、或者丢掉"上下文段"。

**文件里放了几样东西**：4 个对外函数（concat / split 各一个 + 两个 L2 函数）+ 2 个内部实现。

**为什么它叫"向量化"**：整个文件**没有一个 `for` 循环、没有一次 `.item()`**——所有样本用一次张量运算一起处理。这正是它与下面三个草稿版的分水岭。

**谁在调它**：`hstu_processor.py` 三处（1 次 concat + 2 次 split），细节见 §4。

### 3.2 `pt_position.py`（158 行）：位置编码 / 时间编码

**为什么需要它**：attention 本身**对顺序不敏感**——同一条序列打乱顺序喂进去，结果不变（它只算两两之间的关系，不看谁在前）。但推荐场景里**顺序就是信息**：先看手机壳再看手机，和反过来，含义完全不同。所以必须**显式把"位置"信息加到 embedding 上**。

推荐场景还要加第二样东西：**时间**。"三天前的点击"和"一年前的点击"，权重不该一样。所以还要把"这个行为离现在多久"编码成向量加进去。

**两个对外函数**：

| 函数                                            | 作用                                            |
| ----------------------------------------------- | ----------------------------------------------- |
| `pytorch_add_position_embeddings`             | 只加位置编码                                    |
| `pytorch_add_timestamp_positional_embeddings` | 位置编码 + 时间编码一起加（实际训练走的是这个） |

**谁在调它**：`position_encoder.py` 三处（第 114、141、166 行）。

实现思路很有意思，它是 §5 那套 5 步模板的一个**变体**：第 ③ 步做的是"加法"而不是"拼接"，而且因为**加法不改变长度**，第 ④⑤ 步可以整段省掉——第 4.4 节会详细讲。

### 3.3 `pt_add_position_embeddings.py`（34 行）：位置编码的草稿版

```python
def add_position_embeddings(jagged, jagged_offsets, high_inds, max_seq_len, dense, scale=1.0):
    """PyTorch 原生实现为变长序列添加位置编码"""
    L, D = jagged.shape
    B = high_inds.shape[0]
    out = torch.empty_like(jagged)
    for b in range(B):
        start = jagged_offsets[b].item()
        end = jagged_offsets[b + 1].item()
        seq_len = end - start

        pos_ids = torch.arange(seq_len, device=jagged.device) + high_inds[b].item()
        pos_emb = dense[pos_ids]
        out[start:end, :] = jagged[start:end, :] + pos_emb * scale
    return out
```

逐行看：`L` 是"总行数"（所有样本的行都堆在这一维上）；`out = torch.empty_like(jagged)` 先开一块同样大的输出；然后**逐样本**循环——`.item()` 把 `start/end` 搬到 CPU，`arange(seq_len) + high_inds[b]` 给这条样本的每个位置算一个"绝对位置号"，去位置表 `dense` 里取出对应的编码向量，加上去写回。

**为什么被淘汰**：每个样本要 3~4 次 `.item()`（CPU 同步），而循环体里只处理一条样本的数据——在 NPU/GPU 上极度不划算；同时 `out` 是循环中逐步填充的，autograd 也要为每个样本节点建图。对比 `pt_position.py` 的向量化版：用 `arange(max_seq_len).expand(B, max_seq_len)` **一次性**造出 `[B, max_seq_len]` 的索引矩阵，`clamp` 一下，一次 `index_select` 就把整批位置向量取出来。

### 3.4 `pt_concat_2d_jagged.py`（42 行）：拼接的草稿版

```python
def concat_2D_jagged(values_a, offsets_a, values_b, offsets_b):
    """PyTorch 原生 2D 不规则张量拼接算子"""
    B = offsets_a.shape[0] - 1
    D = values_a.shape[1]
    out_values = []
    out_offsets = [0]
    for i in range(B):
        a_start, a_end = offsets_a[i].item(), offsets_a[i + 1].item()
        b_start, b_end = offsets_b[i].item(), offsets_b[i + 1].item()
        a_slice = values_a[a_start:a_end]
        b_slice = values_b[b_start:b_end]
        out_slice = torch.cat([a_slice, b_slice], dim=0)
        out_values.append(out_slice)
        out_offsets.append(out_offsets[-1] + out_slice.shape[0])
    out_values = torch.cat(out_values, dim=0)
    out_offsets = torch.tensor(out_offsets, device=values_a.device, dtype=offsets_a.dtype)
    return out_values, out_offsets
```

比上一份更贵：每个样本 4 次 `.item()`，循环里全是小张量，最后还要 `torch.cat` 一次把所有小张量拼起来，并且**把 Python list 累加出来的长度最后才建成张量**。函数接口也和向量化版不同——它**返回新的 offsets**，而 `pt_jagged_tensors.py` 的向量化版只返回 values（offsets 由调用方"长度相加"得到，见 4.1）。

### 3.5 `pt_split_2d_jagged.py`（43 行）：切分的草稿版（附一个语义问题）

```python
for i in range(B):
    a_start = offsets_a[i].item()
    a_end = offsets_a[i + 1].item()
    b_start = offsets_b[i].item()
    b_end = offsets_b[i + 1].item()
    values_a_list.append(values[a_start:a_end])
    values_b_list.append(values[b_start:b_end])
```

**问题在于它把 `offsets_b` 当成"绝对下标"用了**（直接 `values[b_start:b_end]`）。而真实语义里，A、B 两段是**共处一条 values**：A 占前一段，B 接着 A 往后排，所以 B 段的绝对起点应该是 `offsets_A[i] + offsets_B[i]`（原算子和向量化版正是这么写的：`offsets = offsets_left + offsets_right`）。只有当 `offsets_a` 恒为 0（A 段长度为 0）时，两者才碰巧等价——这大概也是它没被采用的原因之一。

> 这类"看起来对、其实只在特例下对"的代码，是面试时最能体现"你真的读过"的素材：能说出**它在什么条件下才等价**，比说"这段是废弃代码"有价值得多。

### 3.6 目录里另外三个文件不是这次新增的

| 文件                       | 状态                             | 说明                                                                                                                                                             |
| -------------------------- | -------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `pt_norm_mul_dropout.py` | **被改 3 行**（`+3/-0`） | 补一句中文注释 +`if x.shape[0] == 0: return x`（输入为空时直接返回，避免 LayerNorm 在空张量上出问题）。注意 `x.shape[0]` 是**总行数**，不是 batch size |
| `torch_addmm.py`         | 上游原文件                       | 纯 torch 的`addmm + silu` 参考实现，被 `fused_hstu_op.py`、`paged_hstu_infer_layer.py` 引用                                                                |
| `pt_hstu_attention.py`   | 上游原文件                       | 纯 torch 版 HSTU attention，被`hstu_attention.py:112` 在 `kernel_backend == "pytorch"` 分支里引用                                                            |

这两个上游文件走的是"pytorch 后端"分支；本次 NPU 适配启用的是新增的 `npu_fused` 分支，所以那两条分支不会被走到。

**规模核账**：523 行新增里，3 个草稿文件占 119 行；`pt_jagged_tensors.py` 内部还有 2 个无调用点的 L2 函数占 71 行 → **真正接入主流程约 333 行**。面试里说"我重写了变长算子"，要能落到这个数字上。

**为什么留着这 119 行草稿？** 从代码本身只能看出：它们是同一批适配先写的朴素版，后来被向量化版取代，但文件没删。**patch 里没删 ≠ 有用途**——如果被追问"哪段是你写的"，这三份草稿要能识别出来。

## 4. `pt_jagged_tensors.py` 的 4 个对外函数

**先看文件全貌**：它其实有 **4 个对外函数 + 2 个内部实现**：

| 类型 | 函数                                  | 行  | 语义                                       |
| ---- | ------------------------------------- | --- | ------------------------------------------ |
| 对外 | `pytorch_concat_2D_jagged`          | 62  | 把两段变长序列**按样本拼成一段**     |
| 对外 | `pytorch_split_2D_jagged`           | 141 | 把一段变长序列**按样本切成两段**     |
| 对外 | `pytorch_hstu_split_l2_embeddings`  | 176 | 切出 prefix / l2 两段（无调用点）          |
| 对外 | `pytorch_hstu_concat_l2_embeddings` | 207 | 把 l2 拆开、把 prefix 插到中间（无调用点） |
| 内部 | `_concat_2D_jagged_jagged`          | 26  | concat 的真正实现                          |
| 内部 | `_split_2D_jagged_jagged`           | 116 | split 的真正实现                           |

**为什么对外层只有 30 行、真正的活都在内部函数里？** 因为对外层要处理一个"接口层面"的问题：`offsets` 允许传 `None`。这时它得自己把 offsets 补出来：

```python
if offsets_left is None:
    assert max_len_left is not None
    B = values_left.shape[0] // max_len_left       # 总行数 ÷ 每样本最大长度 = 样本数
    offsets_left_non_optional = max_len_left * torch.arange(B + 1, device=values_left.device)
```

`max_len_left * arange(B + 1)` 表达的就是"每条样本都一样长"的 offsets——比如 `max_len = 3`、`B = 2` → `[0, 3, 6]` ✓（间隔恒等于 3）。

同理，参数 `max_len_left` / `max_len_right` 也可能是 `None`，这时它要现算"最长的那个样本有多长"：

```python
max_len_left = int((offsets_left_non_optional[1:] - offsets_left_non_optional[:-1]).max().item())
```

（这里的 `.item()` 没必要避免：它算的是"元信息"，不参与梯度，而且整个 forward 只算一次。）

把这些"降级情况"处理完之后，对外层就把「offsets 一定非空、`max_len` 一定是 int」的参数交给内部层——所以**两个内部实现里一个 `if` 都没有**，逻辑非常干净。

**顺带一个细节**：文件里到处出现 `@torch.fx.wrap` 和 `fx_arange()`：

```python
@torch.fx.wrap
def fx_arange(len: int, device: torch.device) -> torch.Tensor:
    return torch.arange(len, device=device)
```

这是给 **`torch.compile` / FX 图捕获**留的口子：`torch.arange(len)` 里的 `len` 要运行时才知道，图编译器本来会试图把它展开或常量折叠；包一层 `fx.wrap`，就把它标记成"不透明节点"，图里只留一个调用。**它不是性能优化**，eager 执行时没有任何影响。

**三处调用点**（下一节逐个走）：`hstu_processor.py` 第 177 行做 1 次 concat，第 372、390 行各做 1 次 split。

### 4.1 举例：`pytorch_concat_2D_jagged`

**调用点**（[hstu_processor.py](file:///c:/Users/z50066242/Desktop/zb/recsys-examples-25.09/examples/hstu/modules/hstu_processor.py#L177-L184)）：把"上下文特征的 embedding"和"历史行为的 embedding"拼成一条完整序列。

```python
sequence_embeddings = pytorch_concat_2D_jagged(
    values_left=contextual_embeddings,                     # 左段 = 上下文
    values_right=sequence_embeddings,                      # 右段 = 历史行为
    max_len_left=None,                                     # 不给，由 offsets 现算 → 2
    max_len_right=None,                                    # → 3
    offsets_left=contextual_seqlen_offsets,
    offsets_right=sequence_embeddings_lengths_offsets,
)
```

**函数本体**（[pt_jagged_tensors.py:26-58](file:///c:/Users/z50066242/Desktop/zb/recsys-examples-25.09/examples/hstu/ops/pt_ops/pt_jagged_tensors.py#L26-L58)）：

```python
def _concat_2D_jagged_jagged(values_left, values_right, max_len_left, max_len_right,
                            offsets_left, offsets_right):
    max_seq_len = max_len_left + max_len_right                              # ① 整条最长 = 两段最长之和
    lengths_left = offsets_left[1:] - offsets_left[:-1]                     # ① 算每样本左段长度
    lengths_right = offsets_right[1:] - offsets_right[:-1]                  # ① 算每样本右段长度
    padded_left = torch.ops.fbgemm.jagged_to_padded_dense(                  # ② pad 左段
        values=values_left, offsets=[offsets_left],
        max_lengths=[max_len_left], padding_value=0.0,
    )
    padded_right = torch.ops.fbgemm.jagged_to_padded_dense(                 # ② pad 右段
        values=values_right, offsets=[offsets_right],
        max_lengths=[max_len_right], padding_value=0.0,
    )
    concatted_dense = torch.cat([padded_left, padded_right], dim=1)         # ③ 拼！（唯一真正的运算）
    mask = fx_arange(max_seq_len, device=offsets_left.device).view(1, -1)   # ④ 造位置号 c = 0..max_seq_len-1
    mask = torch.logical_or(
        mask < lengths_left.view(-1, 1),                                    # ④ 落在左段真数据内
        torch.logical_and(
            mask >= max_len_left,                                           # ④ 右段起点是【常数】
            mask < max_len_left + lengths_right.view(-1, 1),
        ),
    )
    return concatted_dense.flatten(0, 1)[mask.view(-1), :]                  # ⑤ 去 pad
```

**用一个具体例子走一遍**。设 batch = 2 条样本，`D` 是特征维度（具体几维不影响理解）：

- 左段（上下文）：长度 `[2, 1]` → `offsets_left = [0, 2, 3]`，`values_left` 共 3 行
- 右段（历史行为）：长度 `[3, 2]` → `offsets_right = [0, 3, 5]`，`values_right` 共 5 行

把行起个名字方便对照：左段的 3 行是 `L00 L01`（样本0）和 `L10`（样本1）；右段的 5 行是 `R00 R01 R02`（样本0）和 `R10 R11`（样本1）。

**① 算长度**：`lengths_left = [2-0, 3-2] = [2, 1]`；`lengths_right = [3-0, 5-3] = [3, 2]` ✓

**② pad**（补 0 到各自的 `max_len`）：

```
padded_left  [2, 2, D]              padded_right  [2, 3, D]
  样本0: [ L00   L01 ]                样本0: [ R00   R01   R02 ]
  样本1: [ L10    ●  ]                样本1: [ R10   R11    ●  ]
                 ↑ 补的 0                            ↑ 补的 0
```

**③ `cat(dim=1)`** → `[2, 5, D]`（`max_seq_len = 2 + 3 = 5`）。注意这一步只是"把两块拼在一起"，**没有任何计算**：

```
样本0:  [ L00  L01 | R00  R01  R02 ]      ← 5 个位置全是真的
样本1:  [ L10   ●  | R10  R11   ●  ]      ← 位置 1、4 是洞
```

**④ 算 mask**。先造位置号 `c = arange(5) = [0, 1, 2, 3, 4]`，然后按两个条件做 `OR`：

- 条件一（在左段真数据里）：`c < lengths_left` —— 注意左段**原本有多长**是 `lengths_left[i]`，不是 `max_len_left`
- 条件二（在右段真数据里）：`max_len_left ≤ c < max_len_left + lengths_right`

```
样本0（lengths_left[0]=2, lengths_right[0]=3）:
  条件一  c<2        →  T T F F F
  条件二  c∈[2,5)    →  F F T T T
  OR                 →  T T T T T     （5 个位置全取）

样本1（lengths_left[1]=1, lengths_right[1]=2）:
  条件一  c<1        →  T F F F F
  条件二  c∈[2,4)    →  F F T T F
  OR                 →  T F T T F     （只取 3 个位置）
```

**⑤ 去 pad**。`flatten(0, 1)` 把 `[2, 5, D]` 压成 `[10, D]`（样本0 的 5 行在前、样本1 的 5 行在后），`mask.view(-1)` 是长度 10 的布尔向量，两者对齐取出行——样本0 取 5 行、样本1 取 3 行，**共 8 行**。

**输出的 offsets 从哪来？** 函数**只返回 values，不返回 offsets**。调用方自己算：

```python
sequence_embeddings_lengths = contextual_seqlen + sequence_embeddings_lengths   # 长度直接相加
sequence_embeddings_lengths_offsets = torch.ops.fbgemm.asynchronous_complete_cumsum(
    sequence_embeddings_lengths
)
```

因为"拼完之后样本 `i` 的长度 = 左段长度 + 右段长度"，所以长度是相加的、offsets 是长度再做一次 complete cumsum。这也解释了为什么 `pt_concat_2d_jagged.py` 那个草稿版"返回新 offsets"的接口反而不必要——**调用方本来就能零成本地算出来**。

**最关键的一点：那个"洞"**。右段在 padded 张量里的起点是**常数 `max_len_left`**（因为拼接点固定在 D 维度上的第 `max_len_left` 列），而左段的**真实**长度 `lengths_left[i]` 是**逐行变化**的。只要某条样本的左段长度小于 `max_len_left`，`[lengths_left[i], max_len_left)` 这段就是洞（样本1 的位置 1）。

所以 mask **必须用 `logical_or` 合并两个条件**，不能写成一个连续区间。对比一下 split（下一节）：split 里两段是首尾相接的、没有洞，所以它那里就是两次独立比较，没有 `OR`。

### 4.2 举例：`pytorch_split_2D_jagged`

**两个调用点**（[hstu_processor.py:372](file:///c:/Users/z50066242/Desktop/zb/recsys-examples-25.09/examples/hstu/modules/hstu_processor.py#L362-L379) 与 [390](file:///c:/Users/z50066242/Desktop/zb/recsys-examples-25.09/examples/hstu/modules/hstu_processor.py#L380-L397)）：

```python
# 调用点 2：丢掉历史，只留候选
_, sequence_embeddings = pytorch_split_2D_jagged(
    jd.max_seqlen, jd.values,
    max_len_left=None, max_len_right=None,
    offsets_left=jd.seqlen_offsets - jd.num_candidates_offsets,   # 左段 = 历史（去掉候选）
    offsets_right=seqlen_offsets,                                 # 右段 = 候选
)

# 调用点 3：丢掉上下文，只留上下文之后的部分
_, sequence_embeddings = pytorch_split_2D_jagged(
    jd.max_seqlen, jd.values,
    max_len_left=None, max_len_right=None,
    offsets_left=jd.contextual_seqlen_offsets,                    # 左段 = 上下文
    offsets_right=seqlen_offsets,                                 # 右段 = 其余全部
)
```

两处都只取返回值里的第二个（`_` 表示丢掉第一段）。注意两个调用点传进来的两个 offsets **相加恰好等于整条序列的长度**：`(seqlen_offsets - num_candidates_offsets) + num_candidates_offsets = seqlen_offsets` ✓ ——这不是巧合，而是这个算子的**前提**（见下面①）。

**函数本体**（[pt_jagged_tensors.py:116-137](file:///c:/Users/z50066242/Desktop/zb/recsys-examples-25.09/examples/hstu/ops/pt_ops/pt_jagged_tensors.py#L116-L137)）：

```python
def _split_2D_jagged_jagged(max_seq_len, values, offsets_left, offsets_right):
    offsets = offsets_left + offsets_right                                  # ① 两段 offsets 相加 = 整条 offsets
    padded_values = torch.ops.fbgemm.jagged_to_padded_dense(                # ② pad 一次（只有一个输入张量！）
        values=values, offsets=[offsets],
        max_lengths=[max_seq_len], padding_value=0.0,
    ).flatten(0, 1)
    lengths_left = offsets_left[1:] - offsets_left[:-1]                     # ① 左段每样本长度
    lengths_right = offsets_right[1:] - offsets_right[:-1]                  # ① 右段每样本长度
    mask = fx_arange(max_seq_len, device=values.device).view(1, -1)         # ④ 位置号 c
    mask_left = mask < lengths_left.view(-1, 1)                             # ④ 左段：从头数 lengths_left 个
    mask_right = torch.logical_and(
        mask >= lengths_left.view(-1, 1),                                   # ④ 右段起点是【逐行变化的】
        mask < (lengths_left + lengths_right).view(-1, 1),
    )
    return padded_values[mask_left.view(-1), :], padded_values[mask_right.view(-1), :]   # ⑤ 切成两块
```

**① 为什么 `offsets_left + offsets_right` 能等于整条 offsets？** 因为这两个 offsets 都是 **complete cumsum**（首元素为 0）：

```
offsets_left  = cumsum(左段长度) = [0, 2, 3]
offsets_right = cumsum(右段长度) = [0, 3, 5]
                相加              = [0, 5, 8]   ← 正好是整条序列的 offsets
```

数学依据是前缀和的**可加性**：`cumsum(a) + cumsum(b) = cumsum(a + b)`。这条性质有个前提——**必须是 complete cumsum**（从 0 开始累加）。如果 offsets 只记"段内偏移"、不是从 0 起，这个加法就不成立。这就是为什么代码里到处用 `asynchronous_complete_cumsum` 来造 offsets。

**沿用上一组的数字走一遍**：输入 8 行（`offsets = [0, 5, 8]`，样本0 长 5、样本1 长 3）；要切成左段长度 `[2, 1]`（`offsets_left = [0, 2, 3]`）、右段长度 `[3, 2]`（`offsets_right = [0, 3, 5]`）。

**② pad** 到 `max_seq_len = 5`（**注意只 pad 一个张量**，不像 concat 要 pad 两个）：

```
样本0:  [ X00  X01  X02  X03  X04 ]      ← 5 个位置全是真的
样本1:  [ Y10  Y11  Y12    ●    ●  ]      ← 后两个位置是 pad（样本1 只有 3 行）
```

**④ 两次 mask**（左边不依赖右边、右边不依赖左边，**各算各的**）：

```
样本0（lengths_left=2, lengths_right=3）:
  左段  c<2         →  T T F F F
  右段  c∈[2, 2+3)  →  F F T T T

样本1（lengths_left=1, lengths_right=2）:
  左段  c<1         →  T F F F F
  右段  c∈[1, 1+2)  →  F T T F F     ← 注意 c=1 在这里属于【右段】
```

**⑤ 返回两块**：左段 3 行（2+1）、右段 5 行（3+2），合计 8 行 ✓ 与输入行数对得上（这也说明"切分"不丢数据）。

**和 concat 的唯一区别（很重要的对照）**：

|        | 右段起点                               | 有没有洞                                         | mask 怎么写        |
| ------ | -------------------------------------- | ------------------------------------------------ | ------------------ |
| concat | **常数** `max_len_left`        | 有（左段被"拉长"到`max_len_left`，中间空出来） | 两个条件必须`OR` |
| split  | **逐行变化** `lengths_left[i]` | 没有（两段首尾相接）                             | 两次独立比较即可   |

split 里 `[0, lengths_left)` 和 `[lengths_left, lengths_left + lengths_right)` **严丝合缝**地拼成整条（因为二者的和就是真实长度），所以两个 mask 加起来刚好覆盖所有真数据、不重不漏——不需要 `OR`。

### 4.3 剩下两个 L2 函数

`pytorch_hstu_split_l2_embeddings`（176 行）/ `pytorch_hstu_concat_l2_embeddings`（207 行）这两个函数，**在整个代码库里没有任何调用点**。看它们的实现：

```python
# pytorch_hstu_split_l2_embeddings：从一条 x 里切出 prefix 段和 l2 段
x_offsets = prefix_offsets + l2_offsets                    # 又是"两段 offsets 相加 = 整条"
x_lengths = x_offsets[1:] - x_offsets[:-1]
padded_x = torch.ops.fbgemm.jagged_to_padded_dense(
    values=x, offsets=[x_offsets], max_lengths=[max_seq_len], padding_value=0.0,
).flatten(0, 1)
prefix_lengths = prefix_offsets[1:] - prefix_offsets[:-1]
mask = fx_arange(max_seq_len, device=x_offsets.device).view(1, -1)
mask_prefix = torch.logical_and(
    mask >= contextual_seq_len,                            # prefix 从 ctx 之后开始
    mask < prefix_lengths.view(-1, 1) + contextual_seq_len,
)
mask_l2 = torch.logical_or(                                # l2 被劈成两段（前面 ctx 个 + 后面的）
    mask < contextual_seq_len,
    torch.logical_and(
        mask >= prefix_lengths.view(-1, 1) + contextual_seq_len,
        mask < x_lengths.view(-1, 1),
    ),
)
```

```python
# pytorch_hstu_concat_l2_embeddings：把 l2 劈开、把 prefix 塞到中间
padded_x = torch.cat([
    padded_l2_x[:, 0:contextual_seq_len, :],   # l2 的前 ctx 个
    padded_prefix_x,                           # prefix 全部
    padded_l2_x[:, contextual_seq_len:, :],    # l2 的剩余
], dim=1)
```

第二个函数的 `torch.cat` 把布局写得很直白：**结果 = `[l2 前 ctx 个 | prefix 全部 | l2 剩余]`**。这个"交错"布局不是随手写的——它对应 Triton 里一类叫 `_w_prefix` 的 kernel 变体（[triton_jagged.py:806](file:///c:/Users/z50066242/Desktop/zb/recsys-examples-25.09/examples/hstu/ops/triton_ops/triton_jagged.py#L806-L860)）：

```python
if IS_REPLACE:
    ...
else:
    seq_len = seq_len_a + seq_len_b          # 整条长度 = 两段相加 ← 和 pt_ops 里 offsets 相加同源
    ...
    seq_start = seq_start_a + seq_start_b    # 整条起点 = 两段 offsets 相加
    out_seq_b_start = seq_len_a + n_prefix_to_B
```

`n_prefix_to_B` 的语义就是"把 B 段开头的 `n_prefix` 个元素挪到 A 段前面去"，输出布局是 `[B前n_prefix | A | B剩余]`。对照 `pt_ops` 的 `[l2前ctx | prefix | l2剩余]`，可以一一对上：

- **A = prefix，B = l2，`n_prefix` = `ctx` = `contextual_seq_len`**
- 所以这两个函数是 Triton 那两个 `_w_prefix` kernel 的**忠实移植**，不是自创 API

> **要说清的边界**：它们的业务语义在本次适配中**没有启用**（无调用点）。`l2` 的确切含义（两级序列里的第二级？长序列的二级缓存？）**在代码里没有证据**，面试里只能说"按上游已有的算子形式做了移植，主流程没用它"，不要编造它做什么。

### 4.4 顺带对照：`pt_position.py` 是同一套 5 步模板的"加法变体"

`pytorch_add_position_embeddings`（[pt_position.py:30-61](file:///c:/Users/z50066242/Desktop/zb/recsys-examples-25.09/examples/hstu/ops/pt_ops/pt_position.py#L30-L61)）：

```python
def pytorch_add_position_embeddings(jagged, jagged_offsets, high_inds, max_seq_len, dense, scale=1.0):
    jagged = jagged * scale                                                    # 输入先乘一个缩放
    B = high_inds.size(0)
    col_indices = torch_arange(max_seq_len, device=high_inds.device).expand(B, max_seq_len)
    col_indices = torch.clamp(col_indices, max=high_inds.view(-1, 1))          # 位置号封顶，防越界
    dense_values = torch.index_select(dense, 0, col_indices.reshape(-1)).view(B, max_seq_len, -1)
    padded_left = torch.ops.fbgemm.jagged_to_padded_dense(                     # ② pad
        values=jagged, offsets=[jagged_offsets], max_lengths=[max_seq_len], padding_value=0.0,
    )
    dense_result = padded_left + dense_values                                  # ③ 算 = 【加法】
    jagged_result = torch.ops.fbgemm.dense_to_jagged(dense_result, [jagged_offsets])[0]   # ⑤ 变回去
    return jagged_result
```

对照 §5 的模板看，它做了几件不一样的事：

- **③ 这一步是"加法"，不是"拼接"**：`padded_left + dense_values`（位置编码表 `dense` 里按位置号取出来的向量，直接加到 embedding 上）。
- **④ 不需要 mask**：因为加法**不改变长度**，每条样本的有效位置和 padding 位置一一对应，加完还是同样的长度。
- **⑤ 不用 mask 去 pad，而是用 `dense_to_jagged(..., [jagged_offsets])` 按原 offsets 还原**——既然长度没变，原来的 offsets 直接就能用。这一步等价于"去 pad"，但省掉了 mask 的构造。

再看时间编码版 `pytorch_add_timestamp_positional_embeddings`（第 99-158 行）——它把"位置"和"时间"两件事合起来算：

```python
num_time_buckets = 2048          # 时间桶总数
time_bucket_increments = 60.0    # 一个桶 = 60 秒
timestamps = timestamps[:, :max_seq_len]
query_time = torch.gather(timestamps, dim=1, index=(seq_lengths - 1).unsqueeze(1).clamp(min=0))  # 当前时刻 = 最后一条行为的时间
ts = query_time - timestamps     # 每个行为"距今多少秒"
ts = ts.clamp(min=1e-6) / time_bucket_increments
if time_bucket_fn == "log":
    ts = torch.log(ts)           # log 或 sqrt 压缩量纲
else:
    ts = torch.sqrt(ts)
ts = (ts / time_bucket_divisor).clamp(min=0).int()
ts = torch.clamp(ts, min=0, max=num_time_buckets)     # 桶号封顶
...
return torch.ops.fbgemm.jagged_dense_elementwise_add_jagged_output(
    seq_embeddings, [seq_offsets], (time_embeddings + position_embeddings).to(seq_embeddings.dtype)
)[0]
```

三个值得记住的点：

1. **"距今多久"是相对量**：用"最后一个行为的时间戳"当"当前时刻"，其它行为减去它。这样推理时不管数据是几点跑，语义都一致。
2. **要压缩量纲**：秒数差值跨度极大（1 秒 ~ 几年），所以除以 60 变分钟，再用 `log` 或 `sqrt` 压一下，最后取整当成"桶号"去查一张 `[2048, D]` 的时间编码表——用离散的桶代替连续值，是推荐系统里很常见的做法（省参数、抗噪）。
3. **`jagged_dense_elementwise_add_jagged_output` 这个 fbgemm 算子本身就是"pad → 加 → 去 pad"的融合版**：jagged 输入 + dense 输入 → jagged 输出，长度不变所以不需要 mask。它和 `pytorch_add_position_embeddings` 里手写的那三步是**等价的两种写法**（一个调融合算子、一个自己拆开写）——正好印证了"5 步模板"是这个领域的通用语言。

`_get_col_indices`（第 64-96 行）里还有两处细节，值得知道但**不要过度解读**：

```python
if num_targets is not None:
    high_inds = seq_lengths - num_targets
    col_indices = torch.clamp(col_indices, max=high_inds.view(-1, 1))
    col_indices = high_inds.view(-1, 1) - col_indices      # ← 位置号变成"从后往前数"
...
col_indices = col_indices + max_contextual_seq_len          # 整体平移，给上下文段让位
col_indices = torch.clamp(col_indices, max=max_pos_ind - 1) # 封顶，别越出位置表
if max_contextual_seq_len > 0:
    col_indices[:, :max_contextual_seq_len] = arange(max_contextual_seq_len)   # 上下文段固定给 0..ctx-1
```

- 候选的位置号是**由 `seq_lengths - num_targets` 再减出来的**（`high_inds - c`），也就是"从候选处往回数"——这样**同一个位置在不同长度的序列里含义一致**（都代表"离预测目标第几步"）。
- 整体 `+ max_contextual_seq_len` 是给上下文段"让位置"，而上下文段本身被强制覆写成 `0..ctx-1`（上下文特征没有时序含义，给固定位置即可）。
- 最后 `clamp(..., max=max_pos_ind - 1)` 是防越界：位置表只有 `max_pos_ind` 行。

（代码里**没有注释**解释为什么这么设计，上面这几条是从运算本身读出来的；面试里说到"为什么"时，用"从实现看是……"的措辞更稳。）

## 5. 变长问题的通用套路（5 步）

把前面的函数抽象一下，**所有**变长张量的操作都可以套这个模板：

| 步 | 动作                                                                                        | 为什么需要这一步                                                             |
| -- | ------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------- |
| ① | 由 offsets 算每段长度：`lengths = offsets[1:] - offsets[:-1]`                             | jagged 只记"边界"，不记"长度"；而后面造 mask 全靠长度                        |
| ② | pad 成规整 dense：`jagged_to_padded_dense(..., max_lengths=[max_len], padding_value=0.0)` | 只有形状规整的张量才能批量算（`cat` / 加法 / `index_select` 都要求规整） |
| ③ | 在 dense 上做**真正要做的事**                                                         | 这一步才是"业务"：拼接、加法……                                             |
| ④ | 造布尔 mask，标出"哪些位置是真数据"                                                         | pad 出来的 0 是假的，必须能被筛掉                                            |
| ⑤ | `flatten(0,1)` 后按 mask 取出真数据（去 pad）                                             | 结果要交回 jagged 世界                                                       |

**为什么必须这么绕？** jagged 每行长度都不一样，没法直接做批量运算；pad 成 dense 之后形状规整，才能用上张量化算子、也才能被 autograd 正确追踪。代价是"算了一堆 padding 的无效数据"，所以第 ⑤ 步必须把它们扔掉。

**用 §1 那个 3 条样本的例子，完整走一遍**（长度 `[2, 3, 1]`，`offsets = [0, 2, 5, 6]`，`max_len = 3`）。假设要做的事是"把每个 embedding 乘 2"（长度不变的操作，最能看清模板的骨架）：

```
① 算长度        lengths = [2-0, 5-2, 6-5] = [2, 3, 1]   ✓ 和原始长度对上

② pad 成 dense  padded [3, 3, D]
                  样本0: [ 行0   行1    ●  ]
                  样本1: [ 行2   行3   行4 ]
                  样本2: [ 行5    ●     ●  ]

③ 算            padded = padded * 2                    ← 所有位置（含 padding）一起算，省事

④ 造 mask       c = [0, 1, 2]
                  样本0: c < 2  →  T T F
                  样本1: c < 3  →  T T T
                  样本2: c < 1  →  T F F

⑤ 去 pad        flatten(0,1) → [9, D]；按 mask 取 → 6 行（2+3+1）✓ 回到最初的行数
                  offsets 怎么办？长度没变，【直接复用】原来的 [0, 2, 5, 6]
```

对照这张图，就能理解 §4 里两个函数的差别到底在哪：

| 操作            | 第②步                     | 第③步         | 第④步                | 第⑤步                   | offsets 怎么定                              |
| --------------- | -------------------------- | -------------- | --------------------- | ------------------------ | ------------------------------------------- |
| concat（4.1）   | pad**两个**张量      | `cat(dim=1)` | mask 要`OR`（有洞） | 按 mask 取               | 长度**变了** → 长度相加 + `cumsum` |
| split（4.2）    | pad**一个**张量      | 什么都不做     | 两次独立 mask（无缝） | 返回**两块**       | 两段各自 +`cumsum` 重算                   |
| 位置编码（4.4） | pad 一个（或交给融合算子） | **加法** | 可省（长度不变）      | `dense_to_jagged` 还原 | 长度不变 →**复用**原来 offsets       |

**两个容易被追问的点，提前备好：**

- **为什么 `offsets_left + offsets_right` 可以直接相加？** 因为它俩都是 **complete cumsum**（首元素为 0）：`cumsum(a) + cumsum(b) = cumsum(a + b)`。如果 offsets 不是从 0 起累加的，这个加法立刻失效——这就是代码里到处用 `asynchronous_complete_cumsum` 的原因。
- **concat 和 split 是什么关系？** 在原始实现里，**它们互为对方的 backward**：Triton 版的 `_Concat2DJaggedFunction.backward` 里调的是 `split_2D_jagged`，`_Split2DJaggedFunction.backward` 里调的是 `concat_2D_jagged`。道理很直观——两个算子都只"搬运数据、不做计算"，所以数学上就是"转置即逆"，前向切开的动作就是反向拼接的梯度。华为的 `pt_ops` 版**只搬了前向**，互逆性靠 autograd 自动复用，不需要自己写 backward。

## 6. 为什么会有「两段」（简写）

- **前置事实**：HSTU 是 encoder-only，所有信息必须在**同一条序列**里 → 一条样本 = 「上下文特征 + 历史行为 + 候选物品」三段**拼**成一条长序列。
- **拼一次 = 一个接缝 = 两段**。所以 concat 天然是二元的（左段 / 右段）；split 是它的逆，也必然是二元的。
- **三个调用点，全是单分界点**：

| 位置                      | 操作   | 左段 / 右段                       |
| ------------------------- | ------ | --------------------------------- |
| `hstu_processor.py:177` | concat | 上下文 / 历史                     |
| `hstu_processor.py:372` | split  | 历史 / 候选（只留候选）           |
| `hstu_processor.py:390` | split  | 上下文 / 上下文之后（丢掉上下文） |

- **两个细节**：① concat 的左段被 pad 撑到 `max_len_left`，右段起点是**常数**，中间有洞 → mask 必须 `OR`；② split 的右段起点是**逐行变化**的 `lengths_left[i]`，两段无缝 → 两次 mask 独立算。
- **上游其实有 N 元版**（[jagged_tensor_op.py](file:///c:/Users/z50066242/Desktop/zb/recsys-examples-25.09/examples/hstu/ops/jagged_tensor_op.py#L94) 的 `concat_2D_jagged_tensors`）。华为做成二元版的可见理由：原算子调用点只传 2 个元素、必须和 split 成对；而且 N 元版要额外 pad 一个与 dense **同尺寸的 ones 张量**当 mask（多一次同尺寸显存分配和读写），二元版直接用 offsets 算术算 mask，省掉这一笔。
- **面试一句话**：「一条样本由上下文、历史、候选三段拼成，拼接和拆分都只有一个接缝，所以算子做成二元的；实现上都是 pad → 算 → 去 pad 的同一套模板。」

## 7. 名词速查

- **样本**：一条训练样本 = **一次推荐请求**（一个用户的上下文 + 历史行为 + 待打分的候选物品），**不是"一个用户"**。`batch_size` 就是"这一批里有几次请求"。
- **候选在序列末尾**：`dataset/utils.py` 的 `Batch.random` 里，`num_candidates = seqlen - clamp(seqlen - max_num_candidates, min=0)` —— 候选是从序列**尾部**截出来的。
- **ctx / prefix / l2**：都是序列里的**段**。`ctx`（`contextual_seq_len`）= 上下文特征占的位置数；`prefix` / `l2` 只出现在那两个**无调用点**的 L2 函数里（见 4.3）。
- **values / offsets**：jagged 的两个组成。`values [总行数, D]` 存数据，`offsets [B+1]` 存边界（左闭右开）。
- **complete cumsum**：首元素为 0、末元素等于总行数的前缀和。它是 offsets 的"合法格式"，也是 `offsets_left + offsets_right` 能成立的前提。

## 8. 记忆锚点

1. NPU 上没有 Triton/CUTLASS，**变长算子只能用纯 PyTorch 拼**——`pt_ops` 目录是上游就有的，本次新增 5 个文件 / 523 行
2. 5 个文件 = 2 个向量化版（`pt_jagged_tensors.py`、`pt_position.py`）+ 3 个 for 循环草稿（119 行、无调用点）
3. 变长套路固定 5 步：**算长度 → pad 成 dense → 算 → 造 mask → 去 pad**
4. `offsets_left + offsets_right` 合法的前提是 **complete cumsum 可加**（`cumsum(a)+cumsum(b)=cumsum(a+b)`）
5. concat 有洞（右段起点是常数，mask 要 `OR`）；split 无缝（右段起点逐行变化，两次 mask 独立）
6. concat 与 split **互为对方的 backward**（转置即逆，只搬运不计算）
7. 一条样本 = 上下文 + 历史 + 候选，**拼一次只有一个接缝**，所以是二元的；候选在序列末尾
8. 长度不变的操作（加法）**可以省掉 mask**：用原来的 offsets 交给 `dense_to_jagged` 还原即可
9. 面试要能落到数字：**523 行新增，3 份草稿 119 行无调用点，真正接入约 333 行**
