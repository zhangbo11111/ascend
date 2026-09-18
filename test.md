
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
