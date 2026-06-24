# PyTorch prefill 侧开销

本文件原先把 `FillFunctor<int>` 写成了 prefill 侧开销，这个归类不准确。

当前更正：

- `FillFunctor<int>` 不算 prefill 计算本身的一部分。
- 它应单独拆成 PyTorch / metadata / buffer 初始化开销分析。
- prefill 计算热点仍应单独看 attention、GDN/FLA、GEMM 等实际计算 kernel。

拆分后的 fill 分析见：

```text
pra26/pytorch_fill.md
```
