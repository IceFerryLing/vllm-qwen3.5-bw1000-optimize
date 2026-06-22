# DUMMA 编译

DUMMA probe 使用 `dcc` 编译，不走 `hipcc`。

关键参数：

```bash
dcc -x hip <source>.cpp -O3 --offload-arch=gfx936 \
  -L/opt/dtk-26.04-DCC2602-0317/dcc/lib/clang/17.0.0/lib/linux \
  -o <output>
```

