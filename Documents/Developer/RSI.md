# RSI.md — 10-minute dev loop prompt

Run this prompt as-is to start the loop:

```
/loop 10m

你在 CroqTile repo（/home/albert/workspace/croqtile）根目录跑一个每 10 分钟一次的
开发 loop。每次只推进一个 component 的一小步，不扩 scope。

开始前必读（按顺序）：
1. Documents/research/performance-gap-cuda-vs-croqtile-spmm.md（尤其 §8 实现状态、
   负面实验与剩余 gap 的来源）
2. memory：~/.claude/projects/-home-albert-workspace-croqtile/memory/MEMORY.md，
   croqtile-local-llvm-toolchain.md（backlog/逐轮记录）、spmm-campaign-benchmark.md
3. Documents/Developer/RSI.md（本文件：流程与环境规则）
4. git log --oneline -10 与 git status

## 目标与当前阶段（2026-11-08 起）

F1–F7 语法/编译器能力已全部完成（F1 register metadata、F2 event ring、F3 TMA 诊断、
F4.1 per-warpgroup barrier ids、F4.2 concurrent-branch storage、F5 scoped arena reuse、
F6 commit/wait split、F7 width-1 producer）。当前目标是：

- **阶段 A — 性能实验（需要 GPU）**：用 CroqTile 写出与 1845.08 TFLOPS CUDA kernel
  同配置（A 2:4-sparse e4m3 / B dense e4m3 / C fp16，16384³，256×256 CTA，
  4 warpgroups，2-stage）可对比的 1800-class kernel，并判定真实 gap。当前最佳约
  1440 TFLOPS（gap ≈22%）。不要重复 §8 已记录的负面实验。
- **阶段 B — 编译器 backlog 修复（不需要 GPU）**：影响生成代码正确性的缺陷优先
  （metadata/liveness、fragment 数组槽地址、TMA 视图诊断、GenIndices 坐标线性化等）；
  backlog 与根因在 memory 的 croqtile-local-llvm-toolchain.md。每个修复必须带 lit
  回归测试。
- **默认优先级**：正确性修复（影响发射代码）> 性能实验（GPU 可用时）> 诊断/清理。
  GPU 不可用时自动落到阶段 B。

本 loop 同时维护 F1–F7 的现有行为：每轮必须保证已有 lit 套件不回归。

然后严格执行：

0. **Round lock（防重叠）**：tick 可能在上轮还没收尾时到达；此时不能开始新 task。
   在选 task 前先运行：

       python3 scripts/loop_lock.py check

   - 退出码 0：存在未过期的 round。**不要选新 task**；把脚本打印的 task 当作本轮
     target 继续（先看 `git status` / `git diff` 里未提交的进度）。如果该 task 已经
     commit 且 working tree clean，说明上一轮只是忘了 clear：清掉 lock 并报告完成。
   - 退出码 1：没有 active round，按 A/B 正常选路。
   - 选定 task 后立刻写入 lock（如果本轮是 resume，不要重写）：

         python3 scripts/loop_lock.py start --focus "<focus>" --task "<one-line task>"

   - 另外先做进程检查：`pgrep -af "tests/lit.sh|spmm-bench|verify.py|profile.py|ncu "`。
     有本 loop 之外的测试/benchmark 在跑时**跳过本轮**，只输出一行
     "跳过：上一轮工作仍在进行"，不启动任何新任务、不重复编译或跑测试。
   - E 节正常收尾（commit + 汇报）后清除 lock：

         python3 scripts/loop_lock.py clear

   - 只有在本轮被上下文/工具异常打断、没走到 E 时保留 lock；下一 tick 会优先 resume
     同一 task。lock 超过 60 min 会被 `check` 当 stale 清掉，避免永久卡住。

A. 选路（先看有没有 --focus）
- **没有 `--focus`：默认分支。** 按"默认优先级"从当前阶段选一个下一步。
- **有 `--focus`：focus 分支。** focus 是本轮唯一 target；它决定 B 节选什么、D 节验证
  什么。focus 同样受"单一 component"约束，不能借 focus 扩 scope。
- **focus 是性能语义时**（某个 kernel / metric / 延迟吞吐）：走阶段 A 的实验协议；
  GPU 不可用则本轮报告阻塞，不落到别的工作。

B. 选 task（只选一个）
- 一个 task 只能是**一个 component 的 0→1 或一个明确缺陷的修复**：
  - 正确性修复：给出可复现的错误行为（崩溃/静默错码/错误诊断），修根因，加回归测试。
  - 性能实验：一个 idea、一个可对比指标（TFLOPS）、一次 A/B。
  - 诊断/能力补齐：一个具体诊断或一处代码生成能力。
- **不按 LOC/文件数裁剪**；判断标准是 component 是否单一。混合多个 component 时先
  拆开，只做其中一个，并在输出里说明下一轮做哪个。
- 不顺手重构、不新增依赖、不改设计决策、不扩 docs 主题。
- 只有 task 需要产品/设计决策、新依赖，或与已记录的决策冲突时才停下并报告。

C. 实现
- 只改该 task 必需的文件；匹配周边风格；能复用既有 helper 就不要新造。
- 不做 over-design：不加未用参数、不做插件系统；一个修复配一个回归测试。
- **loop 模式授权**：本文件就是 developer 的 pre-approval，不要为每个 task 再问一次；
  按 scope 规则自行选择、实现、验证、commit。只有触发停止条件时才报告。

D. 验证
- 构建：`cd build-release && ninja choreo`。
- 测试（按影响面从小到大）：
  - 相关文件：`PATH=/tmp/fakebin:$PATH ./tests/lit.sh <file_or_dir>`（一次只传一个
    路径，lit.sh 只取第一个参数）。
  - 编译面全量（不含 `tests/gpu/end2end`）：逐目录运行 `tests/parse`、`tests/check`、
    `tests/cli`、`tests/pp`、`tests/cpu`、`tests/norm`、`tests/infer`、`tests/others`、
    `tests/agent`、`tests/lit-self`、`tests/gpu/{check,cli,codegen,liveness,norm}`、
    `tests/amdgpu/{check,codegen}`。nvidia-smi shim 见"环境规则"。
  - 完整套件：`./tests/lit.sh tests/` **只在 GPU 健康且没有其他套件/benchmark 运行时**
    串行执行；e2e 用例是 2026-11-08 事故的来源。
- 性能实验（阶段 A）：
  - 用本机 harness（/home/albert/workspace/spmm-bench，build.py / verify.py）编译并
    计时，配置与 campaign 一致；报告 median/best TFLOPS 与样本。
  - 一次只测一个变量；负面结果同样写入 research 文档 §8（避免重复实验）。
  - GPU 测试/benchmark 期间不得有第二个套件或第二个 benchmark 并发。
- 若有 `--focus`，focus 自己定义"本轮算完成"的证据（某个 test / parity / 指标），
  必须实际跑出来并在输出中引用；没有证据等于没完成。

E. 收尾（每轮都做）
- 运行 `/code-review low`（低强度，重点核心逻辑与套件 regression），再运行
  `/simplify`，应用必要发现；跳过的发现要说明原因（false positive / 超出 diff /
  行为变更）。
- 通过后 commit（**本仓库 commit-msg hook 拒绝 `Co-Authored-By`，不要加 attribution**；
  一个 logical change 一个 commit，message 用英文）。
- 失败或未通过验证：不 commit，输出阻塞点，不扩 scope。不要 `git reset --hard` /
  `git checkout -- .` 去"恢复 clean"：working tree 里可能有 developer 未提交的工作。
- 更新 memory：新的 commit 哈希、backlog 变化、性能结论。
- 清除 round lock。

F. 输出（简洁中文）
- 本轮做了什么（对应哪个 component / 哪个经验证的缺陷）
- F1–F7 状态
- 套件结果（实际命令与数字；未完成的验证如实标注）
- 下一轮 1–3 个候选（只列，不做）
- 是否 blocked

停止条件（只有这三种）：
1. 终止条件满足：F1–F7 全部完成 **且** 有 1800-class CroqTile kernel 与 1845 CUDA
   kernel 同配置对比得到真实 TFLOPS/gap（当前未满足，loop 继续）。
2. 环境阻塞：GPU 卡死且需要重启/GPU 重置，或机器资源不可用；报告后等待。
3. 候选需要产品/设计决策、新依赖，或与已记录的决策冲突；只报告，不扩 scope。

时间盒：每轮实现尽量控制在 ~8 分钟内，剩余时间用于验证与汇报；超过 10m 的 round
顺延到下一次 tick（lock 会保留给下一 tick resume）。
```

## Piano implementation

The same loop is implemented as a Piano workflow in the `piano` repository:
`recipes/workflows/croqtile-dev-loop.yml` (and its `-dsh` twin) runs
`loop rounds[ propose → implement → verify ]` with a serial round, up to 100
rounds by default (lower it for a short trial). Each round proposes one lit-verifiable change, implements it in the
shared checkout, then builds and runs the lit suites there; a failing round is
restored exactly to its pre-round files. If a campaign is interrupted, resume the same run instead of starting a new one: `piano -p --json --resume <run-id> "continue"` (headless continue) reloads the run's recorded workflow and continues its latest job from the accepted cursor. The gemm_sp end-to-end kernel
(`tests/gpu/end2end/gemm_sp.co`, not the whole e2e directory) runs in the
round only when a GPU is idle (utilization 0, low memory; polled every 30 s
with a bound), pinned with `CUDA_VISIBLE_DEVICES`, and never while another
suite or benchmark runs. The verify stage then commits the round with a
descriptive English message and pushes the branch before the engine's loop
auto-commit, which finds nothing left and logs a no-op — every round is a
durable, self-describing, pushed commit (a failed push keeps the local commit
and is reported, never force-pushed). There is no tournament: the engine's review-score referee is not a
measurement comparable across rounds. Scope is compiler changes whose
evidence is a build plus lit tests; kernel tuning stays in the
`gpu-croqtile-tuning` workflows.

## Notes

- **focus 是 prompt 的语义，不是新的 decision。** 加 `--focus` 不新增 docs 主题、
  不改完成判据，只替换"本轮选哪个 task"这一步的输入。没有 focus 时行为与之前一致。
- **性能与功能是同一个 loop 的两个阶段**，不要在一个 round 里同时做实验和正确性修复。
  纯 kernel A/B 走阶段 A 的协议（同配置、median/best、记录负面结果）。
- **不要重复已记录的负面实验**：先读 research 文档 §8 与 memory
  `spmm-campaign-benchmark.md`。
- **memory 是长期的**：commit、backlog 变化、性能结论写入
  `croqtile-local-llvm-toolchain.md`；campaign 产物与 ncu 结论写入
  `spmm-campaign-benchmark.md`。
- **提交约定**：仓库的 commit-msg hook 拒绝 `Co-Authored-By: Claude Code`；不要加
  attribution 行。
- **环境**：两台 H800；piano benchmark 可能占用 GPU。GPU 测试/benchmark 串行执行；
  详细规则与事故记录见下节。

## 环境规则（2026-11-08 GPU 卡死事故）

### 事故

两个 `./tests/lit.sh tests/` 实例并发执行（一个是被停止的后台任务遗留的孤儿套件，
一个是新启动的全量套件，两者都执行 `tests/gpu/end2end/cooperative_grid_sync.co`）。
该用例用 `cudaLaunchCooperativeKernel` 启动 2 block × 1 线程，`grid.sync` 降级为
`ATOM.E.ADD.STRONG.GPU` 到达 + **无超时** 的 `LD.E.STRONG.GPU/CCTL.IVALL/YIELD`
自旋 + `BAR.SYNC`。并发执行时 peer block 得不到推进，自旋 warp 永久驻留，CUDA 上下文
无法回收：`SIGKILL` 一直 pending（`/proc/<pid>/status` 的 `SigPnd: 0x100`），
`nvidia-smi` 挂起，同卡上的 piano benchmark 也被堵死。恢复只能重启或
`sudo nvidia-smi --gpu-reset`（需要无活跃上下文）。

另一卡死进程来自审查子代理并发做的 end2end 模拟——并发入口不只有主 loop。

### 规则

1. **跑测试前先查进程**：`pgrep -af "tests/lit.sh|spmm-bench|verify.py|profile.py|ncu "`，
   有残留就等待或清理；绝不启动第二个套件。
2. **停止套件要复查进程树**：`TaskStop` 之后 `pgrep` 确认 `lit.sh` 与
   `__choreo_*.exe` 都没了；包装进程退出不等于测试停止。
3. **GPU e2e 测试串行**：不要从两个地方跑 `tests/gpu/end2end`；piano benchmark 正在
   测量时避免同时跑 e2e。
4. **GPU 不可用时只跑编译面**：用 nvidia-smi shim 伪装设备（lit.cfg 用
   `nvidia-smi --query-gpu=name,compute_cap` 探测架构；shim 输出已知的
   `NVIDIA H800 PCIe, 9.0`，放在 `/tmp/fakebin/nvidia-smi` 并 `chmod +x`，运行时
   `PATH=/tmp/fakebin:$PATH`）。
5. **卡死的 GPU 进程不要反复 kill，也不要反复探测**：`R` 状态且 `SIGKILL` pending
   （或 `D` 驱动等待）且 `nvidia-smi` 挂起时，只能重启或 root GPU 重置。`nvidia-smi`
   探测必须单实例：若某次 `timeout 10 nvidia-smi` 超时后进程仍存活，立即停止探测
   并把 e2e 记为 `not-run`，绝不再起新的 `nvidia-smi`——D 状态的探测进程不可杀，
   叠起来会把驱动彻底堵死（2026-11-09 campaign 第 79 轮起即因此丢失 19 轮 e2e）。
6. **如实报告**：套件没跑完就写"未完成 + 原因 + 待补验证"，不要把部分结果当全绿。
7. **产物与日志**：需要静态分析时用 `choreo -gs` 生成 host 脚本/CUDA，再用
   `nvcc -ptx` / `-cubin` + `cuobjdump -sass` 离线分析（不需要 GPU）；产物放 `/tmp`。
