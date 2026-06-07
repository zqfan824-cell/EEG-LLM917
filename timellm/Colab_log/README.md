# Colab_log — 实验日志约定

这个文件夹存放在 Colab（H100）上跑出来的训练日志。约定的目标：**文件名一眼能扫，
日志头能完整溯源到「哪份代码、哪天、什么配置」。**

工作流（已和 Claude 约定）：
**Claude 改代码 / 开 PR → 你在 Colab `git pull` → 一键脚本跑 + 自动推日志回 git →
Claude `git fetch` 直接读日志 → 再迭代。** 不走浏览器自动化，日志只走 git。

---

## 1. 文件命名约定（结构化）

```
<YYYYMMDD-HHMM>__<数据-任务>__<模型-LLM>__<标签>__g<git短哈希>.log
```

例：

```
20250607-1432__DEAP-val2__VQ-GPT2__focal__ge68b5b0.log
```

| 字段 | 含义 | 取值示例 |
|---|---|---|
| `YYYYMMDD-HHMM` | 运行开始的日期时间（可排序） | `20250607-1432` |
| `数据-任务` | 数据集 + 任务 | `DEAP-val2`(valence二分类) / `DEAP-aro2` / `DEAP-4cls` / `SEED-3cls` |
| `模型-LLM` | 模型变体 + 骨干 LLM | `VQ-GPT2` / `VQ-LLAMA` / `base-GPT2`（base=EEGLLM 无 VQ） |
| `标签` | 这次改了什么 / 实验意图 | `focal` / `baseline` / `dw1.0` / `lr5e-5` / `no-recon` |
| `g<git短哈希>` | 产出这份日志的代码版本 | `ge68b5b0` |

- 分隔符：大字段之间用 `__`（双下划线），字段内部用 `-`，避免和模型名里的 `_` 混淆。
- `标签` 是给人看的一句话摘要，自由发挥，但尽量短、只点关键改动。
- **决定性字段是 `g<git短哈希>`**：Claude 据此 100% 确定是哪份代码跑的。

> 想表达更多细节不要硬塞文件名，塞进下面的「日志头」。

---

## 2. 日志头（RUN META）

每份日志开头自动写一段（由 `run_and_log.sh` 生成）：

```
# ===== RUN META =====
# date:    2025-06-07 14:32:10 UTC
# git:     e68b5b0 (main)
# host:    xxxxxxx / GPU: NVIDIA H100 80GB HBM3
# slug:    DEAP-val2__VQ-GPT2__focal
# script:  run_deap_vq.sh
# =====================
```

结尾自动追加 `# exit_status: <码>`（0=正常，非 0=崩了），方便一眼看出是否跑完。

---

## 3. 一键运行（Colab cell）

在 Colab 里粘一个 cell（路径按你的实际 clone 位置改）：

```bash
!cd /content/EEG-LLM917/timellm && git pull --ff-only \
  && bash Colab_log/run_and_log.sh "DEAP-val2__VQ-GPT2__focal"
```

`run_and_log.sh` 会自动：①生成结构化文件名 ②写日志头 ③跑实验并 tee 进日志
④跑完 `commit + pull --rebase + push` 把日志推回 `origin/<当前分支>`。

- 第一个参数就是上表的 **slug**（`数据-任务__模型-LLM__标签`），日期和 git 哈希脚本自己补。
- 第二个参数可选，指定要跑的脚本，默认 `run_deap_vq.sh`。
- 只想本地存、先不推：`PUSH=0 bash Colab_log/run_and_log.sh "..."`。

> ⚠️ 自动 push 需要 Colab 端配好 git 推送凭证（GitHub token 或 SSH）。
> 还没配的话脚本会把日志写在本地、提示「push 失败」，不影响实验本身。

---

## 4. 历史日志

- `917log.txt` —— 约定之前的旧日志（DEAP / VQ-GPT2 / focal，代码版本约 `e68b5b0`，
  运行日期未知）。保留作参考，不强行套新命名。

---

## 5. 待确认（影响自动 push 是否能用）

- Colab 端 `git push` 用的 GitHub 账号 / 凭证是否已配？
- 日志推到 `main` 还是单独的 `results` 分支？（当前脚本推到当前分支；新日志文件名唯一，
  一般不会和代码 PR 冲突。）
