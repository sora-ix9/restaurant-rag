# Memory Measurement — Index Benchmark

> ## 📍 Scope
> | | |
> |---|---|
> | **Question** | How does the index benchmark measure RAM, and how do its memory columns differ? |
> | **Covers** | The memory cycle, step by step, and the five memory columns in [`benchmark_index_types.md` §6.4](benchmark_index_types.md#64-memory) |
> | **Source** | The `load_and_measure` helper and the **build + measure** cell in [`notebook/benchmark_index_types.ipynb`](../../../notebook/benchmark_index_types.ipynb) |
> | **Numbers from** | Full run `index-types-full-20261105-1108`, 2026-11-05 |

**Written:** 2026-11-06 (Asia/Bangkok, UTC+7)
**Author:** claude-opus-5-5 (Claude Code)

All sizes are in MB, where 1 MB is 1,048,576 bytes, as the notebook counts.

---

## Table of Contents

1. [The memory cycle](#1-the-memory-cycle)
2. [What the sweep is](#2-what-the-sweep-is)
3. [The five memory columns](#3-the-five-memory-columns)
4. [Steady against peak](#4-steady-against-peak)
5. [After the sweep](#5-after-the-sweep)
6. [Which column to use](#6-which-column-to-use)

---

## 1. The memory cycle

Each index is built, then measured through a number of memory cycles: 3 in a full run, 1 in a smoke run. One cycle reads RAM before, during and after loading the collection, starting from a freshly restarted Milvus.

```
release → restart Milvus → settle → IDLE ──load──→ settle → STEADY
                                          ↑
                                 sampled throughout: PEAK
```

| Step | What happens | Why |
|---|---|---|
| 1. Release | The bench collection is released | Milvus reloads whatever was loaded when it restarts, so the collection must be released first |
| 2. Restart | `docker restart data_server_milvus`, then wait until its health check answers and it lists collections | Milvus's Go runtime keeps memory it has freed. Releasing alone would leave the previous index's footprint inside the next baseline |
| 3. Settle | Wait 30 s (5 s in a smoke run) | A server that has just restarted is still settling, so its baseline would read low |
| 4. **Idle** | Median of 3 `docker stats` readings | The empty server's memory. Every other column except `memory.peak` is measured against it |
| 5. Load, sampled | `load_collection` runs while a background thread takes `docker stats` readings back to back | To catch the extra memory loading needs while it happens |
| 6. Settle | Wait 30 s again | Lets temporary load memory be freed before the steady reading |
| 7. **Steady** | Median of 3 `docker stats` readings | What the loaded index costs while it sits in memory |
| 8. **Peak** | The highest of the samples from step 5 and the steady value | The worst moment that sampling saw |
| 9. **`memory.peak`** | Read from `/sys/fs/cgroup/memory.peak` inside the container | The container's own high-water mark, which sampling cannot miss |
| 10. Restart check | Milvus's start time must be unchanged since step 2 | An unplanned restart, for example after the PC sleeps, would invalidate the readings, so the run stops |

**How the cycles combine into one row:**

| Column | From the 3 cycles |
|---|---|
| Steady, median | Median of the 3 steady − idle values. This is the RAM figure in the head-to-head |
| Steady, min and max | Lowest and highest of those 3, to show how repeatable the reading is |
| Peak during load, sampled | Highest of the 3 peak − idle values, because this column is about the worst case |
| `memory.peak` | Highest of the 3 readings |

For example, `IVF_FLAT-nlist800` read +258, +262 and +260 MB over its 3 cycles, so its steady median is 259.7 MB, with a minimum of 258.1 and a maximum of 262.4.

**The last cycle doesn't release the collection.** It stays loaded for the sweep, and memory is read once more afterwards ([§5](#5-after-the-sweep)).

---

## 2. What the sweep is

Each index has one setting that trades accuracy for speed: `ef` for `HNSW`, and `nprobe` for `IVF_FLAT` and `IVF_PQ`. The sweep runs the index at every value of that setting, for example `ef` = 10, 16, 32, 64, 128, 256 and 512. `FLAT` and `AUTOINDEX` have one point each.

At each value the sweep runs:

| Pass | Searches |
|---|---|
| Recall | All 6,195 held-out reviews, untimed |
| Latency | 1,000 sampled reviews, twice, after a discarded warm-up pass |
| QPS | 15 s each at 1, 4 and 8 concurrent clients |

So by the end of the sweep, the index has answered many thousands of searches.

---

## 3. The five memory columns

| Column | When it's read | Measured from | Question it answers |
|---|---|---|---|
| **Steady** | After loading and a 30 s settle, before any query | Idle | What does the index cost just sitting in memory? |
| **Peak during load, sampled** | During loading, by back-to-back `docker stats` readings | Idle | Does loading briefly need more than steady? |
| **`memory.peak`** | Once per cycle, after the steady reading | Zero | Same question, without missing short spikes |
| **After the sweep** | After every search for that index has run, and a 30 s settle | The last cycle's idle | Does serving queries use more than holding the index? |
| Idle | After the restart and a 30 s settle | Zero | The baseline. It isn't reported as its own column in the doc. It was about 100–110 MB in this run |

**Example, `HNSW-M16-efc200`:** steady 254.5 MB, sampled peak 390.5 MB, `memory.peak` 880.1 MB, after the sweep 304.6 MB.

---

## 4. Steady against peak

Loading needs temporary memory beyond what the index keeps, for example buffers while index files are read in. That memory is freed afterwards, so a peak sits above steady.

**The sampled peak can miss the spike.** Each `docker stats` reading takes a second or two, and loads took 1.3–1.7 s in this run, so a spike can fall between readings. When no sample beats the steady value, the peak *is* the steady value. That's why both `IVF_PQ` rows show the same number, 35.1 and 35.0 MB, for peak and for the highest steady reading.

**`memory.peak` can't miss a spike, but it overstates.** Two things inflate it:

| Inflation | Why |
|---|---|
| **It counts from zero** | It includes the idle server, about 100 MB, which the other columns subtract |
| **It counts file cache** | The kernel's cache of files read during the load is charged to the container. `docker stats` leaves that cache out |

So treat `memory.peak` as an upper bound, not as what the index needs. It does reset with each restart: it fell from 918.8 MB for `AUTOINDEX` to 158.2 MB for `IVF_PQ-m96`, which it couldn't do if it carried over between configs.

---

## 5. After the sweep

The steady reading is taken before any query, so it can't show what searching costs. Serving queries allocates working memory, such as search buffers and result handling, and Milvus doesn't hand all of it back right away. So once the sweep finishes, the notebook waits 30 s, takes the median of 3 readings, and subtracts the last cycle's idle.

The rise from steady to after the sweep is similar for every index:

| Config | Steady | After the sweep | Rise |
|---|---|---|---|
| `FLAT` | 252.6 | 288.5 | +36 |
| `AUTOINDEX` | 256.9 | 295.7 | +39 |
| `HNSW-M16-efc200` | 254.5 | 304.6 | +50 |
| `IVF_FLAT-nlist800` | 259.7 | 311.3 | +52 |
| `IVF_PQ-nlist800-m96` | 26.4 | 76.9 | +51 |
| `IVF_PQ-nlist800-m192` | 31.5 | 82.9 | +51 |

Because the rise is roughly the same whatever the index, it looks like a cost of serving queries rather than of a particular index.

**It is less repeated than steady.** Steady is a median across 3 cycles. After the sweep comes from the last cycle only, so it has no min or max to show its spread.

---

## 6. Which column to use

| To answer | Use | Why |
|---|---|---|
| Which index costs more RAM? | **Steady, median** | Repeated over 3 cycles, and read before queries add their own, similar cost |
| How much RAM will a running server need? | **After the sweep** | Closest to a server that has been answering traffic |
| Could loading run out of memory, for example after a restart? | **`memory.peak`** | It can't miss a spike. Remember that it includes idle and file cache |
| — | Not the sampled peak | It misses short spikes, as both `IVF_PQ` rows show |

All of these are the index's cost, not the deployed collection's. The bench collection holds only `id` and `vector`, while the live one also stores `text`, `source` and `meta` ([`benchmark_index_types.md` §8](benchmark_index_types.md#8-known-limits)).
