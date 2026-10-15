## Trigger recovery (ToxScreen 1B testbed)

| Objective | Family | N | recovery@top20 | recovery@rank1 | behavioural (margin>=10pp) |
|---|---|---|---|---|---|
| classifier | semantic | 1 | 100% | 100% | 100% |
| classifier | single_token | 1 | 100% | 100% | 100% |
| refusal | emoji | 5 | 80% | 80% | 80% |
| refusal | multi_token | 1 | 0% | 0% | 0% |
| refusal | paraphrase | 1 | 0% | 0% | 0% |
| refusal | semantic | 3 | 33% | 33% | 33% |
| refusal | single_token | 6 | 0% | 0% | 0% |
| sentiment | multi_token | 1 | 0% | 0% | 0% |
| sentiment | semantic | 2 | 0% | 0% | 0% |
| sentiment | single_token | 3 | 0% | 0% | 0% |
| **all** | — | 24 | **29%** | **29%** | **29%** |

Clean-model false-positive rate (behavioural gate margin>=10.0pp): **100%** (2/2 clean models flagged).

Cost per model: median 18.8 min, peak GPU memory 4.1 GB (RTX 4050, 6 GB).
