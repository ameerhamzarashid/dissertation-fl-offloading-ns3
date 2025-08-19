# Experiment Summary

## Configs
- Baseline: experiments/baseline_dense
- Enhanced: experiments/enhanced_sparse

## KPIs (training summary snapshot)
| config   |   reward |   latency_ms |   deadline_ok |
|:---------|---------:|-------------:|--------------:|
| baseline |     -0.6 |           32 |          0.85 |
| enhanced |     -0.6 |           32 |          0.85 |

## Replay aggregates (from ns-3 mec_client)
| config   |   mean_J |    p50_J |   p95_J |   mean_T_local |
|:---------|---------:|---------:|--------:|---------------:|
| baseline |  1.77254 | 0.916651 | 5.57912 |           0.15 |
| enhanced |  1.59701 | 0.671227 | 5.57916 |           0.15 |

Generated plots: reward.png, latency_ms.png, deadline_ok.png, replay_J.png