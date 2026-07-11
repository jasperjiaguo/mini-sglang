# Speculative-decoding improvement over spec-off

Throughput is the concurrency-normalized decode rate. Positive values mean
higher throughput or lower mean request TPOT than spec-off at the same
concurrency.

| Concurrency | N3/K2 throughput increase | N3/K2 mean TPOT decrease | N3/K3 throughput increase | N3/K3 mean TPOT decrease |
|---:|---:|---:|---:|---:|
| 8 | +27.8% | +22.0% | +33.0% | +25.1% |
| 16 | +19.7% | +16.2% | +24.3% | +19.4% |
| 24 | +19.6% | +16.3% | +18.4% | +15.4% |
| 32 | +16.1% | +13.3% | +22.4% | +17.4% |
| 40 | +15.2% | +12.5% | +11.1% | +9.3% |
| 48 | +10.7% | +8.8% | +6.0% | +5.1% |
| 56 | +7.1% | +5.6% | +1.3% | +0.5% |
| 64 | +3.9% | +3.2% | -3.3% | -4.0% |
| 96 | -6.4% | -7.9% | -9.5% | -11.6% |
| 128 | -2.7% | -3.1% | -8.8% | -10.0% |
