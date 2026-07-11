# Speculative-decoding improvement over spec-off

Throughput is the concurrency-normalized decode rate. Positive values mean
higher throughput or lower mean request TPOT than spec-off at the same
concurrency.

| Concurrency | N3/K2 throughput increase | N3/K2 mean TPOT decrease | N3/K3 throughput increase | N3/K3 mean TPOT decrease |
|---:|---:|---:|---:|---:|
| 8 | +40.5% | +29.2% | +39.5% | +28.7% |
| 16 | +14.4% | +13.8% | +24.0% | +19.1% |
| 24 | +20.2% | +16.4% | +16.7% | +13.8% |
| 32 | +17.2% | +14.1% | +13.8% | +12.6% |
| 40 | +14.9% | +12.5% | +15.9% | +13.2% |
| 48 | +15.6% | +13.0% | +19.2% | +15.5% |
| 56 | +15.6% | +13.3% | +16.9% | +14.1% |
| 64 | +16.7% | +14.0% | +10.7% | +9.5% |
