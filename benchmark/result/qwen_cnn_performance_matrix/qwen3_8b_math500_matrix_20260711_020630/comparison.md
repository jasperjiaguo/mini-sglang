# Speculative-decoding improvement over spec-off

Throughput is the concurrency-normalized decode rate. Positive values mean
higher throughput or lower mean request TPOT than spec-off at the same
concurrency.

| Concurrency | N3/K2 throughput increase | N3/K2 mean TPOT decrease | N3/K3 throughput increase | N3/K3 mean TPOT decrease |
|---:|---:|---:|---:|---:|
| 8 | +28.6% | +22.2% | +32.8% | +24.7% |
| 16 | +28.9% | +22.4% | +31.3% | +23.8% |
| 24 | +27.4% | +21.5% | +30.1% | +23.1% |
| 32 | +5.1% | +4.9% | +28.3% | +22.0% |
| 40 | +19.5% | +16.3% | +20.5% | +17.0% |
| 48 | +15.2% | +13.2% | +22.0% | +18.1% |
| 56 | +17.0% | +14.5% | +18.7% | +15.8% |
| 64 | +23.4% | +18.9% | +23.2% | +18.8% |
| 96 | +14.1% | +12.4% | +12.1% | +10.8% |
| 128 | +7.5% | +7.0% | +3.2% | +3.1% |
