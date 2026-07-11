# Aggregate n-gram acceptance

Rates after position zero are conditional on every earlier draft position being
accepted. Counts are cumulative across measured and warmup requests.

| Mode | Draft position | Accepted / attempted | Conditional acceptance |
|---|---:|---:|---:|
| n3_k2 | p0 | 9788 / 29863 | 32.78% |
| n3_k2 | p1 | 5599 / 9776 | 57.27% |
| n3_k3 | p0 | 8879 / 28439 | 31.22% |
| n3_k3 | p1 | 5079 / 8869 | 57.27% |
| n3_k3 | p2 | 2938 / 5072 | 57.93% |
