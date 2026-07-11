# Aggregate n-gram acceptance

Rates after position zero are conditional on every earlier draft position being
accepted. Counts are cumulative across measured and warmup requests.

| Mode | Draft position | Accepted / attempted | Conditional acceptance |
|---|---:|---:|---:|
| n3_k2 | p0 | 1220 / 3743 | 32.59% |
| n3_k2 | p1 | 688 / 1218 | 56.49% |
