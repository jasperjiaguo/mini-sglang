# Aggregate n-gram acceptance

Rates after position zero are conditional on every earlier draft position being
accepted. Counts are cumulative across measured and warmup requests.

| Mode | Draft position | Accepted / attempted | Conditional acceptance |
|---|---:|---:|---:|
| n3_k2 | p0 | 174299 / 608672 | 28.64% |
| n3_k2 | p1 | 90175 / 173240 | 52.05% |
| n3_k3 | p0 | 159337 / 585232 | 27.23% |
| n3_k3 | p1 | 80658 / 158395 | 50.92% |
| n3_k3 | p2 | 47391 / 80088 | 59.17% |
