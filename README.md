# mario-dino

MARIO challenge Tasks 1 and 2 with a frozen DINOv2 encoder, a GRU that forecasts next-visit features, and
one fusion head shared by both tasks. DF41's code ([pzhangwj/mario_challenge_code](https://github.com/pzhangwj/mario_challenge_code),
pinned at `4d4a11b` in `third_party/`) supplies OCTIP preprocessing, scoring and the output format.

- **Task 1**: `head(f_t, f_{t+Δt})`
- **Task 2**: `head(f_t, forecaster(f_t, Δt₀))[:, :3]`

```bash
git clone --recursive <this repo> && cd mario-dino
pip install -r requirements/main.txt
make test && make smoke      # CPU, synthetic data, no downloads
```

See [RUNBOOK.md](RUNBOOK.md) for the real-data sequence (one GPU job), Docker, the run status and the open
questions.
