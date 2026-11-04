`visionfm_upstream.py.txt` is the unmodified `models/vision_transformer.py` from
ABILab-CUHK/VisionFM commit `648ab0895f8e80e2c937f658bd2f1629ea6d27e7`.
It is retained solely for offline numerical parity tests. Copyright and license:
CC BY-NC 4.0, see `mario_dino/models/vendor/VisionFM-LICENSE`.
The test isolates dependency imports, disables FlashAttention, and uses random small weights.

The production adapter and checkpoint contract are documented in
[docs/encoders.md](../../docs/encoders.md). This fixture is not a pretrained checkpoint
or a runtime dependency. The new parity tests were written for later cloud/CI execution;
their presence does not indicate that they or the released OCT weights have been run.
