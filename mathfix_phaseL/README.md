# Phase L

Is arm-`b` GSM8K format damage localized in a few LoRA layers?

- Pre-registration (locked before training / test eval): `PREREG_phaseL.md`
- Driver: `phaseL.py` (imports unchanged `mathfix/common.py`)
- LoRA-delta zeroing: `zeroing.py`
- Tiny-model unit test: `tests_zeroing.py`
- Results: `../results/phaseL/` and `../RESULTS_phaseL.md`

```
python3 tests_zeroing.py
python3 phaseL.py --pilot          # timing + N; commit before test eval
python3 phaseL.py                  # full experiment (reads frozen N)
MATHFIX_SMOKE=1 python3 phaseL.py --smoke
```
