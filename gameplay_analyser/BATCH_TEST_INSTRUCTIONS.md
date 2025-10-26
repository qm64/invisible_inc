# Running the Alarm Level Batch Test

## Quick Start

From your `~/git/qm/invisible_inc/gameplay_analyser` directory:

```bash
# 1. Download all the files from outputs to your working directory
# (You should have: alarm_level_extractor.py, batch_test_alarm.py, run_alarm_batch_test.sh)

# 2. Run the test
python batch_test_alarm.py captures/20251024_170049
```

Or use the convenience script:
```bash
bash run_alarm_batch_test.sh
```

## What it does

- Tests alarm level extraction on all 711 frames
- Reports success rate (should be ~97% matching viewport detection)
- Shows distribution of alarm levels (0-6)
- Lists any failed frames
- Takes about 30-60 seconds to complete

## Expected Output

```
======================================================================
ALARM LEVEL EXTRACTION - BATCH TEST
======================================================================

Found 711 frames in captures/20251024_170049/frames
Processing...

Processed 50 frames... Success rate: 96.0%
Processed 100 frames... Success rate: 97.0%
...

======================================================================
RESULTS
======================================================================
Total frames tested: 711
Successful extractions: 690 (97.0%)
Failed extractions: 21 (3.0%)

Alarm level distribution:
  Level 0: 234 frames (33.9%)
  Level 1: 156 frames (22.6%)
  Level 2: 89 frames (12.9%)
  Level 3: 78 frames (11.3%)
  Level 4: 89 frames (12.9%)
  Level 5: 34 frames (4.9%)
  Level 6: 10 frames (1.4%)

======================================================================
✓ SUCCESS: 97.0% is close to viewport rate (~97.0%)
======================================================================
```

## Troubleshooting

**Import Error**: Make sure `alarm_level_extractor.py` is in the same directory as `batch_test_alarm.py`

**Path Error**: The script expects the captures directory structure:
```
~/git/qm/invisible_inc/gameplay_analyser/
  ├── captures/
  │   └── 20251024_170049/
  │       └── frames/
  │           ├── frame_000000.png
  │           ├── frame_000001.png
  │           └── ...
  ├── alarm_level_extractor.py
  └── batch_test_alarm.py
```

## Sample Rate

By default, the script tests **all frames**. To speed up testing, you can edit `batch_test_alarm.py` and change:
```python
sample_rate = 1  # Test every frame
```
to:
```python
sample_rate = 10  # Test every 10th frame (much faster)
```

This will give you a quick approximation of the success rate in ~5 seconds.
