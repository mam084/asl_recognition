# ASL Sign Recognition from Video (WLASL)

Word-level American Sign Language recognition from raw video using PyTorch. The project compares a frame-level CNN with a CNN-LSTM that models motion across frames, and tests how much accuracy depends on which signers the model has already seen.

*Team project (Franky Liu, Jaron Zhu, Matthew Mitchell) for the UC San Diego M.S. in Data Science program.*

**[Full report (PDF)](asl_recognition_report.pdf)** · **[Notebook](asl_recognition_notebook.ipynb)**

## Results

| Model | Split | Bounding-box crop | Train accuracy | Validation top-1 |
|---|---|---|---|---|
| CNN | Random | No | 97.9% | **60.5%** |
| CNN | Signer-independent | No | 96.8% | 52.8% |
| CNN | Signer-independent | Yes | 96.9% | 53.6% |
| CNN-LSTM | Signer-independent | Yes | 91.6% | 54.2% |

- **Unseen signers are the real test.** With a random split, the same people appear in training and validation, and accuracy is about 8 points higher than when each signer appears in only one split. The signer-independent number is the honest estimate of performance on new users.
- **Cropping to the signer helps a little.** Focusing frames on the hands and upper body raised signer-independent accuracy from 52.8% to 53.6%.
- **Temporal modeling generalizes better.** The CNN-LSTM had the smallest train–validation gap and the best signer-independent accuracy, since many signs share hand shapes and differ mainly in motion.

## Approach

1. **Data pipeline.** Parse the WLASL annotation JSON into a table with one row per video instance (gloss, signer, frame range, bounding box), drop unavailable videos, and verify every remaining file exists locally. The project switched to a complete WLASL mirror after finding 9,103 missing videos in the original release.
2. **Exploratory analysis.** Gloss frequency and class imbalance, clip durations versus frame counts, signers per gloss, bounding-box coverage, and duplicate/leakage checks.
3. **Frame sampling and cropping.** Sample a fixed number of frames uniformly across each clip, crop to the annotated bounding box (or a fallback upper-body region), and resize to 224×224.
4. **Models.** A CNN classifier on frame features, and a CNN-LSTM in which the CNN extracts per-frame features and the LSTM models their sequence.
5. **Evaluation.** Top-1 accuracy under random and signer-independent splits, with and without cropping.

## Data

[WLASL](https://dxli94.github.io/WLASL/) (Li et al., 2020): 21,083 labeled clips of word-level signs performed by many different signers. The videos are too large for this repository.

To run the notebook, place the dataset like this (or edit `JSON_PATH` and `VIDEO_DIR_CANDIDATES` near the top of the notebook):

```
dataset/
├── WLASL_v0.3.json
├── missing.txt
└── videos/
```

## Run

```bash
pip install -r requirements.txt
jupyter notebook asl_recognition_notebook.ipynb
```

A GPU is recommended for training.

## Reference

Li, D., Rodriguez Opazo, C., Yu, X., & Li, H. (2020). Word-level Deep Sign Language Recognition from Video: A New Large-scale Dataset and Methods Comparison. *WACV*.
