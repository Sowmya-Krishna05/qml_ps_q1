# PS-Q2 Feature Dictionary — QML-HIGHD-KER

**Source:** Synthetically generated.  
**Generation:** 6D non-linear latent structure embedded into 50D via random orthogonal projection + Gaussian noise. Standardised to zero mean / unit variance. Decision boundary is non-linearly separable.

## Feature columns

| Columns | Description |
|---------|-------------|
| x01–x50 | Standardised numerical features. x01–x06 carry most signal; x07–x50 are predominantly noise. |

## Label column

| Value | Meaning |
|-------|---------|
| A | Class A (latent score ≤ median) |
| B | Class B (latent score > median) |

## Files

| File | Samples | Labels |
|------|---------|--------|
| hd_data_train.csv | 800 | Yes |
| hd_data_test_features.csv | 200 | No — hidden. First column `sample_id` (s0 … s199); it is an identifier, not a feature |

**Class distribution:** ~50/50.  
**Official metric:** Balanced accuracy. F1 reported for reference.

## Feasibility note

Full quantum kernel matrix for 800 samples = 320,000 circuit evaluations. Reduce to 100–200 samples or apply PCA first. Document any reduction.
