# Residual Persistence and Half-Life Report

## Data Quality and Retention
underlying  n_total  n_retained  retention_rate
    SSE 50     2896         674        0.232735
   CSI 300     4839        1529        0.315974
  CSI 1000     5300        1795        0.338679

## Half-Life Distribution
count    3998.000000
mean        2.307829
std        10.934885
min         0.424606
25%         0.887018
50%         1.204507
75%         1.823746
max       393.633254

## Median Half-Life Table
Median Half-Life (Trading Days) across Indices and Moneyness-Tenor Buckets

                                SSE 50                                   CSI 300                                   CSI 1000
Moneyness           ST            MT            LT            ST            MT            LT            ST            MT            LT
--------------------------------------------------------------------------------------------------------------------------------------
DPW              1.67 (2)      0.76 (3)      0.64 (2)      1.52 (8)     0.99 (15)     1.07 (16)     1.18 (74)     1.26 (54)     1.34 (42)
PW              1.43 (79)     0.85 (46)     1.25 (16)     1.13 (113)    0.98 (138)    1.09 (30)     1.12 (176)    1.02 (146)    0.94 (30)
ATM             1.21 (154)    0.96 (96)     2.13 (14)     1.19 (236)    0.97 (254)    1.19 (44)     1.30 (246)    0.95 (194)    1.29 (28)
CW              1.58 (119)    1.19 (95)     1.62 (12)     1.34 (264)    1.22 (264)    1.31 (58)     1.44 (304)    1.38 (250)    1.30 (48)
DCW             1.20 (10)     1.21 (10)     2.72 (16)     1.31 (17)     1.35 (38)     1.30 (34)     1.27 (68)     1.06 (59)     1.60 (76)
--------------------------------------------------------------------------------------------------------------------------------------
All moneyness   1.39 (364)    1.01 (250)    1.66 (60)     1.27 (638)    1.05 (709)    1.22 (182)    1.28 (868)    1.11 (703)    1.35 (224)
--------------------------------------------------------------------------------------------------------------------------------------
Retention rate             23.3% (674/2896)                         31.6% (1529/4839)                         33.9% (1795/5300)

## Tenor Median Half-Life
tenor_bucket      ST      MT      LT
underlying                          
SSE 50        1.3860  1.0127  1.6586
CSI 300       1.2653  1.0471  1.2221
CSI 1000      1.2761  1.1057  1.3531

## Factorial ANOVA
                                          sum_sq      df       F  PR(>F)
C(underlying)                           286.1699     2.0  1.2045  0.3000
C(moneyness_bucket)                    1057.4963     4.0  2.2255  0.0638
C(tenor_bucket)                        1013.9145     2.0  4.2675  0.0141
C(underlying):C(moneyness_bucket)      1739.3940     8.0  1.8303  0.0668
C(underlying):C(tenor_bucket)           590.5594     4.0  1.2428  0.2904
C(moneyness_bucket):C(tenor_bucket)    2126.0504     8.0  2.2371  0.0222
Residual                             471493.5700  3969.0     NaN     NaN

## Key Numbers for Thesis Text
Overall retained sample median half-life: 1.20 trading days.
Overall retained sample IQR: 0.89-1.82 trading days.
Overall retained sample maximum: 393.63 trading days.
Total regressions: 13035.
Retained regressions: 3998.