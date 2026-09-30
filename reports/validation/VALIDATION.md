# Pipeline validation on synthetic data (known answer)

Overall: **CHECK FAILED** (22/23 checks)

Three synthetic worlds with the real design (22 covariates, region-wise N missingness, spatial blocks). *relation* (positive control): N = OC / r(x) and CEC = a(x) clay + 0.35 OC hold up to 5% lab noise; calcareous profiles break the CEC relation. *shared* (diagnostic): no relation, but N and CEC share drivers with OC and clay. *independent* (negative control): N and CEC depend on drivers orthogonal to those of clay, OC and pH. Contrasts are CRPS skill differences (mean of N and CEC; positive = first arm better) under group thinning (scheme 'all' at 100%), 3 spatial folds, fold-level 95% t-intervals.

Learner settings: qrf {'n_estimators': 200}; tabm {'device': 'cpu', 'k': 8, 'd_block': 128, 'n_blocks': 2, 'max_epochs': 60, 'patience': 8}

| scenario                                 | learner   | check                                                      | expect    |   value | ci               | pass   |
|:-----------------------------------------|:----------|:-----------------------------------------------------------|:----------|--------:|:-----------------|:-------|
| positive (relation holds)                | qrf       | structured vs free at 3%                                   | > 0.0     |   0.279 | [+0.022, +0.536] | True   |
| positive (relation holds)                | qrf       | structured vs chained at 3%                                | > 0.0     |   0.206 | [+0.100, +0.312] | True   |
| positive (relation holds)                | qrf       | structured_ya vs chained at 3%                             | > 0.0     |   0.204 | [+0.102, +0.306] | True   |
| positive (relation holds)                | qrf       | chained vs free at 3%                                      | > 0.0     |   0.073 | [-0.078, +0.225] | True   |
| positive (relation holds)                | qrf       | structured vs chained at 100%                              | > -0.02   |   0.005 | [-0.017, +0.027] | True   |
| positive (relation holds)                | qrf       | oracle_ya skill >= structured skill at 3%                  | >= 0      |   0.167 |                  | True   |
| positive (relation holds)                | tabm      | structured vs free at 3%                                   | > 0.0     |   0.289 | [+0.224, +0.354] | True   |
| positive (relation holds)                | tabm      | structured vs chained at 3%                                | > 0.0     |   0.176 | [+0.116, +0.236] | True   |
| positive (relation holds)                | tabm      | structured_ya vs chained at 3%                             | > 0.0     |   0.165 | [+0.017, +0.314] | True   |
| positive (relation holds)                | tabm      | chained vs free at 3%                                      | > 0.0     |   0.113 | [+0.009, +0.217] | True   |
| positive (relation holds)                | tabm      | structured vs chained at 100%                              | > -0.02   |   0.009 | [-0.014, +0.033] | True   |
| positive (relation holds)                | tabm      | oracle_ya skill >= structured skill at 3%                  | >= 0      |   0.135 |                  | True   |
| relation                                 | qrf       | free QRF 90% coverage                                      | 0.80-0.97 |   0.858 |                  | True   |
| diagnostic (shared drivers, no relation) | qrf       | structured vs chained at 3% (borrowing without a relation) | report    |   0.127 | [-0.018, +0.272] | True   |
| diagnostic (shared drivers, no relation) | tabm      | structured vs chained at 3% (borrowing without a relation) | report    |   0.112 | [-0.056, +0.281] | True   |
| shared                                   | qrf       | free QRF 90% coverage                                      | 0.80-0.97 |   0.897 |                  | True   |
| negative (independent drivers)           | qrf       | structured_ya vs chained at 3%                             | <= 0.01   |  -0.095 | [-0.157, -0.034] | True   |
| negative (independent drivers)           | qrf       | structured vs chained at 3%                                | <= 0.01   |  -0.108 | [-0.199, -0.018] | True   |
| negative (independent drivers)           | qrf       | structured vs chained at 10%                               | <= 0.01   |  -0.069 | [-0.167, +0.029] | True   |
| negative (independent drivers)           | tabm      | structured_ya vs chained at 3%                             | <= 0.01   |   0.02  | [-0.285, +0.325] | False  |
| negative (independent drivers)           | tabm      | structured vs chained at 3%                                | <= 0.01   |  -0.013 | [-0.545, +0.519] | True   |
| negative (independent drivers)           | tabm      | structured vs chained at 10%                               | <= 0.01   |  -0.108 | [-0.175, -0.041] | True   |
| independent                              | qrf       | free QRF 90% coverage                                      | 0.80-0.97 |   0.895 |                  | True   |


**Checks that did not pass** (rule unchanged; interpretation added):

- tabm, negative (independent drivers), structured_ya vs chained at 3%: +0.020 [-0.285, +0.325]: the interval includes zero, so there is no evidence of a spurious gain; with 3 folds and this learner the check is too noisy to pass on the point estimate.

## relation: skill differences (mean of N and CEC)

|                                      |   0.03 |    0.1 |    1.0 |
|:-------------------------------------|-------:|-------:|-------:|
| ('qrf', 'chained vs free')           |  0.073 |  0.036 | -0.009 |
| ('qrf', 'structured vs chained')     |  0.206 |  0.038 |  0.005 |
| ('qrf', 'structured vs clip')        |  0.163 |  0.037 | -0.002 |
| ('qrf', 'structured vs free')        |  0.279 |  0.074 | -0.004 |
| ('qrf', 'structured vs ptf')         |  0.017 |  0.123 |  0.228 |
| ('qrf', 'structured vs ptf_marg')    |  0.009 |  0.11  |  0.211 |
| ('qrf', 'structured_ya vs chained')  |  0.204 |  0.037 |  0.002 |
| ('tabm', 'chained vs free')          |  0.113 |  0.133 | -0.08  |
| ('tabm', 'structured vs chained')    |  0.176 |  0.018 |  0.009 |
| ('tabm', 'structured vs clip')       |  0.184 |  0.105 | -0.065 |
| ('tabm', 'structured vs free')       |  0.289 |  0.151 | -0.071 |
| ('tabm', 'structured vs ptf')        |  0.004 |  0.195 |  0.28  |
| ('tabm', 'structured vs ptf_marg')   | -0.004 |  0.183 |  0.263 |
| ('tabm', 'structured_ya vs chained') |  0.165 | -0.016 |  0.004 |

## shared: skill differences (mean of N and CEC)

|                                      |   0.03 |   0.1 |    1.0 |
|:-------------------------------------|-------:|------:|-------:|
| ('qrf', 'chained vs free')           | -0.001 | 0.012 | -0.006 |
| ('qrf', 'structured vs chained')     |  0.127 | 0.068 | -0.019 |
| ('qrf', 'structured vs clip')        |  0.105 | 0.05  | -0.018 |
| ('qrf', 'structured vs free')        |  0.127 | 0.08  | -0.024 |
| ('qrf', 'structured vs ptf')         |  0.115 | 0.131 |  0.165 |
| ('qrf', 'structured vs ptf_marg')    |  0.071 | 0.083 |  0.123 |
| ('qrf', 'structured_ya vs chained')  |  0.122 | 0.072 | -0.01  |
| ('tabm', 'chained vs free')          | -0.004 | 0.02  | -0.016 |
| ('tabm', 'structured vs chained')    |  0.112 | 0.025 | -0.062 |
| ('tabm', 'structured vs clip')       |  0.091 | 0.031 | -0.06  |
| ('tabm', 'structured vs free')       |  0.109 | 0.046 | -0.078 |
| ('tabm', 'structured vs ptf')        |  0.078 | 0.098 |  0.173 |
| ('tabm', 'structured vs ptf_marg')   |  0.034 | 0.05  |  0.132 |
| ('tabm', 'structured_ya vs chained') |  0.127 | 0.019 | -0.023 |

## independent: skill differences (mean of N and CEC)

|                                      |   0.03 |    0.1 |    1.0 |
|:-------------------------------------|-------:|-------:|-------:|
| ('qrf', 'chained vs free')           | -0.006 | -0.006 | -0.007 |
| ('qrf', 'structured vs chained')     | -0.108 | -0.069 | -0.078 |
| ('qrf', 'structured vs clip')        | -0.118 | -0.034 |  0.003 |
| ('qrf', 'structured vs free')        | -0.114 | -0.074 | -0.085 |
| ('qrf', 'structured vs ptf')         |  0.216 |  0.312 |  0.424 |
| ('qrf', 'structured vs ptf_marg')    |  0.092 |  0.189 |  0.312 |
| ('qrf', 'structured_ya vs chained')  | -0.095 | -0.059 | -0.069 |
| ('tabm', 'chained vs free')          |  0.027 |  0.002 | -0.009 |
| ('tabm', 'structured vs chained')    | -0.013 | -0.108 | -0.109 |
| ('tabm', 'structured vs clip')       |  0.01  | -0.09  | -0.011 |
| ('tabm', 'structured vs free')       |  0.015 | -0.106 | -0.119 |
| ('tabm', 'structured vs ptf')        |  0.286 |  0.289 |  0.521 |
| ('tabm', 'structured vs ptf_marg')   |  0.161 |  0.167 |  0.408 |
| ('tabm', 'structured_ya vs chained') |  0.02  | -0.073 | -0.103 |
