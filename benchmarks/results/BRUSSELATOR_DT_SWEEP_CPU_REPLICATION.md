# Brusselator `dt` sweep: CPU replication on regenerated states

CPU wall times below are diagnostic only and are not comparable with the original GPU run.

## Similarity classification

Classification: **reproduction**; maximum relative difference 3.61828e-12 against threshold 1e-06. Source: `brusselator_dt_sweep_cpu_replication.json#/similarity_classification`.

| State | GPU max reaction norm | CPU max reaction norm | Relative difference | Within threshold | Source |
|---|---:|---:|---:|:---:|---|
| developed64:hopf:1 | 4.41651239094 | 4.41651239094 | 6.03312e-16 | True | `brusselator_dt_sweep_cpu_replication.json#/similarity/developed64:hopf:1` |
| developed64:hopf:10 | 8.80253721495 | 8.80253721495 | 1.5014e-13 | True | `brusselator_dt_sweep_cpu_replication.json#/similarity/developed64:hopf:10` |
| developed64:hopf:20 | 9.87983848681 | 9.87983848681 | 1.43837e-15 | True | `brusselator_dt_sweep_cpu_replication.json#/similarity/developed64:hopf:20` |
| developed64:turing:120 | 4.93148422378 | 4.93148422377 | 3.61828e-12 | True | `brusselator_dt_sweep_cpu_replication.json#/similarity/developed64:turing:120` |
| developed64:turing:200 | 16.5380643068 | 16.5380643068 | 2.53488e-13 | True | `brusselator_dt_sweep_cpu_replication.json#/similarity/developed64:turing:200` |
| developed64:turing:80 | 2.46834566308 | 2.46834566308 | 5.10955e-14 | True | `brusselator_dt_sweep_cpu_replication.json#/similarity/developed64:turing:80` |
| screen64:hopf:0 | 4.41720944115 | 4.41720944115 | 2.01072e-16 | True | `brusselator_dt_sweep_cpu_replication.json#/similarity/screen64:hopf:0` |
| screen64:hopf:1 | 4.41094350618 | 4.41094350618 | 4.02716e-16 | True | `brusselator_dt_sweep_cpu_replication.json#/similarity/screen64:hopf:1` |
| screen64:turing:0 | 2.40639991033 | 2.40639991033 | 3.6909e-16 | True | `brusselator_dt_sweep_cpu_replication.json#/similarity/screen64:turing:0` |
| screen64:turing:1 | 2.40128313209 | 2.40128313209 | 0 | True | `brusselator_dt_sweep_cpu_replication.json#/similarity/screen64:turing:1` |

## Pre-registered decision

| State | CPU FFT counts at dt <= 0.2 | CPU FFT count at dt=1 | Ratio | Pass | Source |
|---|---|---:|---:|:---:|---|
| developed64:hopf:1 | [5, 7, 8] | 37 | 5.28571 | True | `brusselator_dt_sweep_cpu_replication.json#/decision/d1_material_degradation/states/0` |
| developed64:hopf:10 | [6, 8, 11] | 50 | 6.25 | True | `brusselator_dt_sweep_cpu_replication.json#/decision/d1_material_degradation/states/1` |
| developed64:hopf:20 | [6, 8, 11] | 39 | 4.875 | True | `brusselator_dt_sweep_cpu_replication.json#/decision/d1_material_degradation/states/2` |
| developed64:turing:120 | [6, 7, 9] | 23 | 3.28571 | True | `brusselator_dt_sweep_cpu_replication.json#/decision/d1_material_degradation/states/3` |
| developed64:turing:200 | [5, 7, 8] | 20 | 2.85714 | True | `brusselator_dt_sweep_cpu_replication.json#/decision/d1_material_degradation/states/4` |
| developed64:turing:80 | [5, 5, 7] | 18 | 3.6 | True | `brusselator_dt_sweep_cpu_replication.json#/decision/d1_material_degradation/states/5` |

D1: **PASS**; median ratio 4.2375. Source: `brusselator_dt_sweep_cpu_replication.json#/decision/d1_material_degradation`.

D2: **PASS**; median count 30. Source: `brusselator_dt_sweep_cpu_replication.json#/decision/d2_absolute_cost`.

| Probe | FFT median | Probe median | Reduction | Lower disk rates | All runs converged | Pass | Source |
|---|---:|---:|---:|---:|:---:|:---:|---|
| mean_field_coupled_fft | 30 | 10 | 0.666667 | 6 | True | True | `brusselator_dt_sweep_cpu_replication.json#/decision/d3_recoverability/probes/0` |
| split_reaction_after_diffusion | 30 | 36.5 | -0.216667 | 0 | False | False | `brusselator_dt_sweep_cpu_replication.json#/decision/d3_recoverability/probes/1` |

D3: **PASS**. Source: `brusselator_dt_sweep_cpu_replication.json#/decision/d3_recoverability`.

CPU decision: **BUILD**. D1/D2/D3 match GPU: True/True/True. Success criterion met: True. Source: `brusselator_dt_sweep_cpu_replication.json#/decision_comparison`.

## Cell-by-cell comparison

| State | dt | Preconditioner | GPU iterations | CPU iterations | GPU converged | CPU converged | CPU seconds | Source |
|---|---:|---|---:|---:|:---:|:---:|---:|---|
| developed64:hopf:10 | 0.05 | fft_diffusion | 6 | 6 | True | True | 0.747021 | `brusselator_dt_sweep_cpu_replication.json#/cells/main|developed64:hopf:10|dt=0.05|fft_diffusion` |
| developed64:hopf:10 | 0.05 | identity | 10 | 10 | True | True | 0.744193 | `brusselator_dt_sweep_cpu_replication.json#/cells/main|developed64:hopf:10|dt=0.05|identity` |
| developed64:hopf:10 | 0.1 | fft_diffusion | 8 | 8 | True | True | 0.816385 | `brusselator_dt_sweep_cpu_replication.json#/cells/main|developed64:hopf:10|dt=0.1|fft_diffusion` |
| developed64:hopf:10 | 0.1 | identity | 15 | 15 | True | True | 0.731004 | `brusselator_dt_sweep_cpu_replication.json#/cells/main|developed64:hopf:10|dt=0.1|identity` |
| developed64:hopf:10 | 0.2 | fft_diffusion | 11 | 11 | True | True | 0.83235 | `brusselator_dt_sweep_cpu_replication.json#/cells/main|developed64:hopf:10|dt=0.2|fft_diffusion` |
| developed64:hopf:10 | 0.2 | identity | 23 | 23 | True | True | 0.722531 | `brusselator_dt_sweep_cpu_replication.json#/cells/main|developed64:hopf:10|dt=0.2|identity` |
| developed64:hopf:10 | 0.5 | fft_diffusion | 20 | 20 | True | True | 0.761983 | `brusselator_dt_sweep_cpu_replication.json#/cells/main|developed64:hopf:10|dt=0.5|fft_diffusion` |
| developed64:hopf:10 | 0.5 | identity | 51 | 51 | True | True | 0.867549 | `brusselator_dt_sweep_cpu_replication.json#/cells/main|developed64:hopf:10|dt=0.5|identity` |
| developed64:hopf:10 | 1.5 | fft_diffusion | 101 | 101 | False | False | 1.32733 | `brusselator_dt_sweep_cpu_replication.json#/cells/main|developed64:hopf:10|dt=1.5|fft_diffusion` |
| developed64:hopf:10 | 1.5 | identity | 101 | 101 | False | False | 1.31733 | `brusselator_dt_sweep_cpu_replication.json#/cells/main|developed64:hopf:10|dt=1.5|identity` |
| developed64:hopf:10 | 1 | fft_diffusion | 50 | 50 | True | True | 106.853 | `brusselator_dt_sweep_cpu_replication.json#/cells/main|developed64:hopf:10|dt=1|fft_diffusion` |
| developed64:hopf:10 | 1 | identity | 101 | 101 | False | False | 1.29429 | `brusselator_dt_sweep_cpu_replication.json#/cells/main|developed64:hopf:10|dt=1|identity` |
| developed64:hopf:10 | 2 | fft_diffusion | 101 | 101 | False | False | 1.38747 | `brusselator_dt_sweep_cpu_replication.json#/cells/main|developed64:hopf:10|dt=2|fft_diffusion` |
| developed64:hopf:10 | 2 | identity | 101 | 101 | False | False | 1.27128 | `brusselator_dt_sweep_cpu_replication.json#/cells/main|developed64:hopf:10|dt=2|identity` |
| developed64:hopf:1 | 0.05 | fft_diffusion | 5 | 5 | True | True | 1.71736 | `brusselator_dt_sweep_cpu_replication.json#/cells/main|developed64:hopf:1|dt=0.05|fft_diffusion` |
| developed64:hopf:1 | 0.05 | identity | 13 | 13 | True | True | 0.685547 | `brusselator_dt_sweep_cpu_replication.json#/cells/main|developed64:hopf:1|dt=0.05|identity` |
| developed64:hopf:1 | 0.1 | fft_diffusion | 7 | 7 | True | True | 0.697985 | `brusselator_dt_sweep_cpu_replication.json#/cells/main|developed64:hopf:1|dt=0.1|fft_diffusion` |
| developed64:hopf:1 | 0.1 | identity | 18 | 18 | True | True | 0.711149 | `brusselator_dt_sweep_cpu_replication.json#/cells/main|developed64:hopf:1|dt=0.1|identity` |
| developed64:hopf:1 | 0.2 | fft_diffusion | 8 | 8 | True | True | 0.781424 | `brusselator_dt_sweep_cpu_replication.json#/cells/main|developed64:hopf:1|dt=0.2|fft_diffusion` |
| developed64:hopf:1 | 0.2 | identity | 26 | 26 | True | True | 0.722037 | `brusselator_dt_sweep_cpu_replication.json#/cells/main|developed64:hopf:1|dt=0.2|identity` |
| developed64:hopf:1 | 0.5 | fft_diffusion | 15 | 15 | True | True | 0.725998 | `brusselator_dt_sweep_cpu_replication.json#/cells/main|developed64:hopf:1|dt=0.5|fft_diffusion` |
| developed64:hopf:1 | 0.5 | identity | 53 | 53 | True | True | 0.849524 | `brusselator_dt_sweep_cpu_replication.json#/cells/main|developed64:hopf:1|dt=0.5|identity` |
| developed64:hopf:1 | 1.5 | fft_diffusion | 101 | 101 | False | False | 1.40074 | `brusselator_dt_sweep_cpu_replication.json#/cells/main|developed64:hopf:1|dt=1.5|fft_diffusion` |
| developed64:hopf:1 | 1.5 | identity | 101 | 101 | False | False | 1.25034 | `brusselator_dt_sweep_cpu_replication.json#/cells/main|developed64:hopf:1|dt=1.5|identity` |
| developed64:hopf:1 | 1 | fft_diffusion | 37 | 37 | True | True | 114.137 | `brusselator_dt_sweep_cpu_replication.json#/cells/main|developed64:hopf:1|dt=1|fft_diffusion` |
| developed64:hopf:1 | 1 | identity | 101 | 101 | False | False | 1.31853 | `brusselator_dt_sweep_cpu_replication.json#/cells/main|developed64:hopf:1|dt=1|identity` |
| developed64:hopf:1 | 2 | fft_diffusion | 101 | 101 | False | False | 1.4398 | `brusselator_dt_sweep_cpu_replication.json#/cells/main|developed64:hopf:1|dt=2|fft_diffusion` |
| developed64:hopf:1 | 2 | identity | 101 | 101 | False | False | 1.30613 | `brusselator_dt_sweep_cpu_replication.json#/cells/main|developed64:hopf:1|dt=2|identity` |
| developed64:hopf:20 | 0.05 | fft_diffusion | 6 | 6 | True | True | 0.75809 | `brusselator_dt_sweep_cpu_replication.json#/cells/main|developed64:hopf:20|dt=0.05|fft_diffusion` |
| developed64:hopf:20 | 0.05 | identity | 10 | 10 | True | True | 0.712203 | `brusselator_dt_sweep_cpu_replication.json#/cells/main|developed64:hopf:20|dt=0.05|identity` |
| developed64:hopf:20 | 0.1 | fft_diffusion | 8 | 8 | True | True | 0.745653 | `brusselator_dt_sweep_cpu_replication.json#/cells/main|developed64:hopf:20|dt=0.1|fft_diffusion` |
| developed64:hopf:20 | 0.1 | identity | 14 | 14 | True | True | 0.765619 | `brusselator_dt_sweep_cpu_replication.json#/cells/main|developed64:hopf:20|dt=0.1|identity` |
| developed64:hopf:20 | 0.2 | fft_diffusion | 11 | 11 | True | True | 0.917907 | `brusselator_dt_sweep_cpu_replication.json#/cells/main|developed64:hopf:20|dt=0.2|fft_diffusion` |
| developed64:hopf:20 | 0.2 | identity | 22 | 22 | True | True | 0.739754 | `brusselator_dt_sweep_cpu_replication.json#/cells/main|developed64:hopf:20|dt=0.2|identity` |
| developed64:hopf:20 | 0.5 | fft_diffusion | 19 | 19 | True | True | 0.768659 | `brusselator_dt_sweep_cpu_replication.json#/cells/main|developed64:hopf:20|dt=0.5|fft_diffusion` |
| developed64:hopf:20 | 0.5 | identity | 45 | 45 | True | True | 0.828435 | `brusselator_dt_sweep_cpu_replication.json#/cells/main|developed64:hopf:20|dt=0.5|identity` |
| developed64:hopf:20 | 1.5 | fft_diffusion | 72 | 72 | True | True | 1.06389 | `brusselator_dt_sweep_cpu_replication.json#/cells/main|developed64:hopf:20|dt=1.5|fft_diffusion` |
| developed64:hopf:20 | 1.5 | identity | 101 | 101 | False | False | 1.47285 | `brusselator_dt_sweep_cpu_replication.json#/cells/main|developed64:hopf:20|dt=1.5|identity` |
| developed64:hopf:20 | 1 | fft_diffusion | 39 | 39 | True | True | 93.0149 | `brusselator_dt_sweep_cpu_replication.json#/cells/main|developed64:hopf:20|dt=1|fft_diffusion` |
| developed64:hopf:20 | 1 | identity | 91 | 91 | True | True | 1.29848 | `brusselator_dt_sweep_cpu_replication.json#/cells/main|developed64:hopf:20|dt=1|identity` |
| developed64:hopf:20 | 2 | fft_diffusion | 101 | 101 | False | False | 1.4026 | `brusselator_dt_sweep_cpu_replication.json#/cells/main|developed64:hopf:20|dt=2|fft_diffusion` |
| developed64:hopf:20 | 2 | identity | 101 | 101 | False | False | 1.28016 | `brusselator_dt_sweep_cpu_replication.json#/cells/main|developed64:hopf:20|dt=2|identity` |
| developed64:turing:120 | 0.05 | fft_diffusion | 6 | 6 | True | True | 0.803922 | `brusselator_dt_sweep_cpu_replication.json#/cells/main|developed64:turing:120|dt=0.05|fft_diffusion` |
| developed64:turing:120 | 0.05 | identity | 14 | 14 | True | True | 0.720443 | `brusselator_dt_sweep_cpu_replication.json#/cells/main|developed64:turing:120|dt=0.05|identity` |
| developed64:turing:120 | 0.1 | fft_diffusion | 7 | 7 | True | True | 0.747479 | `brusselator_dt_sweep_cpu_replication.json#/cells/main|developed64:turing:120|dt=0.1|fft_diffusion` |
| developed64:turing:120 | 0.1 | identity | 22 | 22 | True | True | 0.720347 | `brusselator_dt_sweep_cpu_replication.json#/cells/main|developed64:turing:120|dt=0.1|identity` |
| developed64:turing:120 | 0.2 | fft_diffusion | 9 | 9 | True | True | 0.745877 | `brusselator_dt_sweep_cpu_replication.json#/cells/main|developed64:turing:120|dt=0.2|fft_diffusion` |
| developed64:turing:120 | 0.2 | identity | 36 | 36 | True | True | 0.860443 | `brusselator_dt_sweep_cpu_replication.json#/cells/main|developed64:turing:120|dt=0.2|identity` |
| developed64:turing:120 | 0.5 | fft_diffusion | 14 | 14 | True | True | 0.751852 | `brusselator_dt_sweep_cpu_replication.json#/cells/main|developed64:turing:120|dt=0.5|fft_diffusion` |
| developed64:turing:120 | 0.5 | identity | 70 | 70 | True | True | 0.994076 | `brusselator_dt_sweep_cpu_replication.json#/cells/main|developed64:turing:120|dt=0.5|identity` |
| developed64:turing:120 | 1.5 | fft_diffusion | 34 | 34 | True | True | 0.843664 | `brusselator_dt_sweep_cpu_replication.json#/cells/main|developed64:turing:120|dt=1.5|fft_diffusion` |
| developed64:turing:120 | 1.5 | identity | 101 | 101 | False | False | 1.3128 | `brusselator_dt_sweep_cpu_replication.json#/cells/main|developed64:turing:120|dt=1.5|identity` |
| developed64:turing:120 | 1 | fft_diffusion | 23 | 23 | True | True | 79.7951 | `brusselator_dt_sweep_cpu_replication.json#/cells/main|developed64:turing:120|dt=1|fft_diffusion` |
| developed64:turing:120 | 1 | identity | 101 | 101 | False | False | 1.37177 | `brusselator_dt_sweep_cpu_replication.json#/cells/main|developed64:turing:120|dt=1|identity` |
| developed64:turing:120 | 2 | fft_diffusion | 48 | 48 | True | True | 0.904684 | `brusselator_dt_sweep_cpu_replication.json#/cells/main|developed64:turing:120|dt=2|fft_diffusion` |
| developed64:turing:120 | 2 | identity | 101 | 101 | False | False | 1.34921 | `brusselator_dt_sweep_cpu_replication.json#/cells/main|developed64:turing:120|dt=2|identity` |
| developed64:turing:200 | 0.05 | fft_diffusion | 5 | 5 | True | True | 0.760776 | `brusselator_dt_sweep_cpu_replication.json#/cells/main|developed64:turing:200|dt=0.05|fft_diffusion` |
| developed64:turing:200 | 0.05 | identity | 9 | 9 | True | True | 0.718441 | `brusselator_dt_sweep_cpu_replication.json#/cells/main|developed64:turing:200|dt=0.05|identity` |
| developed64:turing:200 | 0.1 | fft_diffusion | 7 | 7 | True | True | 0.748633 | `brusselator_dt_sweep_cpu_replication.json#/cells/main|developed64:turing:200|dt=0.1|fft_diffusion` |
| developed64:turing:200 | 0.1 | identity | 14 | 14 | True | True | 0.7155 | `brusselator_dt_sweep_cpu_replication.json#/cells/main|developed64:turing:200|dt=0.1|identity` |
| developed64:turing:200 | 0.2 | fft_diffusion | 8 | 8 | True | True | 0.747957 | `brusselator_dt_sweep_cpu_replication.json#/cells/main|developed64:turing:200|dt=0.2|fft_diffusion` |
| developed64:turing:200 | 0.2 | identity | 22 | 22 | True | True | 0.882004 | `brusselator_dt_sweep_cpu_replication.json#/cells/main|developed64:turing:200|dt=0.2|identity` |
| developed64:turing:200 | 0.5 | fft_diffusion | 12 | 12 | True | True | 0.780539 | `brusselator_dt_sweep_cpu_replication.json#/cells/main|developed64:turing:200|dt=0.5|fft_diffusion` |
| developed64:turing:200 | 0.5 | identity | 44 | 44 | True | True | 0.844732 | `brusselator_dt_sweep_cpu_replication.json#/cells/main|developed64:turing:200|dt=0.5|identity` |
| developed64:turing:200 | 1.5 | fft_diffusion | 27 | 27 | True | True | 0.814603 | `brusselator_dt_sweep_cpu_replication.json#/cells/main|developed64:turing:200|dt=1.5|fft_diffusion` |
| developed64:turing:200 | 1.5 | identity | 96 | 96 | True | True | 1.42449 | `brusselator_dt_sweep_cpu_replication.json#/cells/main|developed64:turing:200|dt=1.5|identity` |
| developed64:turing:200 | 1 | fft_diffusion | 20 | 20 | True | True | 45.972 | `brusselator_dt_sweep_cpu_replication.json#/cells/main|developed64:turing:200|dt=1|fft_diffusion` |
| developed64:turing:200 | 1 | identity | 73 | 73 | True | True | 1.09379 | `brusselator_dt_sweep_cpu_replication.json#/cells/main|developed64:turing:200|dt=1|identity` |
| developed64:turing:200 | 2 | fft_diffusion | 36 | 36 | True | True | 0.843587 | `brusselator_dt_sweep_cpu_replication.json#/cells/main|developed64:turing:200|dt=2|fft_diffusion` |
| developed64:turing:200 | 2 | identity | 101 | 101 | False | False | 1.33469 | `brusselator_dt_sweep_cpu_replication.json#/cells/main|developed64:turing:200|dt=2|identity` |
| developed64:turing:80 | 0.05 | fft_diffusion | 5 | 5 | True | True | 0.757742 | `brusselator_dt_sweep_cpu_replication.json#/cells/main|developed64:turing:80|dt=0.05|fft_diffusion` |
| developed64:turing:80 | 0.05 | identity | 6 | 6 | True | True | 0.9233 | `brusselator_dt_sweep_cpu_replication.json#/cells/main|developed64:turing:80|dt=0.05|identity` |
| developed64:turing:80 | 0.1 | fft_diffusion | 5 | 5 | True | True | 0.724796 | `brusselator_dt_sweep_cpu_replication.json#/cells/main|developed64:turing:80|dt=0.1|fft_diffusion` |
| developed64:turing:80 | 0.1 | identity | 11 | 11 | True | True | 0.737806 | `brusselator_dt_sweep_cpu_replication.json#/cells/main|developed64:turing:80|dt=0.1|identity` |
| developed64:turing:80 | 0.2 | fft_diffusion | 7 | 7 | True | True | 0.742623 | `brusselator_dt_sweep_cpu_replication.json#/cells/main|developed64:turing:80|dt=0.2|fft_diffusion` |
| developed64:turing:80 | 0.2 | identity | 20 | 20 | True | True | 0.759244 | `brusselator_dt_sweep_cpu_replication.json#/cells/main|developed64:turing:80|dt=0.2|identity` |
| developed64:turing:80 | 0.5 | fft_diffusion | 11 | 11 | True | True | 0.773144 | `brusselator_dt_sweep_cpu_replication.json#/cells/main|developed64:turing:80|dt=0.5|fft_diffusion` |
| developed64:turing:80 | 0.5 | identity | 43 | 43 | True | True | 0.822545 | `brusselator_dt_sweep_cpu_replication.json#/cells/main|developed64:turing:80|dt=0.5|identity` |
| developed64:turing:80 | 1.5 | fft_diffusion | 24 | 24 | True | True | 0.805625 | `brusselator_dt_sweep_cpu_replication.json#/cells/main|developed64:turing:80|dt=1.5|fft_diffusion` |
| developed64:turing:80 | 1.5 | identity | 101 | 101 | False | False | 1.33044 | `brusselator_dt_sweep_cpu_replication.json#/cells/main|developed64:turing:80|dt=1.5|identity` |
| developed64:turing:80 | 1 | fft_diffusion | 18 | 18 | True | True | 113.747 | `brusselator_dt_sweep_cpu_replication.json#/cells/main|developed64:turing:80|dt=1|fft_diffusion` |
| developed64:turing:80 | 1 | identity | 80 | 80 | True | True | 1.13139 | `brusselator_dt_sweep_cpu_replication.json#/cells/main|developed64:turing:80|dt=1|identity` |
| developed64:turing:80 | 2 | fft_diffusion | 31 | 31 | True | True | 1.06996 | `brusselator_dt_sweep_cpu_replication.json#/cells/main|developed64:turing:80|dt=2|fft_diffusion` |
| developed64:turing:80 | 2 | identity | 101 | 101 | False | False | 1.32235 | `brusselator_dt_sweep_cpu_replication.json#/cells/main|developed64:turing:80|dt=2|identity` |
| screen64:hopf:0 | 0.05 | fft_diffusion | 5 | 5 | True | True | 0.767636 | `brusselator_dt_sweep_cpu_replication.json#/cells/main|screen64:hopf:0|dt=0.05|fft_diffusion` |
| screen64:hopf:0 | 0.05 | identity | 13 | 13 | True | True | 0.731069 | `brusselator_dt_sweep_cpu_replication.json#/cells/main|screen64:hopf:0|dt=0.05|identity` |
| screen64:hopf:0 | 0.1 | fft_diffusion | 6 | 6 | True | True | 0.758747 | `brusselator_dt_sweep_cpu_replication.json#/cells/main|screen64:hopf:0|dt=0.1|fft_diffusion` |
| screen64:hopf:0 | 0.1 | identity | 18 | 18 | True | True | 0.753026 | `brusselator_dt_sweep_cpu_replication.json#/cells/main|screen64:hopf:0|dt=0.1|identity` |
| screen64:hopf:0 | 0.2 | fft_diffusion | 8 | 8 | True | True | 0.739795 | `brusselator_dt_sweep_cpu_replication.json#/cells/main|screen64:hopf:0|dt=0.2|fft_diffusion` |
| screen64:hopf:0 | 0.2 | identity | 25 | 25 | True | True | 0.749608 | `brusselator_dt_sweep_cpu_replication.json#/cells/main|screen64:hopf:0|dt=0.2|identity` |
| screen64:hopf:0 | 0.5 | fft_diffusion | 14 | 14 | True | True | 0.744938 | `brusselator_dt_sweep_cpu_replication.json#/cells/main|screen64:hopf:0|dt=0.5|fft_diffusion` |
| screen64:hopf:0 | 0.5 | identity | 49 | 49 | True | True | 0.872457 | `brusselator_dt_sweep_cpu_replication.json#/cells/main|screen64:hopf:0|dt=0.5|identity` |
| screen64:hopf:0 | 1.5 | fft_diffusion | 101 | 101 | False | False | 1.36385 | `brusselator_dt_sweep_cpu_replication.json#/cells/main|screen64:hopf:0|dt=1.5|fft_diffusion` |
| screen64:hopf:0 | 1.5 | identity | 101 | 101 | False | False | 1.36757 | `brusselator_dt_sweep_cpu_replication.json#/cells/main|screen64:hopf:0|dt=1.5|identity` |
| screen64:hopf:0 | 1 | fft_diffusion | 35 | 35 | True | True | 0.846817 | `brusselator_dt_sweep_cpu_replication.json#/cells/main|screen64:hopf:0|dt=1|fft_diffusion` |
| screen64:hopf:0 | 1 | identity | 101 | 101 | False | False | 1.53642 | `brusselator_dt_sweep_cpu_replication.json#/cells/main|screen64:hopf:0|dt=1|identity` |
| screen64:hopf:0 | 2 | fft_diffusion | 101 | 101 | False | False | 1.44857 | `brusselator_dt_sweep_cpu_replication.json#/cells/main|screen64:hopf:0|dt=2|fft_diffusion` |
| screen64:hopf:0 | 2 | identity | 101 | 101 | False | False | 1.31055 | `brusselator_dt_sweep_cpu_replication.json#/cells/main|screen64:hopf:0|dt=2|identity` |
| screen64:hopf:1 | 0.05 | fft_diffusion | 5 | 5 | True | True | 0.761619 | `brusselator_dt_sweep_cpu_replication.json#/cells/main|screen64:hopf:1|dt=0.05|fft_diffusion` |
| screen64:hopf:1 | 0.05 | identity | 13 | 13 | True | True | 0.734621 | `brusselator_dt_sweep_cpu_replication.json#/cells/main|screen64:hopf:1|dt=0.05|identity` |
| screen64:hopf:1 | 0.1 | fft_diffusion | 7 | 7 | True | True | 0.959531 | `brusselator_dt_sweep_cpu_replication.json#/cells/main|screen64:hopf:1|dt=0.1|fft_diffusion` |
| screen64:hopf:1 | 0.1 | identity | 18 | 18 | True | True | 0.737327 | `brusselator_dt_sweep_cpu_replication.json#/cells/main|screen64:hopf:1|dt=0.1|identity` |
| screen64:hopf:1 | 0.2 | fft_diffusion | 8 | 8 | True | True | 0.755201 | `brusselator_dt_sweep_cpu_replication.json#/cells/main|screen64:hopf:1|dt=0.2|fft_diffusion` |
| screen64:hopf:1 | 0.2 | identity | 26 | 26 | True | True | 0.751656 | `brusselator_dt_sweep_cpu_replication.json#/cells/main|screen64:hopf:1|dt=0.2|identity` |
| screen64:hopf:1 | 0.5 | fft_diffusion | 14 | 14 | True | True | 0.770684 | `brusselator_dt_sweep_cpu_replication.json#/cells/main|screen64:hopf:1|dt=0.5|fft_diffusion` |
| screen64:hopf:1 | 0.5 | identity | 50 | 50 | True | True | 0.867056 | `brusselator_dt_sweep_cpu_replication.json#/cells/main|screen64:hopf:1|dt=0.5|identity` |
| screen64:hopf:1 | 1.5 | fft_diffusion | 101 | 101 | False | False | 1.3455 | `brusselator_dt_sweep_cpu_replication.json#/cells/main|screen64:hopf:1|dt=1.5|fft_diffusion` |
| screen64:hopf:1 | 1.5 | identity | 101 | 101 | False | False | 1.35022 | `brusselator_dt_sweep_cpu_replication.json#/cells/main|screen64:hopf:1|dt=1.5|identity` |
| screen64:hopf:1 | 1 | fft_diffusion | 37 | 37 | True | True | 0.874926 | `brusselator_dt_sweep_cpu_replication.json#/cells/main|screen64:hopf:1|dt=1|fft_diffusion` |
| screen64:hopf:1 | 1 | identity | 101 | 101 | False | False | 1.44851 | `brusselator_dt_sweep_cpu_replication.json#/cells/main|screen64:hopf:1|dt=1|identity` |
| screen64:hopf:1 | 2 | fft_diffusion | 101 | 101 | False | False | 1.37228 | `brusselator_dt_sweep_cpu_replication.json#/cells/main|screen64:hopf:1|dt=2|fft_diffusion` |
| screen64:hopf:1 | 2 | identity | 101 | 101 | False | False | 1.28307 | `brusselator_dt_sweep_cpu_replication.json#/cells/main|screen64:hopf:1|dt=2|identity` |
| screen64:turing:0 | 0.05 | fft_diffusion | 6 | 6 | True | True | 0.75537 | `brusselator_dt_sweep_cpu_replication.json#/cells/main|screen64:turing:0|dt=0.05|fft_diffusion` |
| screen64:turing:0 | 0.05 | identity | 25 | 25 | True | True | 0.749541 | `brusselator_dt_sweep_cpu_replication.json#/cells/main|screen64:turing:0|dt=0.05|identity` |
| screen64:turing:0 | 0.1 | fft_diffusion | 7 | 7 | True | True | 0.758163 | `brusselator_dt_sweep_cpu_replication.json#/cells/main|screen64:turing:0|dt=0.1|fft_diffusion` |
| screen64:turing:0 | 0.1 | identity | 34 | 34 | True | True | 0.781711 | `brusselator_dt_sweep_cpu_replication.json#/cells/main|screen64:turing:0|dt=0.1|identity` |
| screen64:turing:0 | 0.2 | fft_diffusion | 8 | 8 | True | True | 0.772367 | `brusselator_dt_sweep_cpu_replication.json#/cells/main|screen64:turing:0|dt=0.2|fft_diffusion` |
| screen64:turing:0 | 0.2 | identity | 49 | 49 | True | True | 0.871857 | `brusselator_dt_sweep_cpu_replication.json#/cells/main|screen64:turing:0|dt=0.2|identity` |
| screen64:turing:0 | 0.5 | fft_diffusion | 14 | 14 | True | True | 0.74719 | `brusselator_dt_sweep_cpu_replication.json#/cells/main|screen64:turing:0|dt=0.5|fft_diffusion` |
| screen64:turing:0 | 0.5 | identity | 80 | 80 | True | True | 1.11618 | `brusselator_dt_sweep_cpu_replication.json#/cells/main|screen64:turing:0|dt=0.5|identity` |
| screen64:turing:0 | 1.5 | fft_diffusion | 30 | 30 | True | True | 0.808852 | `brusselator_dt_sweep_cpu_replication.json#/cells/main|screen64:turing:0|dt=1.5|fft_diffusion` |
| screen64:turing:0 | 1.5 | identity | 101 | 101 | False | False | 1.30201 | `brusselator_dt_sweep_cpu_replication.json#/cells/main|screen64:turing:0|dt=1.5|identity` |
| screen64:turing:0 | 1 | fft_diffusion | 22 | 22 | True | True | 0.773403 | `brusselator_dt_sweep_cpu_replication.json#/cells/main|screen64:turing:0|dt=1|fft_diffusion` |
| screen64:turing:0 | 1 | identity | 101 | 101 | False | False | 1.29326 | `brusselator_dt_sweep_cpu_replication.json#/cells/main|screen64:turing:0|dt=1|identity` |
| screen64:turing:0 | 2 | fft_diffusion | 38 | 38 | True | True | 1.10077 | `brusselator_dt_sweep_cpu_replication.json#/cells/main|screen64:turing:0|dt=2|fft_diffusion` |
| screen64:turing:0 | 2 | identity | 101 | 101 | False | False | 1.32585 | `brusselator_dt_sweep_cpu_replication.json#/cells/main|screen64:turing:0|dt=2|identity` |
| screen64:turing:1 | 0.05 | fft_diffusion | 6 | 6 | True | True | 0.739894 | `brusselator_dt_sweep_cpu_replication.json#/cells/main|screen64:turing:1|dt=0.05|fft_diffusion` |
| screen64:turing:1 | 0.05 | identity | 25 | 25 | True | True | 0.752654 | `brusselator_dt_sweep_cpu_replication.json#/cells/main|screen64:turing:1|dt=0.05|identity` |
| screen64:turing:1 | 0.1 | fft_diffusion | 7 | 7 | True | True | 0.760711 | `brusselator_dt_sweep_cpu_replication.json#/cells/main|screen64:turing:1|dt=0.1|fft_diffusion` |
| screen64:turing:1 | 0.1 | identity | 35 | 35 | True | True | 0.815114 | `brusselator_dt_sweep_cpu_replication.json#/cells/main|screen64:turing:1|dt=0.1|identity` |
| screen64:turing:1 | 0.2 | fft_diffusion | 9 | 9 | True | True | 0.755519 | `brusselator_dt_sweep_cpu_replication.json#/cells/main|screen64:turing:1|dt=0.2|fft_diffusion` |
| screen64:turing:1 | 0.2 | identity | 50 | 50 | True | True | 0.870632 | `brusselator_dt_sweep_cpu_replication.json#/cells/main|screen64:turing:1|dt=0.2|identity` |
| screen64:turing:1 | 0.5 | fft_diffusion | 14 | 14 | True | True | 0.758354 | `brusselator_dt_sweep_cpu_replication.json#/cells/main|screen64:turing:1|dt=0.5|fft_diffusion` |
| screen64:turing:1 | 0.5 | identity | 86 | 86 | True | True | 1.16921 | `brusselator_dt_sweep_cpu_replication.json#/cells/main|screen64:turing:1|dt=0.5|identity` |
| screen64:turing:1 | 1.5 | fft_diffusion | 30 | 30 | True | True | 0.820556 | `brusselator_dt_sweep_cpu_replication.json#/cells/main|screen64:turing:1|dt=1.5|fft_diffusion` |
| screen64:turing:1 | 1.5 | identity | 101 | 101 | False | False | 1.31818 | `brusselator_dt_sweep_cpu_replication.json#/cells/main|screen64:turing:1|dt=1.5|identity` |
| screen64:turing:1 | 1 | fft_diffusion | 22 | 22 | True | True | 1.10227 | `brusselator_dt_sweep_cpu_replication.json#/cells/main|screen64:turing:1|dt=1|fft_diffusion` |
| screen64:turing:1 | 1 | identity | 101 | 101 | False | False | 1.36057 | `brusselator_dt_sweep_cpu_replication.json#/cells/main|screen64:turing:1|dt=1|identity` |
| screen64:turing:1 | 2 | fft_diffusion | 38 | 38 | True | True | 0.842661 | `brusselator_dt_sweep_cpu_replication.json#/cells/main|screen64:turing:1|dt=2|fft_diffusion` |
| screen64:turing:1 | 2 | identity | 101 | 101 | False | False | 1.36643 | `brusselator_dt_sweep_cpu_replication.json#/cells/main|screen64:turing:1|dt=2|identity` |
| developed64:hopf:10 | 0.2 | mean_field_coupled_fft | 9 | 9 | True | True | 0.752903 | `brusselator_dt_sweep_cpu_replication.json#/cells/probe|developed64:hopf:10|dt=0.2|mean_field_coupled_fft` |
| developed64:hopf:10 | 0.2 | split_reaction_after_diffusion | 10 | 10 | True | True | 0.802168 | `brusselator_dt_sweep_cpu_replication.json#/cells/probe|developed64:hopf:10|dt=0.2|split_reaction_after_diffusion` |
| developed64:hopf:10 | 1 | mean_field_coupled_fft | 23 | 23 | True | True | 104.588 | `brusselator_dt_sweep_cpu_replication.json#/cells/probe|developed64:hopf:10|dt=1|mean_field_coupled_fft` |
| developed64:hopf:10 | 1 | split_reaction_after_diffusion | 101 | 101 | False | False | 1.38171 | `brusselator_dt_sweep_cpu_replication.json#/cells/probe|developed64:hopf:10|dt=1|split_reaction_after_diffusion` |
| developed64:hopf:10 | 2 | mean_field_coupled_fft | 57 | 57 | True | True | 0.960175 | `brusselator_dt_sweep_cpu_replication.json#/cells/probe|developed64:hopf:10|dt=2|mean_field_coupled_fft` |
| developed64:hopf:10 | 2 | split_reaction_after_diffusion | 101 | 101 | False | False | 1.49928 | `brusselator_dt_sweep_cpu_replication.json#/cells/probe|developed64:hopf:10|dt=2|split_reaction_after_diffusion` |
| developed64:hopf:1 | 0.2 | mean_field_coupled_fft | 4 | 4 | True | True | 0.910297 | `brusselator_dt_sweep_cpu_replication.json#/cells/probe|developed64:hopf:1|dt=0.2|mean_field_coupled_fft` |
| developed64:hopf:1 | 0.2 | split_reaction_after_diffusion | 8 | 8 | True | True | 0.902087 | `brusselator_dt_sweep_cpu_replication.json#/cells/probe|developed64:hopf:1|dt=0.2|split_reaction_after_diffusion` |
| developed64:hopf:1 | 1 | mean_field_coupled_fft | 5 | 5 | True | True | 113.33 | `brusselator_dt_sweep_cpu_replication.json#/cells/probe|developed64:hopf:1|dt=1|mean_field_coupled_fft` |
| developed64:hopf:1 | 1 | split_reaction_after_diffusion | 42 | 42 | True | True | 0.969408 | `brusselator_dt_sweep_cpu_replication.json#/cells/probe|developed64:hopf:1|dt=1|split_reaction_after_diffusion` |
| developed64:hopf:1 | 2 | mean_field_coupled_fft | 11 | 11 | True | True | 0.745389 | `brusselator_dt_sweep_cpu_replication.json#/cells/probe|developed64:hopf:1|dt=2|mean_field_coupled_fft` |
| developed64:hopf:1 | 2 | split_reaction_after_diffusion | 101 | 101 | False | False | 1.4838 | `brusselator_dt_sweep_cpu_replication.json#/cells/probe|developed64:hopf:1|dt=2|split_reaction_after_diffusion` |
| developed64:hopf:20 | 0.2 | mean_field_coupled_fft | 9 | 9 | True | True | 0.758186 | `brusselator_dt_sweep_cpu_replication.json#/cells/probe|developed64:hopf:20|dt=0.2|mean_field_coupled_fft` |
| developed64:hopf:20 | 0.2 | split_reaction_after_diffusion | 10 | 10 | True | True | 0.78467 | `brusselator_dt_sweep_cpu_replication.json#/cells/probe|developed64:hopf:20|dt=0.2|split_reaction_after_diffusion` |
| developed64:hopf:20 | 1 | mean_field_coupled_fft | 18 | 18 | True | True | 102.916 | `brusselator_dt_sweep_cpu_replication.json#/cells/probe|developed64:hopf:20|dt=1|mean_field_coupled_fft` |
| developed64:hopf:20 | 1 | split_reaction_after_diffusion | 88 | 88 | True | True | 1.28762 | `brusselator_dt_sweep_cpu_replication.json#/cells/probe|developed64:hopf:20|dt=1|split_reaction_after_diffusion` |
| developed64:hopf:20 | 2 | mean_field_coupled_fft | 29 | 29 | True | True | 0.842946 | `brusselator_dt_sweep_cpu_replication.json#/cells/probe|developed64:hopf:20|dt=2|mean_field_coupled_fft` |
| developed64:hopf:20 | 2 | split_reaction_after_diffusion | 101 | 101 | False | False | 1.4063 | `brusselator_dt_sweep_cpu_replication.json#/cells/probe|developed64:hopf:20|dt=2|split_reaction_after_diffusion` |
| developed64:turing:120 | 0.2 | mean_field_coupled_fft | 6 | 6 | True | True | 0.873504 | `brusselator_dt_sweep_cpu_replication.json#/cells/probe|developed64:turing:120|dt=0.2|mean_field_coupled_fft` |
| developed64:turing:120 | 0.2 | split_reaction_after_diffusion | 8 | 8 | True | True | 0.799643 | `brusselator_dt_sweep_cpu_replication.json#/cells/probe|developed64:turing:120|dt=0.2|split_reaction_after_diffusion` |
| developed64:turing:120 | 1 | mean_field_coupled_fft | 10 | 10 | True | True | 74.3338 | `brusselator_dt_sweep_cpu_replication.json#/cells/probe|developed64:turing:120|dt=1|mean_field_coupled_fft` |
| developed64:turing:120 | 1 | split_reaction_after_diffusion | 28 | 28 | True | True | 0.955893 | `brusselator_dt_sweep_cpu_replication.json#/cells/probe|developed64:turing:120|dt=1|split_reaction_after_diffusion` |
| developed64:turing:120 | 2 | mean_field_coupled_fft | 15 | 15 | True | True | 0.765463 | `brusselator_dt_sweep_cpu_replication.json#/cells/probe|developed64:turing:120|dt=2|mean_field_coupled_fft` |
| developed64:turing:120 | 2 | split_reaction_after_diffusion | 60 | 60 | True | True | 1.0071 | `brusselator_dt_sweep_cpu_replication.json#/cells/probe|developed64:turing:120|dt=2|split_reaction_after_diffusion` |
| developed64:turing:200 | 0.2 | mean_field_coupled_fft | 6 | 6 | True | True | 0.776026 | `brusselator_dt_sweep_cpu_replication.json#/cells/probe|developed64:turing:200|dt=0.2|mean_field_coupled_fft` |
| developed64:turing:200 | 0.2 | split_reaction_after_diffusion | 10 | 10 | True | True | 0.821493 | `brusselator_dt_sweep_cpu_replication.json#/cells/probe|developed64:turing:200|dt=0.2|split_reaction_after_diffusion` |
| developed64:turing:200 | 1 | mean_field_coupled_fft | 10 | 10 | True | True | 44.3369 | `brusselator_dt_sweep_cpu_replication.json#/cells/probe|developed64:turing:200|dt=1|mean_field_coupled_fft` |
| developed64:turing:200 | 1 | split_reaction_after_diffusion | 31 | 31 | True | True | 1.05864 | `brusselator_dt_sweep_cpu_replication.json#/cells/probe|developed64:turing:200|dt=1|split_reaction_after_diffusion` |
| developed64:turing:200 | 2 | mean_field_coupled_fft | 13 | 13 | True | True | 0.843237 | `brusselator_dt_sweep_cpu_replication.json#/cells/probe|developed64:turing:200|dt=2|mean_field_coupled_fft` |
| developed64:turing:200 | 2 | split_reaction_after_diffusion | 52 | 52 | True | True | 0.956491 | `brusselator_dt_sweep_cpu_replication.json#/cells/probe|developed64:turing:200|dt=2|split_reaction_after_diffusion` |
| developed64:turing:80 | 0.2 | mean_field_coupled_fft | 4 | 4 | True | True | 0.749944 | `brusselator_dt_sweep_cpu_replication.json#/cells/probe|developed64:turing:80|dt=0.2|mean_field_coupled_fft` |
| developed64:turing:80 | 0.2 | split_reaction_after_diffusion | 5 | 5 | True | True | 0.791337 | `brusselator_dt_sweep_cpu_replication.json#/cells/probe|developed64:turing:80|dt=0.2|split_reaction_after_diffusion` |
| developed64:turing:80 | 1 | mean_field_coupled_fft | 5 | 5 | True | True | 117.769 | `brusselator_dt_sweep_cpu_replication.json#/cells/probe|developed64:turing:80|dt=1|mean_field_coupled_fft` |
| developed64:turing:80 | 1 | split_reaction_after_diffusion | 13 | 13 | True | True | 0.919777 | `brusselator_dt_sweep_cpu_replication.json#/cells/probe|developed64:turing:80|dt=1|split_reaction_after_diffusion` |
| developed64:turing:80 | 2 | mean_field_coupled_fft | 5 | 5 | True | True | 0.735838 | `brusselator_dt_sweep_cpu_replication.json#/cells/probe|developed64:turing:80|dt=2|mean_field_coupled_fft` |
| developed64:turing:80 | 2 | split_reaction_after_diffusion | 23 | 23 | True | True | 0.796953 | `brusselator_dt_sweep_cpu_replication.json#/cells/probe|developed64:turing:80|dt=2|split_reaction_after_diffusion` |
| screen64:hopf:0 | 0.2 | mean_field_coupled_fft | 4 | 4 | True | True | 0.767525 | `brusselator_dt_sweep_cpu_replication.json#/cells/probe|screen64:hopf:0|dt=0.2|mean_field_coupled_fft` |
| screen64:hopf:0 | 0.2 | split_reaction_after_diffusion | 8 | 8 | True | True | 0.789346 | `brusselator_dt_sweep_cpu_replication.json#/cells/probe|screen64:hopf:0|dt=0.2|split_reaction_after_diffusion` |
| screen64:hopf:0 | 1 | mean_field_coupled_fft | 4 | 4 | True | True | 0.751184 | `brusselator_dt_sweep_cpu_replication.json#/cells/probe|screen64:hopf:0|dt=1|mean_field_coupled_fft` |
| screen64:hopf:0 | 1 | split_reaction_after_diffusion | 41 | 41 | True | True | 0.888369 | `brusselator_dt_sweep_cpu_replication.json#/cells/probe|screen64:hopf:0|dt=1|split_reaction_after_diffusion` |
| screen64:hopf:0 | 2 | mean_field_coupled_fft | 9 | 9 | True | True | 0.748504 | `brusselator_dt_sweep_cpu_replication.json#/cells/probe|screen64:hopf:0|dt=2|mean_field_coupled_fft` |
| screen64:hopf:0 | 2 | split_reaction_after_diffusion | 101 | 101 | False | False | 1.49547 | `brusselator_dt_sweep_cpu_replication.json#/cells/probe|screen64:hopf:0|dt=2|split_reaction_after_diffusion` |
| screen64:hopf:1 | 0.2 | mean_field_coupled_fft | 4 | 4 | True | True | 0.753883 | `brusselator_dt_sweep_cpu_replication.json#/cells/probe|screen64:hopf:1|dt=0.2|mean_field_coupled_fft` |
| screen64:hopf:1 | 0.2 | split_reaction_after_diffusion | 8 | 8 | True | True | 0.770621 | `brusselator_dt_sweep_cpu_replication.json#/cells/probe|screen64:hopf:1|dt=0.2|split_reaction_after_diffusion` |
| screen64:hopf:1 | 1 | mean_field_coupled_fft | 4 | 4 | True | True | 0.989877 | `brusselator_dt_sweep_cpu_replication.json#/cells/probe|screen64:hopf:1|dt=1|mean_field_coupled_fft` |
| screen64:hopf:1 | 1 | split_reaction_after_diffusion | 42 | 42 | True | True | 0.884428 | `brusselator_dt_sweep_cpu_replication.json#/cells/probe|screen64:hopf:1|dt=1|split_reaction_after_diffusion` |
| screen64:hopf:1 | 2 | mean_field_coupled_fft | 9 | 9 | True | True | 0.741908 | `brusselator_dt_sweep_cpu_replication.json#/cells/probe|screen64:hopf:1|dt=2|mean_field_coupled_fft` |
| screen64:hopf:1 | 2 | split_reaction_after_diffusion | 101 | 101 | False | False | 1.38705 | `brusselator_dt_sweep_cpu_replication.json#/cells/probe|screen64:hopf:1|dt=2|split_reaction_after_diffusion` |
| screen64:turing:0 | 0.2 | mean_field_coupled_fft | 3 | 3 | True | True | 0.739322 | `brusselator_dt_sweep_cpu_replication.json#/cells/probe|screen64:turing:0|dt=0.2|mean_field_coupled_fft` |
| screen64:turing:0 | 0.2 | split_reaction_after_diffusion | 8 | 8 | True | True | 0.773318 | `brusselator_dt_sweep_cpu_replication.json#/cells/probe|screen64:turing:0|dt=0.2|split_reaction_after_diffusion` |
| screen64:turing:0 | 1 | mean_field_coupled_fft | 4 | 4 | True | True | 0.740541 | `brusselator_dt_sweep_cpu_replication.json#/cells/probe|screen64:turing:0|dt=1|mean_field_coupled_fft` |
| screen64:turing:0 | 1 | split_reaction_after_diffusion | 20 | 20 | True | True | 0.81798 | `brusselator_dt_sweep_cpu_replication.json#/cells/probe|screen64:turing:0|dt=1|split_reaction_after_diffusion` |
| screen64:turing:0 | 2 | mean_field_coupled_fft | 4 | 4 | True | True | 0.742847 | `brusselator_dt_sweep_cpu_replication.json#/cells/probe|screen64:turing:0|dt=2|mean_field_coupled_fft` |
| screen64:turing:0 | 2 | split_reaction_after_diffusion | 34 | 34 | True | True | 0.846684 | `brusselator_dt_sweep_cpu_replication.json#/cells/probe|screen64:turing:0|dt=2|split_reaction_after_diffusion` |
| screen64:turing:1 | 0.2 | mean_field_coupled_fft | 3 | 3 | True | True | 0.751605 | `brusselator_dt_sweep_cpu_replication.json#/cells/probe|screen64:turing:1|dt=0.2|mean_field_coupled_fft` |
| screen64:turing:1 | 0.2 | split_reaction_after_diffusion | 8 | 8 | True | True | 0.772989 | `brusselator_dt_sweep_cpu_replication.json#/cells/probe|screen64:turing:1|dt=0.2|split_reaction_after_diffusion` |
| screen64:turing:1 | 1 | mean_field_coupled_fft | 4 | 4 | True | True | 0.742658 | `brusselator_dt_sweep_cpu_replication.json#/cells/probe|screen64:turing:1|dt=1|mean_field_coupled_fft` |
| screen64:turing:1 | 1 | split_reaction_after_diffusion | 20 | 20 | True | True | 0.809564 | `brusselator_dt_sweep_cpu_replication.json#/cells/probe|screen64:turing:1|dt=1|split_reaction_after_diffusion` |
| screen64:turing:1 | 2 | mean_field_coupled_fft | 4 | 4 | True | True | 0.767879 | `brusselator_dt_sweep_cpu_replication.json#/cells/probe|screen64:turing:1|dt=2|mean_field_coupled_fft` |
| screen64:turing:1 | 2 | split_reaction_after_diffusion | 34 | 34 | True | True | 0.850996 | `brusselator_dt_sweep_cpu_replication.json#/cells/probe|screen64:turing:1|dt=2|split_reaction_after_diffusion` |

## Diagnostic CPU cost

The complete CPU replication took 1741.08 seconds. Source: `brusselator_dt_sweep_cpu_replication.json#/cpu_wall_seconds`.
