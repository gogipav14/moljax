# Brusselator `dt` sweep: physical true-residual check

CPU wall times are diagnostic only.

| State | dt | Preconditioner | Preconditioned stop | True stop | True residual at preconditioned stop | kappa_2(P) |
|---|---:|---|---:|---:|---:|---:|
| developed64:hopf:10 | 0.05 | fft_diffusion | 6 | 6 | 9.72976e-09 | 2.31072 |
| developed64:hopf:10 | 0.05 | identity | 10 | 10 | 5.12257e-09 | 1 |
| developed64:hopf:10 | 0.1 | fft_diffusion | 8 | 8 | 4.32714e-09 | 3.62144 |
| developed64:hopf:10 | 0.1 | identity | 15 | 15 | 4.06215e-09 | 1 |
| developed64:hopf:10 | 0.2 | fft_diffusion | 11 | 11 | 5.5659e-09 | 6.24288 |
| developed64:hopf:10 | 0.2 | identity | 23 | 23 | 7.63135e-09 | 1 |
| developed64:hopf:10 | 0.5 | fft_diffusion | 20 | 21 | 1.18711e-08 | 14.1072 |
| developed64:hopf:10 | 0.5 | identity | 51 | 51 | 9.02019e-09 | 1 |
| developed64:hopf:10 | 1.5 | fft_diffusion | 101 | 101 | n/a | 40.3216 |
| developed64:hopf:10 | 1.5 | identity | 101 | 101 | n/a | 1 |
| developed64:hopf:10 | 1 | fft_diffusion | 50 | 51 | 1.45955e-08 | 27.2144 |
| developed64:hopf:10 | 1 | identity | 101 | 101 | n/a | 1 |
| developed64:hopf:10 | 2 | fft_diffusion | 101 | 101 | n/a | 53.4288 |
| developed64:hopf:10 | 2 | identity | 101 | 101 | n/a | 1 |
| developed64:hopf:1 | 0.05 | fft_diffusion | 5 | 5 | 5.30939e-09 | 2.31072 |
| developed64:hopf:1 | 0.05 | identity | 13 | 13 | 3.73724e-09 | 1 |
| developed64:hopf:1 | 0.1 | fft_diffusion | 7 | 7 | 6.44176e-10 | 3.62144 |
| developed64:hopf:1 | 0.1 | identity | 18 | 18 | 3.88847e-09 | 1 |
| developed64:hopf:1 | 0.2 | fft_diffusion | 8 | 8 | 6.9621e-09 | 6.24288 |
| developed64:hopf:1 | 0.2 | identity | 26 | 26 | 8.47446e-09 | 1 |
| developed64:hopf:1 | 0.5 | fft_diffusion | 15 | 15 | 2.90437e-09 | 14.1072 |
| developed64:hopf:1 | 0.5 | identity | 53 | 53 | 6.20273e-09 | 1 |
| developed64:hopf:1 | 1.5 | fft_diffusion | 101 | 101 | n/a | 40.3216 |
| developed64:hopf:1 | 1.5 | identity | 101 | 101 | n/a | 1 |
| developed64:hopf:1 | 1 | fft_diffusion | 37 | 38 | 1.02754e-08 | 27.2144 |
| developed64:hopf:1 | 1 | identity | 101 | 101 | n/a | 1 |
| developed64:hopf:1 | 2 | fft_diffusion | 101 | 101 | n/a | 53.4288 |
| developed64:hopf:1 | 2 | identity | 101 | 101 | n/a | 1 |
| developed64:hopf:20 | 0.05 | fft_diffusion | 6 | 6 | 8.56078e-09 | 2.31072 |
| developed64:hopf:20 | 0.05 | identity | 10 | 10 | 3.01807e-09 | 1 |
| developed64:hopf:20 | 0.1 | fft_diffusion | 8 | 8 | 4.09966e-09 | 3.62144 |
| developed64:hopf:20 | 0.1 | identity | 14 | 14 | 5.76987e-09 | 1 |
| developed64:hopf:20 | 0.2 | fft_diffusion | 11 | 11 | 3.29524e-09 | 6.24288 |
| developed64:hopf:20 | 0.2 | identity | 22 | 22 | 5.0966e-09 | 1 |
| developed64:hopf:20 | 0.5 | fft_diffusion | 19 | 20 | 1.06878e-08 | 14.1072 |
| developed64:hopf:20 | 0.5 | identity | 45 | 45 | 7.17284e-09 | 1 |
| developed64:hopf:20 | 1.5 | fft_diffusion | 72 | 77 | 2.45445e-08 | 40.3216 |
| developed64:hopf:20 | 1.5 | identity | 101 | 101 | n/a | 1 |
| developed64:hopf:20 | 1 | fft_diffusion | 39 | 41 | 1.86638e-08 | 27.2144 |
| developed64:hopf:20 | 1 | identity | 91 | 91 | 8.79932e-09 | 1 |
| developed64:hopf:20 | 2 | fft_diffusion | 101 | 101 | n/a | 53.4288 |
| developed64:hopf:20 | 2 | identity | 101 | 101 | n/a | 1 |
| developed64:turing:120 | 0.05 | fft_diffusion | 6 | 6 | 7.54209e-10 | 7.5536 |
| developed64:turing:120 | 0.05 | identity | 14 | 14 | 8.12255e-09 | 1 |
| developed64:turing:120 | 0.1 | fft_diffusion | 7 | 7 | 2.23236e-09 | 14.1072 |
| developed64:turing:120 | 0.1 | identity | 22 | 22 | 8.26135e-09 | 1 |
| developed64:turing:120 | 0.2 | fft_diffusion | 9 | 9 | 3.95536e-09 | 27.2144 |
| developed64:turing:120 | 0.2 | identity | 36 | 36 | 8.06371e-09 | 1 |
| developed64:turing:120 | 0.5 | fft_diffusion | 14 | 15 | 1.05887e-08 | 66.536 |
| developed64:turing:120 | 0.5 | identity | 70 | 70 | 9.963e-09 | 1 |
| developed64:turing:120 | 1.5 | fft_diffusion | 34 | 37 | 2.66057e-08 | 197.608 |
| developed64:turing:120 | 1.5 | identity | 101 | 101 | n/a | 1 |
| developed64:turing:120 | 1 | fft_diffusion | 23 | 24 | 1.80243e-08 | 132.072 |
| developed64:turing:120 | 1 | identity | 101 | 101 | n/a | 1 |
| developed64:turing:120 | 2 | fft_diffusion | 48 | 51 | 3.89958e-08 | 263.144 |
| developed64:turing:120 | 2 | identity | 101 | 101 | n/a | 1 |
| developed64:turing:200 | 0.05 | fft_diffusion | 5 | 5 | 8.14614e-09 | 7.5536 |
| developed64:turing:200 | 0.05 | identity | 9 | 9 | 6.55293e-09 | 1 |
| developed64:turing:200 | 0.1 | fft_diffusion | 7 | 7 | 2.51619e-09 | 14.1072 |
| developed64:turing:200 | 0.1 | identity | 14 | 14 | 7.90739e-09 | 1 |
| developed64:turing:200 | 0.2 | fft_diffusion | 8 | 9 | 1.36945e-08 | 27.2144 |
| developed64:turing:200 | 0.2 | identity | 22 | 22 | 8.59932e-09 | 1 |
| developed64:turing:200 | 0.5 | fft_diffusion | 12 | 13 | 2.33775e-08 | 66.536 |
| developed64:turing:200 | 0.5 | identity | 44 | 44 | 8.43906e-09 | 1 |
| developed64:turing:200 | 1.5 | fft_diffusion | 27 | 30 | 3.40169e-08 | 197.608 |
| developed64:turing:200 | 1.5 | identity | 96 | 96 | 9.8407e-09 | 1 |
| developed64:turing:200 | 1 | fft_diffusion | 20 | 21 | 2.0661e-08 | 132.072 |
| developed64:turing:200 | 1 | identity | 73 | 73 | 8.43826e-09 | 1 |
| developed64:turing:200 | 2 | fft_diffusion | 36 | 39 | 2.5303e-08 | 263.144 |
| developed64:turing:200 | 2 | identity | 101 | 101 | n/a | 1 |
| developed64:turing:80 | 0.05 | fft_diffusion | 5 | 5 | 2.01647e-10 | 7.5536 |
| developed64:turing:80 | 0.05 | identity | 6 | 6 | 6.50486e-09 | 1 |
| developed64:turing:80 | 0.1 | fft_diffusion | 5 | 5 | 7.1575e-09 | 14.1072 |
| developed64:turing:80 | 0.1 | identity | 11 | 11 | 5.6757e-09 | 1 |
| developed64:turing:80 | 0.2 | fft_diffusion | 7 | 7 | 2.00031e-09 | 27.2144 |
| developed64:turing:80 | 0.2 | identity | 20 | 20 | 8.76234e-09 | 1 |
| developed64:turing:80 | 0.5 | fft_diffusion | 11 | 11 | 6.33289e-09 | 66.536 |
| developed64:turing:80 | 0.5 | identity | 43 | 43 | 9.66648e-09 | 1 |
| developed64:turing:80 | 1.5 | fft_diffusion | 24 | 25 | 1.5646e-08 | 197.608 |
| developed64:turing:80 | 1.5 | identity | 101 | 101 | n/a | 1 |
| developed64:turing:80 | 1 | fft_diffusion | 18 | 19 | 1.08752e-08 | 132.072 |
| developed64:turing:80 | 1 | identity | 80 | 80 | 8.89448e-09 | 1 |
| developed64:turing:80 | 2 | fft_diffusion | 31 | 34 | 2.38431e-08 | 263.144 |
| developed64:turing:80 | 2 | identity | 101 | 101 | n/a | 1 |
| screen64:hopf:0 | 0.05 | fft_diffusion | 5 | 5 | 4.61626e-09 | 2.31072 |
| screen64:hopf:0 | 0.05 | identity | 13 | 13 | 2.82359e-09 | 1 |
| screen64:hopf:0 | 0.1 | fft_diffusion | 6 | 6 | 5.47982e-09 | 3.62144 |
| screen64:hopf:0 | 0.1 | identity | 18 | 18 | 4.23976e-09 | 1 |
| screen64:hopf:0 | 0.2 | fft_diffusion | 8 | 8 | 3.89399e-09 | 6.24288 |
| screen64:hopf:0 | 0.2 | identity | 25 | 25 | 7.80433e-09 | 1 |
| screen64:hopf:0 | 0.5 | fft_diffusion | 14 | 14 | 2.722e-09 | 14.1072 |
| screen64:hopf:0 | 0.5 | identity | 49 | 49 | 7.22967e-09 | 1 |
| screen64:hopf:0 | 1.5 | fft_diffusion | 101 | 101 | n/a | 40.3216 |
| screen64:hopf:0 | 1.5 | identity | 101 | 101 | n/a | 1 |
| screen64:hopf:0 | 1 | fft_diffusion | 35 | 34 | 2.62814e-09 | 27.2144 |
| screen64:hopf:0 | 1 | identity | 101 | 101 | n/a | 1 |
| screen64:hopf:0 | 2 | fft_diffusion | 101 | 101 | n/a | 53.4288 |
| screen64:hopf:0 | 2 | identity | 101 | 101 | n/a | 1 |
| screen64:hopf:1 | 0.05 | fft_diffusion | 5 | 5 | 4.91158e-09 | 2.31072 |
| screen64:hopf:1 | 0.05 | identity | 13 | 13 | 3.9185e-09 | 1 |
| screen64:hopf:1 | 0.1 | fft_diffusion | 7 | 7 | 5.10327e-10 | 3.62144 |
| screen64:hopf:1 | 0.1 | identity | 18 | 18 | 5.10735e-09 | 1 |
| screen64:hopf:1 | 0.2 | fft_diffusion | 8 | 8 | 5.6341e-09 | 6.24288 |
| screen64:hopf:1 | 0.2 | identity | 26 | 26 | 7.49637e-09 | 1 |
| screen64:hopf:1 | 0.5 | fft_diffusion | 14 | 14 | 5.33281e-09 | 14.1072 |
| screen64:hopf:1 | 0.5 | identity | 50 | 50 | 9.41024e-09 | 1 |
| screen64:hopf:1 | 1.5 | fft_diffusion | 101 | 101 | n/a | 40.3216 |
| screen64:hopf:1 | 1.5 | identity | 101 | 101 | n/a | 1 |
| screen64:hopf:1 | 1 | fft_diffusion | 37 | 35 | 2.90928e-09 | 27.2144 |
| screen64:hopf:1 | 1 | identity | 101 | 101 | n/a | 1 |
| screen64:hopf:1 | 2 | fft_diffusion | 101 | 101 | n/a | 53.4288 |
| screen64:hopf:1 | 2 | identity | 101 | 101 | n/a | 1 |
| screen64:turing:0 | 0.05 | fft_diffusion | 6 | 6 | 1.64826e-10 | 7.5536 |
| screen64:turing:0 | 0.05 | identity | 25 | 25 | 5.04381e-09 | 1 |
| screen64:turing:0 | 0.1 | fft_diffusion | 7 | 6 | 5.62506e-10 | 14.1072 |
| screen64:turing:0 | 0.1 | identity | 34 | 34 | 7.86249e-09 | 1 |
| screen64:turing:0 | 0.2 | fft_diffusion | 8 | 8 | 3.53864e-09 | 27.2144 |
| screen64:turing:0 | 0.2 | identity | 49 | 49 | 7.31772e-09 | 1 |
| screen64:turing:0 | 0.5 | fft_diffusion | 14 | 13 | 7.6944e-10 | 66.536 |
| screen64:turing:0 | 0.5 | identity | 80 | 80 | 9.33006e-09 | 1 |
| screen64:turing:0 | 1.5 | fft_diffusion | 30 | 27 | 1.03835e-09 | 197.608 |
| screen64:turing:0 | 1.5 | identity | 101 | 101 | n/a | 1 |
| screen64:turing:0 | 1 | fft_diffusion | 22 | 20 | 7.89683e-10 | 132.072 |
| screen64:turing:0 | 1 | identity | 101 | 101 | n/a | 1 |
| screen64:turing:0 | 2 | fft_diffusion | 38 | 35 | 1.4729e-09 | 263.144 |
| screen64:turing:0 | 2 | identity | 101 | 101 | n/a | 1 |
| screen64:turing:1 | 0.05 | fft_diffusion | 6 | 6 | 3.42483e-10 | 7.5536 |
| screen64:turing:1 | 0.05 | identity | 25 | 25 | 6.42348e-09 | 1 |
| screen64:turing:1 | 0.1 | fft_diffusion | 7 | 7 | 1.23679e-09 | 14.1072 |
| screen64:turing:1 | 0.1 | identity | 35 | 35 | 6.95717e-09 | 1 |
| screen64:turing:1 | 0.2 | fft_diffusion | 9 | 8 | 1.04151e-09 | 27.2144 |
| screen64:turing:1 | 0.2 | identity | 50 | 50 | 8.64536e-09 | 1 |
| screen64:turing:1 | 0.5 | fft_diffusion | 14 | 13 | 2.03086e-09 | 66.536 |
| screen64:turing:1 | 0.5 | identity | 86 | 86 | 8.46539e-09 | 1 |
| screen64:turing:1 | 1.5 | fft_diffusion | 30 | 28 | 2.43483e-09 | 197.608 |
| screen64:turing:1 | 1.5 | identity | 101 | 101 | n/a | 1 |
| screen64:turing:1 | 1 | fft_diffusion | 22 | 21 | 2.10151e-09 | 132.072 |
| screen64:turing:1 | 1 | identity | 101 | 101 | n/a | 1 |
| screen64:turing:1 | 2 | fft_diffusion | 38 | 37 | 3.77132e-09 | 263.144 |
| screen64:turing:1 | 2 | identity | 101 | 101 | n/a | 1 |
| developed64:hopf:10 | 0.2 | mean_field_coupled_fft | 9 | 9 | 2.51822e-09 | 10.0156 |
| developed64:hopf:10 | 0.2 | split_reaction_after_diffusion | 10 | 11 | 2.38272e-08 | unavailable |
| developed64:hopf:10 | 1 | mean_field_coupled_fft | 23 | 23 | 5.71445e-09 | 83.8121 |
| developed64:hopf:10 | 1 | split_reaction_after_diffusion | 101 | 101 | n/a | unavailable |
| developed64:hopf:10 | 2 | mean_field_coupled_fft | 57 | 58 | 1.29398e-08 | 143.801 |
| developed64:hopf:10 | 2 | split_reaction_after_diffusion | 101 | 101 | n/a | unavailable |
| developed64:hopf:1 | 0.2 | mean_field_coupled_fft | 4 | 4 | 1.18522e-11 | 11.7861 |
| developed64:hopf:1 | 0.2 | split_reaction_after_diffusion | 8 | 8 | 8.8557e-09 | unavailable |
| developed64:hopf:1 | 1 | mean_field_coupled_fft | 5 | 5 | 3.28067e-10 | 243.059 |
| developed64:hopf:1 | 1 | split_reaction_after_diffusion | 42 | 47 | 7.8479e-08 | unavailable |
| developed64:hopf:1 | 2 | mean_field_coupled_fft | 11 | 12 | 2.51726e-08 | 34604.6 |
| developed64:hopf:1 | 2 | split_reaction_after_diffusion | 101 | 101 | n/a | unavailable |
| developed64:hopf:20 | 0.2 | mean_field_coupled_fft | 9 | 9 | 1.61303e-09 | 9.24934 |
| developed64:hopf:20 | 0.2 | split_reaction_after_diffusion | 10 | 11 | 2.40321e-08 | unavailable |
| developed64:hopf:20 | 1 | mean_field_coupled_fft | 18 | 18 | 9.66335e-09 | 60.7395 |
| developed64:hopf:20 | 1 | split_reaction_after_diffusion | 88 | 101 | 3.00553e-07 | unavailable |
| developed64:hopf:20 | 2 | mean_field_coupled_fft | 29 | 30 | 1.75789e-08 | 88.9647 |
| developed64:hopf:20 | 2 | split_reaction_after_diffusion | 101 | 101 | n/a | unavailable |
| developed64:turing:120 | 0.2 | mean_field_coupled_fft | 6 | 6 | 3.83316e-09 | 31.6946 |
| developed64:turing:120 | 0.2 | split_reaction_after_diffusion | 8 | 9 | 5.76233e-08 | unavailable |
| developed64:turing:120 | 1 | mean_field_coupled_fft | 10 | 11 | 1.46243e-08 | 172.749 |
| developed64:turing:120 | 1 | split_reaction_after_diffusion | 28 | 35 | 5.39272e-07 | unavailable |
| developed64:turing:120 | 2 | mean_field_coupled_fft | 15 | 16 | 1.49479e-08 | 355.971 |
| developed64:turing:120 | 2 | split_reaction_after_diffusion | 60 | 79 | 1.8971e-06 | unavailable |
| developed64:turing:200 | 0.2 | mean_field_coupled_fft | 6 | 6 | 5.86375e-09 | 27.6912 |
| developed64:turing:200 | 0.2 | split_reaction_after_diffusion | 10 | 13 | 2.35064e-07 | unavailable |
| developed64:turing:200 | 1 | mean_field_coupled_fft | 10 | 11 | 1.19939e-08 | 105.574 |
| developed64:turing:200 | 1 | split_reaction_after_diffusion | 31 | 44 | 1.37048e-06 | unavailable |
| developed64:turing:200 | 2 | mean_field_coupled_fft | 13 | 14 | 2.6691e-08 | 155.175 |
| developed64:turing:200 | 2 | split_reaction_after_diffusion | 52 | 81 | 7.0814e-06 | unavailable |
| developed64:turing:80 | 0.2 | mean_field_coupled_fft | 4 | 4 | 1.10896e-10 | 31.8516 |
| developed64:turing:80 | 0.2 | split_reaction_after_diffusion | 5 | 6 | 1.20568e-08 | unavailable |
| developed64:turing:80 | 1 | mean_field_coupled_fft | 5 | 5 | 3.74509e-10 | 176.424 |
| developed64:turing:80 | 1 | split_reaction_after_diffusion | 13 | 16 | 1.37708e-07 | unavailable |
| developed64:turing:80 | 2 | mean_field_coupled_fft | 5 | 5 | 7.69982e-09 | 371.614 |
| developed64:turing:80 | 2 | split_reaction_after_diffusion | 23 | 31 | 3.61923e-07 | unavailable |
| screen64:hopf:0 | 0.2 | mean_field_coupled_fft | 4 | 4 | 1.0416e-11 | 11.7856 |
| screen64:hopf:0 | 0.2 | split_reaction_after_diffusion | 8 | 8 | 7.04954e-09 | unavailable |
| screen64:hopf:0 | 1 | mean_field_coupled_fft | 4 | 4 | 2.56679e-09 | 243.023 |
| screen64:hopf:0 | 1 | split_reaction_after_diffusion | 41 | 45 | 2.3195e-08 | unavailable |
| screen64:hopf:0 | 2 | mean_field_coupled_fft | 9 | 9 | 8.21541e-09 | 36154.2 |
| screen64:hopf:0 | 2 | split_reaction_after_diffusion | 101 | 101 | n/a | unavailable |
| screen64:hopf:1 | 0.2 | mean_field_coupled_fft | 4 | 4 | 6.69021e-12 | 11.7855 |
| screen64:hopf:1 | 0.2 | split_reaction_after_diffusion | 8 | 8 | 9.44357e-09 | unavailable |
| screen64:hopf:1 | 1 | mean_field_coupled_fft | 4 | 4 | 3.66523e-09 | 243.021 |
| screen64:hopf:1 | 1 | split_reaction_after_diffusion | 42 | 45 | 4.45811e-08 | unavailable |
| screen64:hopf:1 | 2 | mean_field_coupled_fft | 9 | 9 | 9.15649e-09 | 36214.1 |
| screen64:hopf:1 | 2 | split_reaction_after_diffusion | 101 | 101 | n/a | unavailable |
| screen64:turing:0 | 0.2 | mean_field_coupled_fft | 3 | 3 | 1.6078e-09 | 31.8527 |
| screen64:turing:0 | 0.2 | split_reaction_after_diffusion | 8 | 8 | 6.10127e-09 | unavailable |
| screen64:turing:0 | 1 | mean_field_coupled_fft | 4 | 4 | 2.91506e-11 | 176.447 |
| screen64:turing:0 | 1 | split_reaction_after_diffusion | 20 | 22 | 1.59072e-08 | unavailable |
| screen64:turing:0 | 2 | mean_field_coupled_fft | 4 | 4 | 2.41663e-10 | 371.719 |
| screen64:turing:0 | 2 | split_reaction_after_diffusion | 34 | 39 | 5.168e-08 | unavailable |
| screen64:turing:1 | 0.2 | mean_field_coupled_fft | 3 | 3 | 2.00364e-09 | 31.8527 |
| screen64:turing:1 | 0.2 | split_reaction_after_diffusion | 8 | 9 | 1.44494e-08 | unavailable |
| screen64:turing:1 | 1 | mean_field_coupled_fft | 4 | 4 | 3.1408e-11 | 176.447 |
| screen64:turing:1 | 1 | split_reaction_after_diffusion | 20 | 23 | 3.87197e-08 | unavailable |
| screen64:turing:1 | 2 | mean_field_coupled_fft | 4 | 4 | 1.91178e-10 | 371.717 |
| screen64:turing:1 | 2 | split_reaction_after_diffusion | 34 | 39 | 1.3105e-07 | unavailable |

## Decision under the physical true-residual convention

D1: **PASS**, median ratio 4.4625.
D2: **PASS**, median 31 iterations.
D3: **PASS**; decision **BUILD**.
D1/D2/D3 match the preconditioned convention: True/True/True. Robustness criterion met: **True**.

## Developed Hopf mean-field cells at dt = 2

| State | Preconditioned stop | True stop | True residual at preconditioned stop |
|---|---:|---:|---:|
| developed64:hopf:1 | 11 | 12 | 2.51726e-08 |
| developed64:hopf:10 | 57 | 58 | 1.29398e-08 |
| developed64:hopf:20 | 29 | 30 | 1.75789e-08 |

The run took 354.167 CPU seconds (diagnostic only).
