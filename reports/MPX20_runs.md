# MPX-20 — wszystkie przebiegi

Wygenerowane przez `tools/summarize_mpx.py` z `reports/mpx_verdicts.csv`
i `reports/mpx_trajectory.csv`. **Nie edytować ręcznie** — przebudować po
każdym ściągnięciu logów z klastra.

Stan `running` znaczy, że przebieg nie ma jeszcze linii werdyktu, a `cancelled`
albo `partial`, że został zatrzymany — w obu wypadkach liczby pochodzą
z ostatniego punktu pomiarowego, nie z wyniku końcowego.

`a0_nc` i `a1_nc` to klasy błędnej odpowiedzi. To one głodzą, więc sufit
w kolumnie knowledge czyta się właśnie tam: dwie klasy puste dają 0,50,
jedna 0,75. Pusta kolumna znaczy, że przebieg biegł bez `--log-coverage`.

## Kodowanie kanoniczne (flip) · epsilon = 0.8

| ziarno | u_max | stan | próby | knowledge | accuracy | reguły | spec | a0_nc | a0_c | a1_nc | a1_c | godz. | prób/s | log |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 42 | 5 | **SUCCESS** | 160 000 | 1.0000 | - | 95 | 5.00 | - | - | - | - | 0.0 | 9195 | `20_s42_u5` |
| 42 | 6 | **SUCCESS** | 120 000 | 1.0000 | - | 93 | 5.00 | - | - | - | - | 0.0 | 15385 | `20_s42_addr` |
| 42 | 6 | **SUCCESS** | 80 000 | 1.0000 | - | 108 | 5.09 | - | - | - | - | 0.0 | 11594 | `20_s42_u6` |
| 42 | 7 | **SUCCESS** | 130 000 | 1.0000 | - | 92 | 5.01 | - | - | - | - | 0.0 | 12871 | `20_s42_u7` |
| 42 | 8 | **SUCCESS** | 150 000 | 1.0000 | - | 94 | 5.15 | - | - | - | - | 0.0 | 14151 | `20_s42_u8` |
| 42 | 9 | **SUCCESS** | 110 000 | 1.0000 | - | 105 | 5.30 | - | - | - | - | 0.0 | 12360 | `20_s42_u9` |
| 42 | 10 | **SUCCESS** | 120 000 | 1.0000 | - | 102 | 5.42 | - | - | - | - | 0.0 | 12500 | `20_s42_u10` |

## Podsumowanie

- przebiegów w archiwum: **7**
- rozwiązanych (knowledge = 1,0): **7**
- najmniej prób: ziarno 42, 80 000 prób, 0.0 h, kodowanie flip, epsilon 0.8, acs2
