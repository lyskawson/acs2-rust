# MPX-37 — wszystkie przebiegi

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
| 42 | 6 | **SUCCESS** | 1 020 000 | 1.0000 | - | 151 | 6.00 | - | - | - | - | 0.1 | 4909 | `37_s42_u6` |
| 42 | 7 | **SUCCESS** | 780 000 | 1.0000 | - | 148 | 6.00 | - | - | - | - | 0.0 | 6070 | `37_s42_addr` |
| 42 | 7 | **SUCCESS** | 780 000 | 1.0000 | - | 148 | 6.00 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.0 | 6367 | `37_s42_cover` |
| 42 | 7 | **SUCCESS** | 690 000 | 1.0000 | - | 151 | 6.03 | - | - | - | - | 0.0 | 4829 | `37_s42_u7` |
| 42 | 7 | **SUCCESS** | 678 000 | 1.0000 | - | 144 | 6.03 | - | - | - | - | 0.0 | 9469 | `mpx_m2b_reach37.log` |
| 43 | 7 | **SUCCESS** | 1 116 000 | 1.0000 | - | 156 | 6.05 | - | - | - | - | 0.0 | 11171 | `mpx_m2b_reach37.log` |
| 44 | 7 | **SUCCESS** | 882 000 | 1.0000 | - | 144 | 6.01 | - | - | - | - | 0.0 | 11293 | `mpx_m2b_reach37.log` |
| 42 | 8 | **SUCCESS** | 900 000 | 1.0000 | - | 151 | 6.03 | - | - | - | - | 0.1 | 3600 | `37_s42_u8` |
| 42 | 9 | **SUCCESS** | 900 000 | 1.0000 | - | 152 | 6.02 | - | - | - | - | 0.1 | 2150 | `37_s42_u9` |
| 42 | 10 | **SUCCESS** | 810 000 | 1.0000 | - | 151 | 6.22 | - | - | - | - | 0.1 | 2286 | `37_s42_u10` |
| 42 | 11 | **SUCCESS** | 870 000 | 1.0000 | - | 148 | 6.24 | - | - | - | - | 0.2 | 1478 | `37_s42_u11` |
| 42 | 12 | **SUCCESS** | 900 000 | 1.0000 | - | 164 | 6.85 | - | - | - | - | 0.1 | 2361 | `37_s42_u12` |

## Kodowanie kanoniczne (flip) · epsilon = 0.8 · acs2er

| ziarno | u_max | stan | próby | knowledge | accuracy | reguły | spec | a0_nc | a0_c | a1_nc | a1_c | godz. | prób/s | log |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 42 | 7 | **SUCCESS** | 960 000 | 1.0000 | - | 144 | 6.00 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.1 | 3442 | `37_s42_er1` |
| 42 | 7 | **SUCCESS** | 300 000 | 1.0000 | - | 219 | 6.01 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 3.4 | 25 | `37_s42_er` |
| 43 | 7 | **SUCCESS** | 660 000 | 1.0000 | - | 160 | 6.05 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.1 | 2291 | `37_s43_er1` |

## Podsumowanie

- przebiegów w archiwum: **15**
- rozwiązanych (knowledge = 1,0): **15**
- najmniej prób: ziarno 42, 300 000 prób, 3.4 h, kodowanie flip, epsilon 0.8, acs2er
- najkrótszy czas: ziarno 42, 678 000 prób, 0.0 h, kodowanie flip, epsilon 0.8, acs2
