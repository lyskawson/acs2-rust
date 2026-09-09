# MPX-70 — wszystkie przebiegi

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
| 42 | 8 | **SUCCESS** | 17 880 000 | 1.0000 | - | 277 | 7.04 | - | - | - | - | 1.7 | 2929 | `mpx_m3_e1_traj70_pyalcs.log` |
| 42 | 8 | **SUCCESS** | 17 880 000 | 1.0000 | - | 277 | 7.04 | - | - | - | - | 2.4 | 2104 | `70_s42_addr` |
| 42 | 8 | **SUCCESS** | 17 880 000 | 1.0000 | - | 277 | 7.04 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 2.2 | 2234 | `70_s42_cover` |
| 42 | 8 | **SUCCESS** | 17 880 000 | 1.0000 | - | 277 | 7.04 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 2.9 | 1696 | `70_s42_qdetail` |
| 42 | 8 | TIME-LIMITED | 1 625 500 | 0.2406 | - | 110 | 8.63 | - | - | - | - | 0.2 | 2709 | `mpx_m2b_reach70.log` |
| 43 | 8 | **SUCCESS** | 17 820 000 | 1.0000 | - | 269 | 7.00 | - | - | - | - | 3.8 | 1301 | `70_s43` |
| 43 | 8 | **SUCCESS** | 17 820 000 | 1.0000 | - | 269 | 7.00 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 3.7 | 1355 | `70_s43_cover` |
| 43 | 8 | cancelled | 4 740 000 | 0.2885 | - | 89 | 7.92 | - | - | - | - | 1.5 | 890 | `70_s43.cap40k.cancelled` |
| 43 | 8 | TIME-LIMITED | 1 555 000 | 0.1349 | - | 75 | 9.20 | - | - | - | - | 0.2 | 2591 | `mpx_m2b_reach70.log` |
| 44 | 8 | **SUCCESS** | 44 580 000 | 1.0000 | - | 274 | 7.00 | - | - | - | - | 8.1 | 1538 | `70_s44` |
| 44 | 8 | cancelled | 5 280 000 | 0.2932 | - | 90 | 7.87 | - | - | - | - | 1.5 | 988 | `70_s44.cap40k.cancelled` |
| 44 | 8 | TIME-LIMITED | 1 503 000 | 0.1419 | - | 91 | 9.79 | - | - | - | - | 0.2 | 2505 | `mpx_m2b_reach70.log` |
| 45 | 8 | **SUCCESS** | 21 300 000 | 1.0000 | - | 271 | 7.00 | - | - | - | - | 3.2 | 1847 | `70_s45` |
| 46 | 8 | **SUCCESS** | 66 420 000 | 1.0000 | - | 268 | 7.00 | - | - | - | - | 12.9 | 1436 | `70_s46` |
| 42 | 9 | TIME-LIMITED | 5 714 000 | 0.4001 | - | 114 | 7.21 | - | - | - | - | 2.0 | 793 | `mpx_m3_e1_traj70_butz.log` |

## Kodowanie kanoniczne (flip) · epsilon = 0.8 · acs2er

| ziarno | u_max | stan | próby | knowledge | accuracy | reguły | spec | a0_nc | a0_c | a1_nc | a1_c | godz. | prób/s | log |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 42 | 8 | **SUCCESS** | 12 120 000 | 1.0000 | - | 268 | 7.00 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 4.3 | 779 | `70_s42_er1` |
| 42 | 8 | **SUCCESS** | 3 960 000 | 1.0000 | - | 318 | 7.00 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 160.1 | 7 | `70_s42_er` |
| 43 | 8 | **SUCCESS** | 22 860 000 | 1.0000 | - | 270 | 7.00 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 3.8 | 1665 | `70_s43_er1` |

## Kodowanie zmienione (outcome) · epsilon = 0.8

| ziarno | u_max | stan | próby | knowledge | accuracy | reguły | spec | a0_nc | a0_c | a1_nc | a1_c | godz. | prób/s | log |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 42 | 8 | **SUCCESS** | 4 680 000 | 1.0000 | - | 276 | 7.01 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.6 | 2171 | `70_s42_outcome` |
| 43 | 8 | **SUCCESS** | 62 340 000 | 1.0000 | 1.0000 | 280 | 7.05 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 3.8 | 4532 | `70_s43_outcome` |
| 44 | 8 | **SUCCESS** | 10 260 000 | 1.0000 | 1.0000 | 276 | 7.00 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.8 | 3596 | `70_s44_outcome` |

## Podsumowanie

- przebiegów w archiwum: **21**
- rozwiązanych (knowledge = 1,0): **15**
- najmniej prób: ziarno 42, 3 960 000 prób, 160.1 h, kodowanie flip, epsilon 0.8, acs2er
- najkrótszy czas: ziarno 42, 4 680 000 prób, 0.6 h, kodowanie outcome, epsilon 0.8, acs2
