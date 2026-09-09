# MPX-135 — wszystkie przebiegi

Wygenerowane przez `tools/summarize_mpx.py` z `reports/mpx_verdicts.csv`
i `reports/mpx_trajectory.csv`. **Nie edytować ręcznie** — przebudować po
każdym ściągnięciu logów z klastra.

Stan `running` znaczy, że przebieg nie ma jeszcze linii werdyktu: liczby są
z ostatniego punktu pomiarowego, nie z wyniku końcowego. `a0_nc` i `a1_nc` to
klasy błędnej odpowiedzi — przy 135 bitach to one głodzą, więc sufit 0,75 albo
0,50 w kolumnie knowledge czyta się właśnie tam.

## Kodowanie kanoniczne (flip) · epsilon = 0.8

| ziarno | u_max | stan | próby | knowledge | accuracy | reguły | spec | a0_nc | a0_c | a1_nc | a1_c | godz. | prób/s | log |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 42 | 8 | TIME-LIMITED | 5 537 000 | 0.0000 | - | 0 | 0.00 | - | - | - | - | 6.0 | 256 | `135_s42_u8` |
| 42 | 9 | TIME-LIMITED | 105 621 000 | 0.0000 | - | 0 | 0.00 | - | - | - | - | 69.4 | 422 | `135_s42` |
| 42 | 9 | TIME-LIMITED | 9 509 500 | 0.0000 | - | 0 | 0.00 | - | - | - | - | 6.0 | 440 | `135_s42_addr` |
| 42 | 9 | TIME-LIMITED | 6 080 000 | 0.0000 | - | 0 | 0.00 | - | - | - | - | 6.0 | 281 | `135_s42_u9` |
| 42 | 9 | TIME-LIMITED | 324 500 | 0.0000 | - | 0 | 0.00 | - | - | - | - | 0.1 | 901 | `mpx_m2b_reach135.log` |
| 42 | 10 | TIME-LIMITED | 294 775 500 | 0.2717 | - | 170 | 8.97 | 0.0000 | 0.5273 | 0.0000 | 0.5596 | 166.7 | 491 | `135_s42_u10long` |
| 42 | 10 | TIME-LIMITED | 7 820 000 | 0.0000 | - | 0 | 0.00 | - | - | - | - | 6.0 | 362 | `135_s42_u10` |
| 42 | 11 | TIME-LIMITED | 655 383 500 | 0.7499 | - | 396 | 8.00 | - | - | - | - | 166.7 | 1092 | `135_s42_u11long` |
| 42 | 11 | TIME-LIMITED | 583 842 500 | 0.7499 | - | 395 | 8.00 | 0.0000 | 1.0000 | 1.0000 | 1.0000 | 166.7 | 973 | `135_s42_qdetail_u11` |
| 42 | 11 | TIME-LIMITED | 533 668 500 | 0.7499 | - | 388 | 8.00 | 0.0000 | 1.0000 | 1.0000 | 1.0000 | 166.7 | 889 | `135_s42_u11cover` |
| 42 | 11 | cancelled | 325 800 000 | 0.7499 | 1.0000 | 392 | 8.00 | 0.0000 | 1.0000 | 1.0000 | 1.0000 | 150.3 | 602 | `135_s42_acc_u11.cancelled` |
| 43 | 11 | TIME-LIMITED | 281 210 000 | 0.4980 | - | 257 | 8.05 | 0.0000 | 1.0000 | 0.0000 | 0.9922 | 166.7 | 469 | `135_s43_u11cover` |
| 44 | 11 | TIME-LIMITED | 256 199 500 | 0.4980 | - | 260 | 8.01 | 0.0000 | 1.0000 | 0.0000 | 0.9920 | 166.7 | 427 | `135_s44_u11cover` |
| 45 | 11 | TIME-LIMITED | 233 203 500 | 0.4918 | - | 261 | 8.11 | 0.0000 | 0.9671 | 0.0000 | 1.0000 | 166.7 | 389 | `135_s45_u11cover` |
| 42 | 12 | TIME-LIMITED | 366 848 000 | 0.5000 | - | 259 | 8.05 | 0.0000 | 1.0000 | 0.0000 | 1.0000 | 166.7 | 611 | `135_s42_qdetail_u12` |
| 42 | 12 | TIME-LIMITED | 339 107 500 | 0.5000 | - | 258 | 8.05 | - | - | - | - | 166.7 | 565 | `135_s42_u12long` |
| 42 | 12 | cancelled | 169 560 000 | 0.4821 | 0.9821 | 255 | 8.16 | 0.0000 | 0.9702 | 0.0000 | 0.9584 | 150.3 | 313 | `135_s42_acc_u12.cancelled` |
| 42 | 12 | TIME-LIMITED | 7 042 500 | 0.0129 | - | 226 | 13.33 | - | - | - | - | 6.0 | 326 | `135_s42_u12` |
| 43 | 12 | TIME-LIMITED | 353 092 500 | 0.4985 | - | 261 | 8.05 | - | - | - | - | 166.7 | 588 | `135_s43_u12long` |
| 42 | 13 | TIME-LIMITED | 158 236 000 | 0.4889 | - | 258 | 8.14 | - | - | - | - | 166.7 | 264 | `135_s42_u13long` |
| 42 | 14 | TIME-LIMITED | 25 090 000 | 0.1628 | - | 6861 | 15.20 | - | - | - | - | 166.7 | 42 | `135_s42_u14long` |
| 42 | 16 | TIME-LIMITED | 35 512 500 | 0.3242 | - | 8092 | 16.89 | - | - | - | - | 166.7 | 59 | `135_s42_u16long` |
| 42 | 16 | TIME-LIMITED | 3 955 000 | 0.0000 | - | 0 | 0.00 | - | - | - | - | 6.0 | 183 | `135_s42_u16` |
| 42 | 24 | TIME-LIMITED | 1 594 500 | 0.0000 | - | 0 | 0.00 | - | - | - | - | 6.0 | 74 | `135_s42_u24` |

## Kodowanie kanoniczne (flip) · epsilon = 0.8 · acs2er

| ziarno | u_max | stan | próby | knowledge | accuracy | reguły | spec | a0_nc | a0_c | a1_nc | a1_c | godz. | prób/s | log |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 42 | 11 | TIME-LIMITED | 1 894 500 | 0.0902 | - | 1753 | 13.14 | 0.0066 | 0.1869 | 0.0063 | 0.1611 | 166.7 | 3 | `135_s42_erfine_u11` |
| 42 | 12 | TIME-LIMITED | 1 235 500 | 0.0738 | - | 2530 | 13.86 | 0.0044 | 0.1406 | 0.0086 | 0.1414 | 166.7 | 2 | `135_s42_erfine_u12` |
| 42 | 12 | running | 120 000 | 0.0478 | - | 1190 | 13.15 | 0.0136 | 0.0869 | 0.0085 | 0.0824 | 21.9 | 2 | `135_s42_er_u12` |

## Kodowanie kanoniczne (flip) · epsilon = 1

| ziarno | u_max | stan | próby | knowledge | accuracy | reguły | spec | a0_nc | a0_c | a1_nc | a1_c | godz. | prób/s | log |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 42 | 11 | TIME-LIMITED | 301 483 000 | 0.7442 | - | 393 | 8.01 | 0.0000 | 0.9773 | 1.0000 | 1.0000 | 166.7 | 502 | `135_s42_eps1_u11` |
| 43 | 11 | **SUCCESS** | 427 920 000 | 1.0000 | - | 532 | 8.02 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 152.3 | 780 | `135_s43_eps1_u11` |

## Kodowanie zmienione (outcome) · epsilon = 0.8

| ziarno | u_max | stan | próby | knowledge | accuracy | reguły | spec | a0_nc | a0_c | a1_nc | a1_c | godz. | prób/s | log |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 42 | 9 | running | 12 840 000 | 0.1525 | - | 91 | 8.25 | 0.0000 | 0.6101 | 0.0000 | 0.0000 | 26.6 | 134 | `135_s42_outcome_u9` |
| 42 | 11 | **SUCCESS** | 43 200 000 | 1.0000 | - | 539 | 8.00 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 50.5 | 238 | `135_s42_outcome_u11` |
| 43 | 11 | **SUCCESS** | 46 800 000 | 1.0000 | - | 539 | 8.00 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 78.8 | 165 | `135_s43_outcome_u11` |
| 44 | 11 | cancelled | 4 800 000 | 0.1583 | 0.6495 | 423 | 10.92 | 0.0918 | 0.3379 | 0.1092 | 0.0944 | 79.0 | 17 | `135_s44_outcome_u11.cancelled` |
| 45 | 11 | **SUCCESS** | 55 560 000 | 1.0000 | 1.0000 | 532 | 8.00 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 80.9 | 191 | `135_s45_outcome_u11` |
| 46 | 11 | **SUCCESS** | 30 240 000 | 1.0000 | 1.0000 | 535 | 8.01 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 57.6 | 146 | `135_s46_outcome_u11` |

## Podsumowanie

- przebiegów w archiwum: **35**
- rozwiązanych (knowledge = 1,0): **5**
- najtańsze rozwiązanie: ziarno 46, 30 240 000 prób, kodowanie outcome, 57.6 h
