# MPX-264 — wszystkie przebiegi

Wygenerowane przez `tools/summarize_mpx.py` z `reports/mpx_verdicts.csv`
i `reports/mpx_trajectory.csv`. **Nie edytować ręcznie** — przebudować po
każdym ściągnięciu logów z klastra.

Stan `running` znaczy, że przebieg nie ma jeszcze linii werdyktu, a `cancelled`
albo `partial`, że został zatrzymany — w obu wypadkach liczby pochodzą
z ostatniego punktu pomiarowego, nie z wyniku końcowego.

`a0_nc` i `a1_nc` to klasy błędnej odpowiedzi. To one głodzą, więc sufit
w kolumnie knowledge czyta się właśnie tam: dwie klasy puste dają 0,50,
jedna 0,75. Pusta kolumna znaczy, że przebieg biegł bez `--log-coverage`.

## Kodowanie zmienione (outcome) · epsilon = 0.8

| ziarno | u_max | stan | próby | knowledge | accuracy | reguły | spec | a0_nc | a0_c | a1_nc | a1_c | godz. | prób/s | log |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 42 | 12 | TIME-LIMITED | 1 165 000 | 0.0000 | 0.5050 | 0 | 0.00 | - | - | - | - | 12.0 | 27 | `264_s42_probe264` |

## Podsumowanie

- przebiegów w archiwum: **1**
- rozwiązanych (knowledge = 1,0): **0**
