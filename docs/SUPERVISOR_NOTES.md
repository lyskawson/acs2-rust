# Notatka robocza — materiał do maila do promotora

Stan na moment ostatniej aktualizacji. Notatka służy do złożenia wiadomości, nie jest
raportem naukowym — pełny raport to `reports/MPX_final.md`. Liczby poniżej są
odtwarzalne z `reports/mpx_{trajectory,diagnostics,verdicts}.csv`.

## Nagłówek: multiplekser 70-bitowy rozwiązany, na wszystkich seedach

ACS2 z kanoniczną generalizacją ALP osiąga `knowledge = 1.0` na MPX-70 na **5 z 5**
próbowanych seedów. Literatura dla ACS/ACS2 kończy się na 20–37 bitach.

| seed | próby do sukcesu | reguły wiarygodne | specyficzność (ideał 7) |
|------|------------------|-------------------|--------------------------|
| 42 | 17 880 000 | 277 | 7,04 |
| 43 | 17 820 000 | 269 | 7,00 |
| 44 | 44 580 000 | 274 | 7,00 |
| 45 | 21 300 000 | 271 | 7,00 |
| 46 | 66 420 000 | 268 | 7,00 |

Rozwiązanie jest **niezmiennicze**, koszt nie: rozrzut 3,73×. Każdy seed ląduje na tej
samej strukturze — 268–277 reguł o idealnej specyficzności 7.

Ostrzeżenie metodologiczne warte wspomnienia: seed 46 stał na `knowledge ≈ 0,745` przez
**29,9 mln kolejnych prób** (45% swojego przebiegu), po czym doszedł do 1,0. Eksperyment
ucięty wcześniej zgłosiłby pewny, fałszywy wynik negatywny.

## Multiplekser 135-bitowy: z zera do 0,75

Przy kanonicznej wyprowadzonej wartości `u_max = 9` MPX-135 daje **twarde zero** —
105,6 mln prób, ani jednej wiarygodnej reguły, populacja bez trendu.

Diagnoza wskazała podejrzanego: reguły osiadały na specyficzności 8,8–9,6, czyli
*dokładnie na limicie*. Rozluźnienie limitu przełamuje granicę:

| `u_max` | knowledge | reguły wiarygodne | specyficzność (ideał 8) |
|---------|-----------|-------------------|--------------------------|
| 9 (kanoniczne) | 0,0000 | 0 | — |
| 11 | **0,7499** | 394 | 8,00 |
| 12 | 0,5000 | 259 | 8,05 |
| 12 (drugi seed) | 0,4985 | 261 | 8,05 |

Kanoniczny próg `a+2` jest przy 135 atrybutach za ciasny.

## Dlaczego zatrzymuje się na 0,75 — zmierzone

Sufity są dokładnymi ułamkami, bo brakuje **całych klas przejść**. Rozbicie metryki na
cztery klasy (akcja × czy przejście zmienia bit walidacyjny) przy `u_max = 11`:

- „akcja 0, zmiana" — 0,99
- „akcja 1, brak zmiany" — 1,00
- „akcja 1, zmiana" — 1,00
- **„akcja 0, brak zmiany" — 0,0000, w każdym punkcie przez 219 mln prób**

Nie istnieje ani jedna wiarygodna reguła przewidująca „nic się nie stanie" dla tej klasy.
Nie ma tam też reguł **błędnych** — klasa jest pusta. To porażka odkrywania reguł, nie
progu wiarygodności.

Kontrola na MPX-70, który dochodzi do 1,0, przechodzi przez ten sam kształt (jedna klasa
na 0,11, reszta blisko 1,0) i dopiero potem ją wypełnia. Czyli głodzona klasa to normalny
stan przejściowy — przy 135 bitach po prostu nigdy się nie rozwiązuje.

Hipoteza, **jeszcze niezmierzona**: reguła przewidująca brak zmiany ma efekt z samych
wildcardów, czyli jest stanem domyślnym, i musi zostać *zawężona* do swojej dziedziny,
podczas gdy reguły przewidujące zmianę ALP buduje wprost, kierunkowo.

## Kontrola metodologiczna

Zarzut, że wyprowadzony `u_max` przemyca wiedzę o rozwiązaniu, jest sprawdzony i odparty:
na MPX-20 (wartości 5–10) i MPX-37 (6–12) **każda** próbowana wartość rozwiązuje zadanie
przy idealnej specyficzności. Wyprowadzona wartość jest tylko najszybsza. `u_max` jest
obojętny tam, gdzie zadanie jest wykonalne, i rozstrzygający tam, gdzie nie jest.

## Sprostowanie do maila z 1.09.2026

W mailu napisałem, że wariant ACS2ER z jednym odtworzeniem uczy się wolniej od
podstawowego ACS2. **Nasze dane tego nie potwierdzają** — opierałem się na jednym
porównaniu. Przy k=70: ziarno 42 dało ER szybszy (12,12 vs 17,88 mln prób), ziarno 43
wolniejszy (22,86 vs 17,82 mln). Średnie praktycznie równe. Przy zmierzonym rozrzucie
ziaren 3,73x dwa ziarna niczego nie rozstrzygają. Właściwe sformułowanie: przy równym
nakładzie obliczeń różnicy na razie nie widać.

## Obraz sufitu jest szerszy, niż napisałem

W mailu podałem, że brakuje jednej klasy. Pełniejszy obraz: głodzone są **obie klasy
błędnej odpowiedzi**. Ziarno 42 jako jedyne wypełniło jedną z nich — stąd 0,75. Ziarna
43, 44 i 45 mają obie na zerze i zmierzają do 0,50. Wynik 0,75 jest więc **odstępstwem,
nie normą**, i tak trzeba go przedstawić.

## Co dalej

ACS2ER jest zaimplementowany i zwalidowany różnicowo przeciw pyalcs. Trwają
eksperymenty rozstrzygające, czy głodzenie bierze się z kodowania problemu (błędna
odpowiedź nie zmienia percepcji, więc jej reguła musi być zawężana, a nie budowana),
czy z obciążenia eksploracji (zachłanna gałąź ACS2 wybiera akcje antycypujące zmianę,
czyli omija błędne odpowiedzi). Osobno mierzona jest **dokładność odpowiedzi** —
metryka, którą raportuje literatura, w odróżnieniu od naszego surowszego `knowledge`.
