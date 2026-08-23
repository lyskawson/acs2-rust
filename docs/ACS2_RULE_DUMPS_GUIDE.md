# Jak czytać zrzut populacji reguł ACS2 — notatka

Ta notatka to praktyczny słownik do dwóch konkretnych zrzutów wyuczonej populacji
(`mpx_a2_rules.txt` i `maze4_rules.txt`), wygenerowanych demonstracyjnie przez
dodatkowe binaria `dump_rules.rs` / `dump_rules_maze.rs` (nie wchodzą do repo —
posłużyły tylko do zobaczenia, jak wygląda populacja "od środka"). Kontekst
teoretyczny jest w `docs/ACS2_PRIMER.md` — tu jest skrót zorientowany na
**odczytanie konkretnego wiersza**.

## 1. Notacja pojedynczej reguły

Każdy wiersz zrzutu to jeden **makroklasyfikator** — jedna reguła w populacji.
Ogólny wzorzec czytania:

```text
JEŻELI condition pasuje do bieżącej percepcji
I agent wykona action
TO effect przewiduje, jak zmieni się percepcja
```

### `condition` i `effect` — dwa różne znaczenia tego samego `#`

To jest najczęstsza pułapka przy czytaniu zrzutu — wildcard `#` znaczy co innego
po lewej, a co innego po prawej stronie reguły:

| Symbol | W `condition` | W `effect` |
|---|---|---|
| `#` | "dowolna wartość na tej pozycji pasuje" | "ta pozycja **nie zmieni się** po akcji" |
| cyfra (`0`,`1`,`9`,...) | "musi być dokładnie ta wartość" | "ta pozycja **zmieni się** na tę wartość" |

Czyli `effect = #######` (sam wildcard) to poprawny, pełny model — "nic się
nie zmieni" — a nie "nie wiem, co się stanie".

### Notacja pozycji — zależna od środowiska

Percepcja to `N`-symbolowa tablica, a co oznacza każda pozycja, definiuje
środowisko:

**Multiplekser `a=2` (`N=7`):**

```text
pozycja:  0     1     2     3     4     5     6
znaczenie: addr0 addr1 data0 data1 data2 data3 valid
```

`addr0/addr1` to 2 bity adresu, `data0..data3` to 4 bity danych (bo `2^a = 4`),
`valid` to bit walidacyjny — startuje jako `0`, poprawna odpowiedź agenta
zmienia go na `1` i kończy epizod nagrodą `1000`.

**Labirynt `Maze4-v0` (`N=8`):**

```text
pozycja:  0  1   2  3   4  5   6  7
znaczenie: N  NE  E  SE  S  SW  W  NW
```

To 8 sąsiadów agenta (nie jego współrzędne!). Wartości: `0` = ścieżka,
`1` = ściana, `9` = cel. Akcja `0..7` odpowiada temu samemu kierunkowi — akcja
`2` to ruch na `E` itd.

## 2. Słownik kolumn liczbowych

| Kolumna | Nazwa pełna | Co mierzy | Wzór / zachowanie | Skąd w kodzie |
|---|---|---|---|---|
| `q` | jakość antycypacji | Jak wiarygodny jest **model przejścia** tej reguły (condition→effect). **Nie** jest nagrodą. | Start `initial_q = 0.5`. Trafna antycypacja: `q += beta*(1-q)`. Błędna: `q -= beta*q`. `reliable` = `q > theta_r` (domyślnie `0.9`). | `classifier.rs:84-90` |
| `r` | predykcja zdyskontowanej wypłaty | Ile nagrody (zdyskontowanej) warto oczekiwać po wykonaniu tej reguły. Odpowiednik Q-value z klasycznego RL. | `r += beta*(reward + gamma*bootstrap - r)` | `rl.rs:16-25` |
| `num` | numerosity | Ile mikroklasyfikatorów reprezentuje ten jeden wiersz (makroklasyfikator). | Rośnie tylko przez GA (scalanie identycznych reguł) — przy `do_ga=false` zawsze `num=1`. | `population.rs:29-30` |
| `exp` | doświadczenie | Ile razy ALP w ogóle przetworzył tę regułę — niezależnie, czy trafnie, czy błędnie. | `exp += 1` przy każdym przejściu przez jej action set. Próg dojrzałości do subsumpcji to `theta_exp=20`. | `alp.rs:193-200` |
| `fit` | fitness | Łączna ocena "czy model jest wiarygodny **i** czy warto to robić". Nie jest osobnym polem — to iloczyn. | `fitness = q * r` | `classifier.rs:80-82` |

**Ważne zastrzeżenie:** wysoki `fit` nie oznacza automatycznie, że reguła
jest brana pod uwagę przy wyborze akcji. Selektor `BestAction`/`EpsilonGreedy`
dodatkowo filtruje po `does_anticipate_change()` — reguła z `effect` w całości
wildcardowym (przewiduje "brak zmiany") jest **zawsze pomijana** przy wyborze
najlepszej akcji, nawet jeśli jest `reliable` i ma sensowne `q`. Widać to w obu
zrzutach jako osobną, liczną grupę reguł z `fit ≈ 0`.

## 3. Przykład 1 — Multiplekser `a=2`, `N=7`

Config: `Configuration::mpx()` (`theta_r=0.9`, `do_ga=false`, `u_max=100000`,
`alp_gen_variant=Pyalcs`), `epsilon=0.8`, 20 000 prób eksploracyjnych, seed `42`.

**Wynik populacji:** 280 makroklasyfikatorów, wszystkie 280 `reliable`,
`knowledge = 1.0000` (populacja poprawnie modeluje 100% możliwych przejść).

### Odczytanie jednego wiersza krok po kroku

```text
condition  act  effect     q       r        num  exp   fit
01#0##0    0    ######1   1.000   1000.00    1   1439  1000.00
```

1. `condition = 01#0##0` → reguła pasuje tylko, gdy `addr0=0, addr1=1,
   data1=0`; pozostałe 4 pozycje (w tym `data0`, `data2`, `data3` i `valid`)
   są dowolne.
2. `act = 0` → reguła "głosuje" na akcję (odpowiedź) `0`.
3. `effect = ######1` → jedyna przewidywana zmiana to ostatnia pozycja
   (`valid`) → `0→1`, wszystko inne bez zmian.
4. `q = 1.000, r = 1000.00` → model jest w pełni wiarygodny i historycznie
   zawsze prowadził do pełnej nagrody. `fit = 1000.00`.
5. `exp = 1439` → ta konkretna kombinacja (adres `01`, jeden konkretny bit
   danych) trafiła się 1439 razy w 20 000 próbach.

Sens: "gdy adres wskazuje na `data1` i `data1=0`, poprawną odpowiedzią jest
`0`" — czyli reguła nauczyła się fragmentu funkcji multipleksera.

### Dwie grupy reguł w populacji

| Grupa | Liczba reguł | `effect` | `r` | Sens |
|---|---:|---|---|---|
| **predict-change** | 147 | `######1` | rośnie do `≈1000` | Reguła trafiła poprawną odpowiedź — przewiduje ustawienie `valid→1`. |
| **predict-no-change** | 133 | `#######` | maleje do `≈0` | Reguła "wie", że dla tej kombinacji jej akcja jest błędna — percepcja się nie zmieni, `valid` zostaje `0`. |

Grupa druga jest `reliable` (poprawnie przewiduje "nic się nie stanie"), ale
ma `fit ≈ 0` i jest wykluczona z wyboru akcji przez `does_anticipate_change()`
— więc nie wpływa na to, którą akcję agent faktycznie wybierze.

### Nieoczywisty wniosek: specyficzność jest wyższa niż "idealna"

Primer (`docs/ACS2_PRIMER.md`, sekcja 11) podaje, że idealna reguła MPX ma
specyficzność `a+1 = 3` (2 bity adresu + 1 bit danych, reszta wildcard). W tym
zrzucie żadna reguła "predict-change" nie ma specyficzności 3:

| Specyficzność condition | Liczba reguł (ze 147 "predict-change") |
|---:|---:|
| 4 | 6 |
| 5 | 31 |
| 6 | 46 |
| 7 (pełna) | 64 |

To bezpośrednia ilustracja tego, że `do_ga=false` + `u_max=100000` = tryb
**specialize-only** (primer, sekcja 14.7): ALP tylko dodaje szczegóły, a bez
GA nic nie generalizuje z powrotem zbędnych, nadmiarowo ustalonych bitów.
Populacja mimo to osiąga `knowledge = 1.0`, ale robi to za pomocą wielu
nadmiernie szczegółowych reguł zamiast garstki maksymalnie ogólnych.

## 4. Przykład 2 — Labirynt `Maze4-v0`, `N=8`

Config: `Configuration::default_protocol()` (`theta_r=0.9`, `do_ga=false`,
`u_max=100000`), `epsilon=0.8`, 3000 prób eksploracyjnych, seed `42`.

**Wynik populacji:** 291 makroklasyfikatorów, wszystkie `reliable`. Po
treningu, w trybie eksploatacyjnym (greedy), agent dochodzi do celu średnio w
**3,86 kroku** (50 prób testowych) — blisko optymalnej ścieżki w tym
niewielkim labiryncie.

### Odczytanie jednego wiersza krok po kroku

```text
condition  act  dir  effect     q       r       num  exp   fit
090###10   1    NE   111###01  1.000   1000.00   1   1255  1000.00
```

1. `condition = 090###10` → pozycje `N=0` (ścieżka), `NE=9` (**cel widoczny
   po skosie!**), `SE=0`, `W=1` (ściana), `NW=0`; pozostałe dowolne.
2. `act = 1` (`dir = NE`) → reguła każe iść po skosie w kierunku, w którym
   widać cel.
3. `effect = 111###01` → aż 5 z 8 sensorów zmieni się po tym ruchu
   (`N→1, NE→1, E→1, W→0, NW→1`) — bo agent fizycznie przechodzi do innej
   komórki i cała jego 8-sensorowa okolica jest inna.
4. `q=1.000, r=1000.00, fit=1000.00` → to jest krok bezpośrednio kończący
   epizod sukcesem (wejście na cel).

### Kluczowa różnica względem MPX: co zmienia `effect`

| | Multiplekser | Labirynt |
|---|---|---|
| Ile pozycji `effect` zmienia typowa "dobra" reguła | zwykle **1** (bit `valid`) | zwykle **kilka do wszystkich 8** |
| Dlaczego | Akcja to "odpowiedź", zmienia tylko flagę poprawności. | Ruch przenosi agenta do innej komórki — cała lokalna okolica (8 sąsiadów) jest inna. |

### Ten sam wzorzec "no-change", inny kontekst

Podobnie jak w MPX, w populacji labiryntu jest wyraźna druga grupa:

| Grupa | Liczba reguł (przybliżona) | `effect` | Sens |
|---|---:|---|---|
| ruch skuteczny | ok. 230 | zmienia część/wszystkie pozycje | Reguła prowadzi do realnego przesunięcia agenta. |
| odbicie od ściany | ok. 63 | `########` | Reguła poprawnie przewiduje "ta akcja tu nic nie zmieni" (ściana). |

To dokładnie ten sam mechanizm co w MPX: grupa "odbicie od ściany" jest
`reliable`, ale ma niskie `r` → niski `fit` → jest pomijana przez
`does_anticipate_change()` przy wyborze najlepszej akcji. Agent "unika" ścian
nie przez twardą blokadę, tylko przez to, że nauczony model tej akcji
wypada z puli kandydatów.

`r` w labiryncie dodatkowo dobrze pokazuje propagację nagrody wstecz przez
`gamma=0.95`: reguła bezpośrednio prowadząca do celu ma `r≈1000`, a reguły
kilka kroków wcześniej mają coraz niższe `r` (w pełnym zrzucie widać spadek
`950.00 → 949.xx → ... → 753.xx` w miarę oddalania się od celu) — to
bootstrap `q*r` propagowany krok po kroku wstecz przez kolejne action sety.

## 5. Szybka ściąga na przyszłość

Kiedy wracasz do zrzutu reguł i nie pamiętasz co czytasz:

1. Sprawdź **układ pozycji percepcji** dla danego środowiska (sekcja 1) — bez
   tego `condition`/`effect` to bezsensowny ciąg znaków.
2. `#` w `condition` = "obojętne"; `#` w `effect` = "bez zmian" — nigdy
   odwrotnie.
3. `q` = wiarygodność modelu, `r` = wartość nagrody, `fit = q*r` — trzy różne
   pytania, trzy różne liczby.
4. Reguły z `effect` w całości wildcardowym są realną, wyuczoną wiedzą
   ("tu nic się nie stanie"), ale nie liczą się przy wyborze najlepszej akcji.
5. Przy `do_ga=false` zawsze `num=1` — to nie błąd, to brak mechanizmu, który
   mógłby zwiększyć numerosity.
6. Wysoka specyficzność `condition` przy `do_ga=false` jest oczekiwana
   (specialize-only) — nie oznacza to błędu w porcie, tylko brak presji
   generalizującej.
