# ACS2 od podstaw — primer oparty na tym repozytorium

Ten dokument opisuje **to, co wykonuje aktualny port Rust**, którego wzorcem jest
pyalcs. Nie jest to swobodne streszczenie literatury o Learning Classifier Systems
(LCS). Jeżeli kod pyalcs różni się od algorytmicznego opisu Butza, punktem odniesienia
dla wariantu `Pyalcs` jest pyalcs; repo zawiera też jawnie oddzielony wariant `Butz`
generalizacji ALP. Wybór wariantu widać w `AlpGenVariant`
(`acs2-core/src/config.rs:1-5`) i w rozgałęzieniu `expected_case`
(`acs2-core/src/alp.rs:77-84`).

Odwołania typu `acs2-core/src/alp.rs:193-217` wskazują linie kodu, które wykonują
opisywany mechanizm. Odwołania do plików `.md` oznaczają kontekst naukowy albo
udokumentowaną decyzję projektu, a nie wykonywalną semantykę.

## 1. Najpierw podstawy uczenia ze wzmocnieniem

### Agent, środowisko i stan

**Agent** jest programem, który wielokrotnie obserwuje środowisko, wybiera akcję i
otrzymuje skutek tej akcji. Interfejs środowiska ma tutaj dwie operacje: `reset()`
zwraca początkową percepcję, a `step(action)` zwraca nową percepcję, nagrodę oraz
informację o zakończeniu albo ucięciu epizodu
(`acs2-core/src/environment.rs:5-15`). Stan dostępny agentowi jest w praktyce
**percepcją**: tablicą `N` symboli (`acs2-core/src/perception.rs:3-15`). Nie należy
zakładać, że jest to pełny fizyczny stan świata; w labiryncie agent widzi tylko osiem
sąsiednich pól, a nie swoje współrzędne (`acs2-envs/src/maze.rs:75-84`).

**Akcja** jest liczbą całkowitą. Konfiguracja podaje liczbę możliwych akcji
(`acs2-core/src/config.rs:8-10`): domyślnie osiem, a dla multipleksera dwie
(`acs2-core/src/config.rs:32-35`, `acs2-core/src/config.rs:57-61`).

**Funkcja nagrody** należy do środowiska. W labiryncie wejście na pole celu daje
`1000`, a każdy inny krok `0` (`acs2-envs/src/maze.rs:115-130`). W multiplekserze
poprawna odpowiedź daje `1000`, błędna `0` (`acs2-envs/src/multiplexer.rs:78-92`).
Nagroda mówi więc, co jest pożądane; nie mówi jeszcze agentowi, jaką akcję ma wybrać.

**Epizod**, w kodzie najczęściej nazywany próbą (`trial`), zaczyna się od `reset()` i
kończy, gdy środowisko zwróci `terminated` lub `truncated`
(`acs2-core/src/agent.rs:69-74`, `acs2-core/src/agent.rs:117-154`). `terminated`
oznacza naturalny koniec, np. osiągnięcie celu. `truncated` oznacza administracyjne
ucięcie po limicie kroków. Dla obu przypadków obecny agent wykonuje końcową aktualizację
z bootstrapem równym zero (`acs2-core/src/agent.rs:122-153`).

### Polityka, eksploracja i eksploatacja

**Polityka** to reguła wyboru akcji na podstawie bieżącej informacji. Ten port nie
przechowuje osobnej tablicy prawdopodobieństw polityki. Polityką jest implementacja
`ActionSelector`, która otrzymuje populację i zbiór dopasowanych klasyfikatorów
(`acs2-core/src/action_selection.rs:4-11`).

Podczas uczenia używane jest zwykle **epsilon-greedy**:

- z prawdopodobieństwem `epsilon` wybiera jednolicie losową akcję — to
  **eksploracja**;
- w przeciwnym przypadku wybiera najlepszą znaną akcję — to **eksploatacja**.

Dokładne rozgałęzienie jest w `acs2-core/src/action_selection.rs:57-74`. „Najlepsza”
oznacza tu akcję klasyfikatora przewidującego zmianę i maksymalizującego
`fitness * num`; jeżeli takiego klasyfikatora nie ma, wybór wraca do akcji losowej
(`acs2-core/src/action_selection.rs:26-54`). To ważne: nawet przy `epsilon = 0`
zachowanie nie musi być deterministyczne, gdy brak reguły przewidującej zmianę albo
gdy remis jest rozstrzygany po przetasowaniu kandydatów.

Osobna metoda eksploatacyjna `run_exploit_trial` zawsze tworzy `BestAction`, więc nie
czyta `epsilon` (`acs2-core/src/agent.rs:171-188`,
`acs2-core/src/agent.rs:195-213`). Eksploatacja nadal aktualizuje `r` i `ir`, ale nie
uruchamia ALP, coveringu ani GA (`acs2-core/src/agent.rs:198-230`). W tym projekcie
„eksploatacja” nie oznacza zatem całkowicie zamrożonych liczb w regułach; zamrożona
jest struktura populacji i `q`, ale predykcje nagrody nadal się uczą.

### Co agent właściwie próbuje oszacować

W zwykłym uczeniu ze wzmocnieniem często wystarcza nauczyć się wartości stanu albo
akcji. ACS2 rozdziela dwa pytania:

1. **Co stanie się ze stanem po akcji?** Odpowiadają `condition`, `action`, `effect`
   i jakość antycypacji `q`.
2. **Ile nagrody warto oczekiwać?** Odpowiadają predykcje `r` i `ir`.

Domyślna miara używana przy wyborze i bootstrapie łączy te warstwy jako
`fitness = q * r` (`acs2-core/src/classifier.rs:80-82`). Dzięki temu wysoka
przewidywana nagroda nie wystarcza, jeśli reguła jest słabym modelem przejścia.

## 2. LCS, ACS2 i znaczenie „antycypacyjny”

### Robocza definicja LCS w tym repo

Learning Classifier System jest tutaj populacją reguł. Dla bieżącej percepcji
tworzony jest **match set** `[M]`: indeksy wszystkich reguł, których warunek pasuje
do percepcji (`acs2-core/src/population.rs:59-63`). Po wyborze akcji powstaje
**action set** `[A]`: podzbiór `[M]` o tej właśnie akcji
(`acs2-core/src/population.rs:65-71`). Uczenie działa lokalnie przede wszystkim na
poprzednim `[A]`, a nie od razu na całej populacji
(`acs2-core/src/agent.rs:77-110`).

Pojedynczy zapis w `Population` jest **makroklasyfikatorem**. Pole `num` mówi, ile
identycznych mikroklasyfikatorów reprezentuje. Liczba struktur to `len`, a łączna
liczebność mikroklasyfikatorów to suma `num`
(`acs2-core/src/population.rs:21-30`). To rozróżnienie ma znaczenie w GA i przy
interpretacji rozmiaru populacji.

### Co odróżnia ACS2

Reguła ACS2 ma jawny `effect`, czyli model przejścia. Dla każdego atrybutu efekt mówi:

- wildcard oznacza „ten atrybut się nie zmieni”;
- konkretny symbol oznacza „atrybut zmieni się na ten symbol”.

Kod sprawdza pierwszy przypadek przez `before == after`, a drugi przez jednoczesne
`before != after` i `effect == after` (`acs2-core/src/effect.rs:9-15`). Cały efekt
jest poprawny tylko wtedy, gdy każdy atrybut spełnia swoją część przewidywania
(`acs2-core/src/effect.rs:24-26`). To właśnie jest **antycypacja**: reguła nie tylko
mówi „akcja 1 jest wartościowa”, ale przewiduje relację `p0 --akcja--> p1`.

Konsekwencje są praktyczne:

- agent może osobno oceniać poprawność modelu świata (`q`) i nagrodę (`r`);
- błędne przewidywanie uruchamia mechanizm specjalizacji oparty na `Mark`, a nie tylko
  korektę liczby wartości;
- metryka `knowledge` może pytać, czy populacja zna konkretne przejścia, również te,
  w których nic się nie zmieniło (`acs2-core/src/knowledge.rs:17-24`).

### A co z XCS?

W repozytorium **nie ma implementacji XCS**, więc szczegółowego opisu XCS nie da się
uczciwie potwierdzić liniami `.rs`. Najbezpieczniejsza różnica na potrzeby tej pracy
jest taka: wykonywalna reguła ACS2 ma jawny `effect` i uczy go przez ALP; pokazują to
`Classifier` (`acs2-core/src/classifier.rs:9-23`) oraz `apply_alp`
(`acs2-core/src/alp.rs:177-223`). Porównanie literaturowe projektu opisuje XCS jako
system oparty na dokładności predykcji wypłaty i nacisku generalizacyjnym GA, a ACS2
jako system z deterministycznym operatorem specjalizacji ALP
(`reports/MPX_literature_review.md:59-69`). Tę część, zwłaszcza zdania o XCS, należy
w pracy cytować z literatury pierwotnej wymienionej w raporcie, nie z kodu Rust.

## 3. Anatomia klasyfikatora

Struktura ma wszystkie omawiane pola w jednym miejscu
(`acs2-core/src/classifier.rs:8-23`). Jej część symboliczną można czytać jako:

```text
JEŻELI condition pasuje do p0
I wykonano action
TO effect przewiduje zmianę p0 -> p1
```

### `condition`

Warunek jest tablicą `N` symboli (`acs2-core/src/condition.rs:5-8`). Symbol konkretny
musi zgadzać się z percepcją, a wildcard pasuje do dowolnej wartości
(`acs2-core/src/condition.rs:10-12`, `acs2-core/src/condition.rs:21-23`). Warunek
decyduje więc, w jakich sytuacjach reguła wchodzi do `[M]`.

### `action`

`action: Option<usize>` identyfikuje akcję przewidywaną przez regułę
(`acs2-core/src/classifier.rs:10-12`). Zwykła reguła działająca w populacji ma
`Some(action)`; `None` jest dopuszczone przez konstruktor ogólny, ale taka reguła nie
wejdzie do żadnego action setu tworzonego dla konkretnej akcji
(`acs2-core/src/population.rs:65-70`).

### `effect`

Efekt ma również `N` pozycji (`acs2-core/src/effect.rs:4-7`), ale wildcard znaczy w
nim coś innego niż w warunku. W `condition` `#` znaczy „dowolna wartość”; w `effect`
znaczy „wartość przechodzi bez zmiany”. Konkretny symbol efektu oznacza zmianę na tę
wartość (`acs2-core/src/effect.rs:9-15`). Nie wolno interpretować wildcardu efektu
jako „nie wiem”.

### `mark`

`Mark` przechowuje po jednym zbiorze symboli na atrybut
(`acs2-core/src/mark.rs:8-17`). Jest pamięcią kontekstów, w których reguła przewidziała
źle, wykorzystywaną później do wskazania rozróżniającego atrybutu. Szczegółowy opis i
przykład są w sekcji 8.

### Pola liczbowe i czasowe

| Pole | Znaczenie w tej implementacji | Gdzie jest używane |
|---|---|---|
| `q` | Jakość antycypacji, czyli pamięć o tym, jak wiarygodny jest model przejścia reguły. Nie jest nagrodą. | Wzrost/spadek: `acs2-core/src/classifier.rs:84-90`; progi: `acs2-core/src/classifier.rs:107-113`; fitness: `acs2-core/src/classifier.rs:80-82`. |
| `r` | Predykcja zdyskontowanej wypłaty: bieżąca nagroda plus bootstrap przyszłego fitness. | Aktualizacja: `acs2-core/src/rl.rs:16-25`; składnik fitness: `acs2-core/src/classifier.rs:80-82`. |
| `ir` | Predykcja wyłącznie natychmiastowej nagrody. | Aktualizacja: `acs2-core/src/rl.rs:23-25`. W obecnym core nie wpływa na wybór akcji ani fitness. |
| `num` | Numerosity: liczba mikroklasyfikatorów reprezentowanych przez jedną strukturę. | Suma populacji: `acs2-core/src/population.rs:29-30`; wybór akcji: `acs2-core/src/action_selection.rs:44-50`; GA: `acs2-core/src/ga.rs:61-70`. |
| `exp` | Licznik doświadczenia klasyfikatora; rośnie przy każdym przetworzeniu go przez ALP. | Wzrost: `acs2-core/src/alp.rs:193-200`; próg subsumpcji: `acs2-core/src/subsumption.rs:3-5`. |
| `talp` | Czas ostatniej aplikacji ALP do tej reguły; `None` oznacza brak wcześniejszego zastosowania. | Aktualizacja średniego odstępu: `acs2-core/src/classifier.rs:96-105`. |
| `tga` | Czas ostatniego zastosowania GA dla niszy/reguły. | Warunek uruchomienia i reset: `acs2-core/src/ga.rs:8-35`. |
| `tav` | Adaptacyjna średnia odstępu między zastosowaniami reguły. Dla małego `exp` używa średniej przyrostowej, później kroku `beta`. | `acs2-core/src/classifier.rs:96-104`; pomocniczy tie-break usuwania w GA: `acs2-core/src/ga.rs:107-125`. |
| `ee` | Flaga „enhanceable” związana z Probability-Enhanced Effects. | Jest inicjalizowana na `false` i zerowana przy zmianie marku (`acs2-core/src/classifier.rs:26-40`, `acs2-core/src/classifier.rs:131-134`). W tym porcie nie ma kodu ustawiającego ją na `true`, więc funkcjonalnie pozostaje nieaktywna. |

Konstruktor ogólny rozpoczyna od całkowicie ogólnego warunku i efektu, pustego marku,
`num = 1`, `exp = 1`, pustego `talp` i wartości początkowych z konfiguracji
(`acs2-core/src/classifier.rs:26-41`). Covering świadomie różni się wartościami
`r = 0`, `exp = 0`, `talp = time`, `tga = time`
(`acs2-core/src/classifier.rs:44-59`). Potomek kopiuje część wiedzy rodzica, ale dostaje
pusty mark, `num = 1` i `exp = 1` (`acs2-core/src/classifier.rs:62-77`).

Tożsamość klasyfikatora obejmuje tylko `(condition, action, effect)`, a nie `q`, `r`,
mark ani liczniki (`acs2-core/src/classifier.rs:174-180`). Dwie strukturalnie identyczne
reguły są więc traktowane jako ten sam typ reguły mimo odmiennej historii uczenia.

## 4. Wildcard `#` i specyficzność

W kodzie wildcard nie jest przechowywany jako znak ASCII `#`, tylko jako wariant
`Symbol::Wildcard`; drugi wariant to `Token(u8)`
(`acs2-core/src/symbol.rs:1-16`). Znak `#` pozostaje właściwą notacją przy zapisie
reguł w tekście.

**Specyficzność** (`specificity()`) warunku jest liczbą pozycji, które nie są wildcardami
(`acs2-core/src/condition.rs:29-31`). Dla czterech binarnych atrybutów:

| Warunek | Specyficzność | Liczba pasujących wejść |
|---|---:|---:|
| `####` | 0 | 16 |
| `1#0#` | 2 | 4 |
| `1100` | 4 | 1 |

Pierwsza reguła ma duży zasięg, ale może połączyć sytuacje o różnych skutkach akcji.
Ostatnia łatwo opisze jedno zaobserwowane przejście, lecz prawie nie ma okazji do
ponownego użycia. Kod odzwierciedla to bezpośrednio: match set zawiera regułę tylko,
gdy jej condition pasuje (`acs2-core/src/population.rs:59-63`), a ścisła relacja
„bardziej ogólna” to po prostu mniejsza specyficzność
(`acs2-core/src/classifier.rs:115-117`).

Napięcie **specjalizacja ↔ generalizacja** jest osią eksperymentów MPX:

- specjalizacja dodaje konkretny symbol do warunku; robi to ALP przez
  `specialize_with` oraz `Classifier::specialize`
  (`acs2-core/src/condition.rs:33-39`, `acs2-core/src/classifier.rs:159-170`);
- generalizacja usuwa konkretny symbol, zastępując go wildcardem
  (`acs2-core/src/condition.rs:41-52`); używają tego generalizacja ALP i mutacja GA.

Nie istnieje jedna „najlepsza” specyficzność niezależna od zadania. Dobra reguła ma
ustalone wszystkie cechy potrzebne do jednoznacznego przewidzenia skutku, ale żadnych
zbędnych. W multiplekserze struktura problemu pozwala podać ideał `a+1`; w labiryncie
kod nie zawiera analogicznej zamkniętej formuły.

## 5. `reliable`, `inadequate` i dynamika `q`

Reguła jest:

- **reliable**, gdy `q > theta_r`;
- **inadequate**, gdy `q < theta_i`.

Oba porównania są **ostre**, więc `q == theta_r` nie jest jeszcze reliable, a
`q == theta_i` nie jest inadequate (`acs2-core/src/classifier.rs:107-113`).

Po poprawnej antycypacji bez potrzeby specjalizacji:

```text
q <- q + beta * (1 - q)
```

Po niepoprawnej antycypacji:

```text
q <- q - beta * q = (1 - beta) * q
```

Formuły są dosłownie w `acs2-core/src/classifier.rs:84-90`; miejsca wywołania to
`expected_case` bez różnicy marku (`acs2-core/src/alp.rs:68-72`) i każdy
`unexpected_case` (`acs2-core/src/alp.rs:155-163`). Przy domyślnych `q = 0.5` i
`beta = 0.05` jeden prosty sukces daje `0.525`, a jedna porażka `0.475`. Przy samych
sukcesach bez specjalizacji potrzeba 32 takich aktualizacji, aby przejść z `0.5`
powyżej `theta_r = 0.9`.

Ważny szczegół: poprawna antycypacja **nie zawsze od razu zwiększa `q` rodzica**.
Jeżeli mark zwraca niepustą różnicę, `expected_case` tworzy wyspecjalizowanego potomka
i omija `increase_quality` (`acs2-core/src/alp.rs:68-90`). Zdanie „q rośnie przy
trafionej antycypacji” wymaga więc dopowiedzenia: rośnie w gałęzi trafnej antycypacji,
w której mark nie żąda nowej specjalizacji, albo gdy identyczny/subsumujący
klasyfikator przejmie potomka i dostanie wzrost jakości
(`acs2-core/src/alp.rs:41-54`).

Reguła, która przewidziała źle i po obniżeniu `q` jest inadequate, trafia na listę
ofiar ALP (`acs2-core/src/alp.rs:208-212`). Usunięcie jest odroczone do końca
przetwarzania całego action setu (`acs2-core/src/alp.rs:234-240`).

Liczba reguł reliable jest podstawową, łatwą miarą postępu, ponieważ odlicza reguły,
których historia poprawnych antycypacji przebiła wysoki próg. Implementacja liczy je
bezpośrednio w `Population::reliable_count`
(`acs2-core/src/population.rs:33-38`). Nie jest jednak miarą wystarczającą: wiele
reliable reguł może opisywać ten sam mały fragment przestrzeni. Dlatego `knowledge`
sprawdza pokrycie prawdziwych przejść, a eksperymenty MPX dodatkowo raportują średnią
specyficzność reliable reguł.

## 6. Trzy mechanizmy uczenia i ich podział ról

### 6.1 ALP — uczenie modelu przejścia

**Po co istnieje.** ALP odpowiada na pytanie: „czy reguły wybranej akcji poprawnie
przewidziały obserwowaną zmianę i jak poprawić ich model?”. Zmienia `q`, `condition`,
`effect`, `mark`, doświadczenie i populację. Nie uczy bezpośrednio predykcji nagrody.

**Kiedy działa.** W próbie eksploracyjnej ALP jest wykonywany na action secie
**poprzedniej** akcji, gdy dostępne jest już `p1`
(`acs2-core/src/agent.rs:74-89`). Na końcu epizodu działa natychmiast na ostatnim
action secie (`acs2-core/src/agent.rs:122-134`). W próbie eksploatacyjnej nie działa.

**Wspólny początek.** Każda reguła poprzedniego `[A]` dostaje `exp += 1`, aktualizację
`tav`, a potem test pełnej poprawności efektu
(`acs2-core/src/alp.rs:193-205`).

#### Expected case

To przypadek, w którym efekt reguły dokładnie zgadza się z przejściem `p0 -> p1`.
ALP pyta jeszcze mark, czy bieżący kontekst różni się od zapamiętanych kontekstów
porażki (`acs2-core/src/alp.rs:61-70`).

- Jeżeli różnica jest pusta, reguła pozostaje strukturalnie bez zmian, a `q` rośnie
  (`acs2-core/src/alp.rs:70-72`).
- Jeżeli różnica jest niepusta, powstaje potomek. Jego condition jest specjalizowany
  symbolami różnicy, a `q` potomka ma dolne ograniczenie `0.5`
  (`acs2-core/src/alp.rs:75-90`).
- Przed specjalizacją może zadziałać ograniczenie nadmiernej specyficzności zależne
  od `u_max` i `alp_gen_variant` (`acs2-core/src/alp.rs:77-84`). Wariant Pyalcs
  generalizuje rodzica i/lub redukuje różnicę
  (`acs2-core/src/alp.rs:93-123`); wariant Butz generalizuje potomka i/lub różnicę
  (`acs2-core/src/alp.rs:125-153`).

Mały przykład: reguła `condition = #1#`, `effect = ##1` pasuje do `p0 = 010` i
poprawnie przewiduje `p1 = 011`. Jeżeli mark jest pusty, to prosty expected case i
rośnie `q`. Jeżeli mark wskazuje, że pierwsza pozycja odróżnia wcześniejsze porażki,
różnica może być `0##`; wtedy powstaje potomek `condition = 01#`. Efekt pozostaje
ten sam. Dokładny wybór różnicy jest opisany w sekcji o marku.

#### Unexpected case

To przypadek, w którym choć jedna pozycja efektu nie zgadza się z rzeczywistym
przejściem. Najpierw spada `q` rodzica i aktualizowany jest jego mark
(`acs2-core/src/alp.rs:155-164`). Jeśli istniejący efekt da się zgodnie
wyspecjalizować, powstaje potomek; dla każdej faktycznie zmienionej pozycji potomek
ustawia efekt na symbol z `p1`, a condition na symbol z `p0`
(`acs2-core/src/alp.rs:165-174`, `acs2-core/src/classifier.rs:159-170`). Jeśli efekt
jest sprzeczny z obserwacją i nie jest specjalizowalny, potomka nie ma
(`acs2-core/src/effect.rs:28-34`).

Przykład: przy `p0 = 010`, `p1 = 011` all-wildcard effect błędnie przewiduje brak
zmiany na pozycji 3. Potomek dostaje na tej pozycji `condition = 0` oraz `effect = 1`.
Pozycje 1 i 2 pozostają wildcardami, bo się nie zmieniły. To nie jest kopiowanie
całego `p0`, tylko zapis minimalnej zaobserwowanej zmiany.

Po obsłużeniu wszystkich reguł, jeżeli **ani jedna** nie weszła w expected case,
ALP uruchamia covering (`acs2-core/src/alp.rs:220-223`). Potomkowie są następnie
dodawani do populacji i odpowiednich zbiorów, a ofiary inadequate usuwane
(`acs2-core/src/alp.rs:225-240`).

### 6.2 RL — uczenie wartości nagrody

**Po co istnieje.** Poprawne przewidywanie świata nie mówi jeszcze, którą zmianę
warto wywołać. RL uczy liczby `r` i `ir`, dzięki którym wybór akcji może preferować
przewidywane przejścia prowadzące do nagrody.

**Jak działa.** Najpierw wyznaczany jest cel:

```text
P  = reward + gamma * bootstrap
r  = r  + beta * (P - r)
ir = ir + beta * (reward - ir)
```

Kod jest w `acs2-core/src/rl.rs:16-26`, a aktualizacja obejmuje każdą regułę
poprzedniego action setu (`acs2-core/src/rl.rs:28-38`). `beta` określa szybkość
zbliżania się do nowego celu. `gamma` określa wagę przewidywanej przyszłości.

Domyślny bootstrap to maksimum `q*r` w następnym match secie, ale tylko wśród reguł,
które przewidują jakąś zmianę (`acs2-core/src/population.rs:73-80`,
`acs2-core/src/rl.rs:8-13`). Na końcu epizodu bootstrap jest równy `0`
(`acs2-core/src/agent.rs:135-142`). To nie jest klasyczne maksimum samego `r`:
niepewność modelu `q` obniża wartość używaną do propagowania nagrody.

RL nie zmienia `q`, condition, effect ani marku. W szczególności sama eksploatacja
może dostroić `r` i `ir`, ale nie może utworzyć ani naprawić modelu przejścia.

### 6.3 GA — genetyczna presja generalizacyjna

**Po co istnieje.** ALP ma silną skłonność do dodawania szczegółów. GA dostarcza
przeciwną presję: produkuje potomków o warunkach z większą liczbą wildcardów, aby
poprawne reguły obejmowały więcej sytuacji.

**Kiedy działa.** Tylko gdy `do_ga` jest prawdziwe
(`acs2-core/src/agent.rs:99-110`, `acs2-core/src/agent.rs:143-153`). Dodatkowo dla
danego action setu musi zachodzić:

```text
time - weighted_mean(tga, weights=num) > theta_ga
```

Warunek i reset `tga` są w `acs2-core/src/ga.rs:8-37`.

**Kroki GA:**

1. Dwa losowania rodziców ruletką z wagą `q^3 * num`; ten wybór nie używa `r`
   (`acs2-core/src/ga.rs:56-71`).
2. Powstają kopie potomne (`acs2-core/src/ga.rs:251-253`).
3. **Mutacja generalizująca** niezależnie zamienia każdy ustalony symbol condition na
   wildcard z prawdopodobieństwem `mu`. Nie istnieje tu mutacja odwrotna
   `# -> symbol`, a effect nie jest mutowany (`acs2-core/src/ga.rs:74-84`).
4. Z prawdopodobieństwem `chi`, ale tylko przy identycznych efektach, krzyżowanie
   zamienia odcinek condition między potomkami. Następnie uśrednia `q` i `r`
   (`acs2-core/src/ga.rs:258-266`; sam operator odcinka:
   `acs2-core/src/ga.rs:86-105`).
5. `q` obu potomków jest dzielone przez dwa; całkiem ogólne warunki o
   specyficzności zero są odrzucane (`acs2-core/src/ga.rs:268-279`).
6. GA egzekwuje `theta_as` przez usuwanie w bieżącym action secie, a potem scala lub
   wstawia potomków (`acs2-core/src/ga.rs:281-285`).

Precyzyjne zastrzeżenie: **mutacja** jest ściśle generalizująca. Krzyżowanie samo nie
usuwa ustalonych atrybutów globalnie; tylko przenosi fragment warunku między dwojgiem
potomków i może jednego z nich uczynić bardziej szczegółowym, a drugiego bardziej
ogólnym. Nazwa „genetic generalization” opisuje więc kierunek mutacji i łączną presję
procesu, nie monotoniczną zmianę każdego potomka.

GA nie jest zwykłym algorytmem genetycznym przeszukującym dowolne ciągi bitów.
Operuje wewnątrz action setu, nie mutuje efektu ani akcji, a jedyna mutacja warunku
usuwa informacje. Specjalizację na podstawie realnej różnicy `p0 -> p1` pozostawia
ALP.

## 7. Covering

**Po co istnieje.** Agent musi mieć od czego zacząć i musi umieć zapisać przejście,
którego żadna dotychczasowa reguła wybranej akcji poprawnie nie przewidziała.

Najważniejsza korekta intuicji: w tej implementacji covering nie jest wyzwalany
wyłącznie przez pusty match set. Uruchamia się, gdy po sprawdzeniu poprzedniego action
setu nie było ani jednego expected case (`acs2-core/src/alp.rs:188-190`,
`acs2-core/src/alp.rs:208-223`). Pusty match/action set jest jednym z takich
przypadków, ale covering wystąpi również wtedy, gdy reguły pasowały i miały właściwą
akcję, lecz wszystkie przewidziały skutek źle.

Nowa reguła zaczyna od all-wildcard condition i effect, wykonanej akcji, `q =
initial_q`, `r = 0`, `ir = initial_ir`, `exp = 0`, `num = 1` oraz bieżących znaczników
czasu (`acs2-core/src/classifier.rs:44-59`). Potem `specialize(p0,p1,false)` ustala
wyłącznie pozycje, które rzeczywiście się zmieniły: w condition wartość z `p0`, w
effect wartość z `p1` (`acs2-core/src/alp.rs:9-18`,
`acs2-core/src/classifier.rs:159-170`).

Przy przejściu bez zmiany reguła coveringowa może więc pozostać całkowicie ogólna z
all-wildcard effect. To poprawny model „po tej akcji nic się nie zmienia”, nie brak
wiedzy o efekcie.

## 8. Marking — najważniejsza intuicja krok po kroku

### Problem, który rozwiązuje Mark

Załóżmy, że reguła jest ogólna i czasem przewiduje dobrze, a czasem źle. Sam fakt
porażki mówi, że jej condition łączy zbyt różne sytuacje, ale nie mówi, **który**
wildcard należy zastąpić konkretną wartością. `Mark` zbiera wartości widziane w
kontekstach porażki. Gdy później ta sama reguła przewidzi dobrze w kontekście
odróżniającym się od marku, ALP może stworzyć wyspecjalizowanego potomka.

Mark ma tablicę zbiorów, a nie jeden zapamiętany stan
(`acs2-core/src/mark.rs:8-17`). Przy pierwszym oznaczeniu zapisuje wartości `p0` tylko
na pozycjach, na których condition ma wildcard
(`acs2-core/src/mark.rs:24-39`). Przy kolejnych oznaczeniach nie otwiera nowych
pozycji: dopisuje wartości tylko do zbiorów już niepustych
(`acs2-core/src/mark.rs:29-30`, `acs2-core/src/mark.rs:42-50`). Mark jest ustawiany w
`unexpected_case`, po obniżeniu `q` (`acs2-core/src/alp.rs:155-164`).

### Przykład liczbowy

Niech percepcja ma cztery pozycje, a rodzic ma:

```text
condition = ##1#
mark      = [ {}, {}, {}, {} ]
```

Pozycja 3 jest już ustalona w condition, więc Mark nie ma jej dalej rozróżniać.

**Krok 1 — pierwsza porażka.** Reguła przewiduje źle dla:

```text
p0 = 0 1 1 0
```

Pierwsze `set_using_condition` zapisuje tylko wildcardowe pozycje 1, 2 i 4:

```text
mark = [ {0}, {1}, {}, {0} ]
```

Dokładnie tę pętlę realizuje `acs2-core/src/mark.rs:32-38`.

**Krok 2 — druga porażka.** Reguła przewiduje źle dla:

```text
p0 = 1 1 1 1
```

Ponieważ mark już istnieje, `complement` dopisuje wartości wyłącznie do wcześniej
otwartych zbiorów:

```text
mark = [ {0,1}, {1}, {}, {0,1} ]
```

Pozycja 3 nadal ma pusty zbiór (`acs2-core/src/mark.rs:42-50`).

**Krok 3 — trafna antycypacja w nowym kontekście.** Później reguła przewiduje dobrze
dla:

```text
p0 = 0 0 1 1
```

`get_differences` klasyfikuje pozycje marku dwojako
(`acs2-core/src/mark.rs:53-65`):

- pozycja 2: zbiór `{1}` **nie zawiera** bieżącego `0`; to kandydat typu `nr1` —
  bieżący kontekst pokazuje wartość niewidzianą tam w porażkach;
- pozycje 1 i 4: zbiory mają więcej niż jedną wartość; to kandydaci typu `nr2`.

Jeżeli istnieje choć jeden kandydat `nr1`, algorytm losuje dokładnie **jedną** taką
pozycję i wpisuje do różnicy jej bieżącą wartość
(`acs2-core/src/mark.rs:67-75`). Tu kandydat jest tylko jeden:

```text
difference = #0##
```

`expected_case` kopiuje rodzica i specjalizuje condition tą różnicą:

```text
rodzic:    ##1#
potomek:   #01#
```

Wykonuje to `acs2-core/src/alp.rs:75-90`. Sens jest następujący: „w porażkach na
pozycji 2 widziałem `1`, a teraz przy poprawnej antycypacji widzę `0`; warto rozdzielić
te konteksty”.

Jeżeli nie ma żadnego `nr1`, ale jakiś zbiór zawiera więcej niż jedną wartość, różnica
ustala **wszystkie** takie pozycje na wartości bieżącego `p0`
(`acs2-core/src/mark.rs:76-82`). Jeżeli nie zachodzi żaden przypadek, różnica pozostaje
all-wildcard i zwykły expected case tylko podnosi `q`.

### Co Mark robi, a czego nie robi

- Mark nie przechowuje nagród i nie wpływa bezpośrednio na `r`.
- Mark nie jest częścią dopasowania condition.
- Samo oznaczenie nie specjalizuje rodzica. Dostarcza sygnału późniejszemu expected
  case; bez trafnej antycypacji po markowaniu ten etap może nie nadejść.
- Potomek dostaje pusty mark (`Classifier::copy_from`), więc nie dziedziczy całej
  historii porażek rodzica (`acs2-core/src/classifier.rs:62-77`).
- Reguła oznaczona nie może być subsumerem (`acs2-core/src/subsumption.rs:3-5`).

Mark jest zatem pamięcią **kontrastu**, nie zwykłym licznikiem błędów: łączy wcześniejsze
konteksty porażki z późniejszym kontekstem sukcesu, aby wybrać atrybut specjalizacji.

## 9. Subsumption

**Po co istnieje.** Jeżeli doświadczona, wiarygodna i bardziej ogólna reguła opisuje
to samo działanie oraz efekt co nowy, bardziej szczegółowy kandydat, przechowywanie obu
struktur może być zbędne. Subsumption pozwala ogólniejszej regule pochłonąć potomka.

Reguła może być subsumerem tylko, gdy jednocześnie:

```text
exp > theta_exp
q > theta_r
mark jest pusty
```

(`acs2-core/src/subsumption.rs:3-5`). Następnie musi być ściśle bardziej ogólna,
jej condition musi subsumować condition potomka, a action i effect muszą się zgadzać
(`acs2-core/src/subsumption.rs:7-18`). „Bardziej ogólna” wymaga mniejszej
specyficzności, nie dopuszcza remisu (`acs2-core/src/classifier.rs:115-117`). Efekty
w praktyce muszą być identyczne, bo `Effect::subsumes` porównuje całe tablice
(`acs2-core/src/effect.rs:40-42`).

W ścieżce ALP wyszukiwany jest najbardziej ogólny subsumer. Jeżeli istnieje, nie
dodaje się dziecka, tylko zwiększa `q` subsumera
(`acs2-core/src/alp.rs:21-43`). Jeśli nie ma subsumera, identyczny klasyfikator w
nowej liście lub action secie również dostaje wzrost `q`; dopiero brak obu powoduje
dodanie nowej struktury (`acs2-core/src/alp.rs:46-59`). Ta ścieżka **nie zwiększa
`num`**.

W GA `do_subsumption` decyduje, czy szukać subsumera; bez niego pozostaje tylko
scalanie identycznej reguły (`acs2-core/src/ga.rs:179-209`). Znaleziony nieoznaczony
klasyfikator dostaje `num += 1`; oznaczony nie jest zwiększany
(`acs2-core/src/ga.rs:211-233`).

Nieoczywistość: `do_subsumption` steruje tylko ścieżką GA. ALP zawsze wywołuje
`does_subsume` i nie sprawdza tej flagi (`acs2-core/src/alp.rs:21-43`).

## 10. Parametry

### Jak czytać „typowy zakres”

`Configuration` nie waliduje zakresów; jest zwykłą strukturą pól
(`acs2-core/src/config.rs:7-29`). Poniższy „typowy zakres” oznacza zakres sensowny dla
formuł i prawdopodobieństw w tej implementacji, nie ograniczenie narzucone przez typ.
`epsilon`, `mu` i `chi` trafiają do `gen_bool`, więc praktycznie muszą być w `[0,1]`
(`acs2-core/src/rng.rs:30-33`). Dla parametrów zależnych od zadania nie istnieje w
repo jeden literaturowo potwierdzony „dobry” przedział.

| Parametr | Domyślnie | Znaczenie i miejsce użycia | Gdy zwiększyć / zmniejszyć | Typowy lub sensowny zakres |
|---|---:|---|---|---|
| `beta` | `0.05` | Współczynnik uczenia `q`, `r`, `ir` i późnej aktualizacji `tav` (`acs2-core/src/config.rs:35`; `acs2-core/src/classifier.rs:84-103`; `acs2-core/src/rl.rs:23-25`). | Większy: szybsza reakcja, większa zmienność i szybsze zapominanie. Mniejszy: stabilniej, ale wolniej. | Zwykle `0 < beta <= 1`; kod nie waliduje. |
| `gamma` | `0.95` | Dyskonto bootstrapu w celu `reward + gamma*bootstrap` (`acs2-core/src/config.rs:36`; `acs2-core/src/rl.rs:23-25`). | Większy: większa waga dalszych nagród. Mniejszy: bardziej krótkowzroczne `r`; `0` zostawia tylko bieżącą nagrodę. | Zwykle `[0,1]`. |
| `theta_i` | `0.1` | Próg nieadekwatności `q < theta_i` i usunięcia po błędnej antycypacji (`acs2-core/src/config.rs:37`; `acs2-core/src/alp.rs:208-212`). | Większy: szybsze/agresywniejsze usuwanie słabych reguł. Mniejszy: dłuższa tolerancja błędów. | Zwykle `[0,1]`, sensownie poniżej `theta_r`. |
| `theta_r` | `0.9` | Próg reliable `q > theta_r`; używany też przez subsumpcję i knowledge (`acs2-core/src/config.rs:38`; `acs2-core/src/subsumption.rs:3-5`; `acs2-core/src/knowledge.rs:34-37`). | Większy: mniej, ale mocniej sprawdzonych reguł. Mniejszy: wcześniejsza wiarygodność i bardziej optymistyczne knowledge/subsumpcja. | Zwykle `[0,1]`, sensownie powyżej `theta_i`. |
| `theta_exp` | `20` | Minimalna dojrzałość subsumera, przy czym kod wymaga `exp > theta_exp`, nie `>=` (`acs2-core/src/config.rs:39`; `acs2-core/src/subsumption.rs:3-5`). | Większy: późniejsza, ostrożniejsza subsumpcja. Mniejszy: szybsza kompresja, większe ryzyko pochłaniania przez niedojrzałą regułę. | Nieujemna liczba całkowita. |
| `theta_as` | `20` | Limit **numerosity bieżącego action setu** podczas wstawiania dzieci GA (`acs2-core/src/config.rs:40`; `acs2-core/src/ga.rs:128-176`). | Większy: mniej usuwania, większe nisze/populacja. Mniejszy: silniejsza presja usuwania. Bez GA jest nieaktywny. | Dodatnia liczba całkowita; nie jest globalnym limitem populacji. |
| `theta_ga` | `100` | Minimalne opóźnienie względem średniego `tga`, aby GA zadziałał (`acs2-core/src/config.rs:41`; `acs2-core/src/ga.rs:8-27`). | Większy: rzadszy GA. Mniejszy: częstsza mutacja, krzyżowanie i presja usuwania. | Nieujemna liczba całkowita w krokach. |
| `mu` | `0.3` | Prawdopodobieństwo usunięcia każdego ustalonego symbolu condition przez mutację (`acs2-core/src/config.rs:42`; `acs2-core/src/ga.rs:74-84`). | Większy: silniejsza/skokowa generalizacja; może usuwać potrzebne cechy. Mniejszy: zachowawcza generalizacja. | `[0,1]`. |
| `chi` | `0.8` | Prawdopodobieństwo krzyżowania, dodatkowo wymagające identycznych efektów (`acs2-core/src/config.rs:43`; `acs2-core/src/ga.rs:258-266`). | Większy: częstsza rekombinacja fragmentów condition. Mniejszy: potomkowie bliżsi rodzicom poza mutacją. | `[0,1]`. |
| `u_max` | `100000` | Próg ograniczania nadmiernej specjalizacji w expected case ALP (`acs2-core/src/config.rs:44`; `acs2-core/src/alp.rs:93-153`). | Mniejszy: generalizacja ALP włącza się wcześniej i mocniej. Większy: pozwala na bardziej szczegółowe reguły; wartość większa niż możliwa specyficzność praktycznie wyłącza gałąź. | Aktywnie zwykle dodatnia liczba rzędu `1..N`; `100000` celowo wyłącza mechanizm na ścieżce maze. Semantyka progu różni się między Pyalcs i Butz. |
| `epsilon` | `0.5` | Prawdopodobieństwo losowej akcji w `EpsilonGreedy` (`acs2-core/src/config.rs:45`; `acs2-core/src/action_selection.rs:57-74`). | Większy: więcej eksploracji. Mniejszy: więcej użycia bieżącej wiedzy. Nie wpływa na `run_exploit_trial`. | `[0,1]`. Benchmarki eksploracyjne nadpisują je na `0.8` (`acs2-bench/src/main.rs:93-104`). |
| `initial_q` | `0.5` | Początkowa jakość zwykłej i coveringowej reguły (`acs2-core/src/config.rs:46`; `acs2-core/src/classifier.rs:26-34`, `acs2-core/src/classifier.rs:44-52`). | Większy: nowa reguła szybciej może stać się reliable i ma wyższy fitness. Mniejszy: bardziej sceptyczny start. | Zwykle `[0,1]`, aby zachować interpretację jakości. |
| `initial_r` | `0.5` | Początkowa predykcja wypłaty dla konstruktora ogólnego (`acs2-core/src/config.rs:47`; `acs2-core/src/classifier.rs:26-34`). Covering mimo tego zaczyna z `r=0` (`acs2-core/src/classifier.rs:44-52`). | Większy: bardziej optymistyczny początkowy fitness. Mniejszy: bardziej zachowawczy. | Skala nagrody/zwrótu; kod nie ogranicza do `[0,1]` i środowiska dają nagrodę `1000`. |
| `initial_ir` | `0.0` | Początkowa predykcja natychmiastowej nagrody (`acs2-core/src/config.rs:48`; `acs2-core/src/classifier.rs:26-34`). | Większy lub mniejszy zmienia tylko punkt startu `ir`; obecnie `ir` nie steruje akcją. | Skala natychmiastowej nagrody; brak walidacji. |

### `u_max` — szczególne zastrzeżenie

`u_max` nie jest limitem populacji ani parametrem coveringu. Działa w expected case,
po uzyskaniu różnicy z Marku. Wariant Pyalcs liczy ustalone, **niezmieniające się**
atrybuty rodzica i może generalizować rodzica; wariant Butz liczy pełną specyficzność
potomka i generalizuje potomka (`acs2-core/src/classifier.rs:141-156`,
`acs2-core/src/alp.rs:93-153`).

Dla MPX wartość nie pochodzi z literatury. Projekt wyprowadza ją z budowy rozwiązania:

- Pyalcs: `u_max = a + 2`;
- Butz: `u_max = a + 3`.

Kod wyprowadzenia jest w `acs2-bench/src/lib.rs:26-33`. Różnica jednego wynika z tego,
że wariant Butz liczy w condition również bit walidacyjny reguły przewidującej zmianę,
a licznik Pyalcs obejmuje tylko atrybuty condition ustalone przy wildcardowym efekcie.
Pełne uzasadnienie i uczciwe zastrzeżenie „derived, not literature” są w
`docs/ARCHITECTURE.md:584-608`. Wartości dla MPX nie należy przedstawiać jako
hiperparametru zaczerpniętego z publikacji.

### Flagi i dodatkowa konfiguracja

| Flaga / pole | Domyślnie | Co faktycznie robi i czy jest używane |
|---|---:|---|
| `do_ga` | `false` | Bramka obu wywołań GA w próbie eksploracyjnej (`acs2-core/src/agent.rs:99-110`, `acs2-core/src/agent.rs:143-153`). Domyślny benchmark maze i podstawowy MPX mają GA wyłączony; binaria pozwalają go włączyć. Eksperymenty MPX M2a/M2b używały GA. |
| `do_pee` | `false` | W tym porcie nie ma reprezentacji probability-enhanced effect. Flaga dociera tylko do argumentu `leave_specialized` w unexpected case (`acs2-core/src/alp.rs:169-170`). `ee` nigdy nie jest ustawiane na `true`. Ustawienie flagi nie włącza pełnego PEE i nie powinno być tak opisywane. |
| `do_action_planning` | `false` | Pole istnieje w konfiguracji (`acs2-core/src/config.rs:24-27`), ale żaden kod wykonawczy w `acs2-core/src` go nie odczytuje. Planowanie akcji nie jest zaimplementowane. |
| `do_subsumption` | `true` | Steruje wyszukiwaniem subsumera tylko przy dodawaniu dziecka GA (`acs2-core/src/ga.rs:179-209`). ALP wykonuje subsumpcję bezwarunkowo (`acs2-core/src/alp.rs:21-43`). Przy domyślnym `do_ga=false` warunkowe użycie flagi jest nieaktywne. |
| `alp_gen_variant` | `Pyalcs` | Wybiera generalizację rodzica zgodną z pyalcs albo generalizację potomka według wariantu Butza (`acs2-core/src/config.rs:1-5`, `acs2-core/src/alp.rs:77-84`). |
| `number_of_possible_actions` | `8`; MPX `2` | Zakres losowania akcji i konfiguracja selektorów (`acs2-core/src/config.rs:32-35`, `acs2-core/src/config.rs:57-61`; `acs2-core/src/action_selection.rs:57-74`). |

## 11. Środowiska

### Labirynt: percepcja 8-sensorowa

Percepcja ma długość osiem (`acs2-envs/src/maze.rs:8`). Kolejność sensorów i zarazem
kierunków ruchu to:

```text
N, NE, E, SE, S, SW, W, NW
```

co wynika z tablicy przesunięć `(-1,0), (-1,1), ...`
(`acs2-envs/src/maze.rs:15-24`). Każdy sensor zwraca kod sąsiedniego pola jako znak
ASCII: ścieżka `'0'`, ściana `'1'`, nagroda `'9'`
(`acs2-envs/src/maze.rs:10-13`, `acs2-envs/src/maze.rs:75-83`). To nie są
współrzędne agenta.

Akcja wybiera ten sam indeks kierunku. Wejście w ścianę nie zmienia pozycji; wejście
na inne pole przesuwa agenta, a pole `9` kończy epizod
(`acs2-envs/src/maze.rs:109-124`). Reset losuje jednolicie jedną z komórek ścieżki
o kodzie `0` (`acs2-envs/src/maze.rs:87-106`). Limity epizodu są własnością geometrii,
a środowisko ustawia `truncated` po ich osiągnięciu
(`acs2-envs/src/maze.rs:31-32`, `acs2-envs/src/maze.rs:122-130`).

### Multiplekser

Niech `a` oznacza liczbę bitów adresowych, a `k` liczbę bitów wejścia klasycznego
multipleksera. Rozmiary spełniają:

```text
k = a + 2^a
```

Pierwsze `a` bitów tworzy adres w kolejności most-significant-bit first, a wskazany
bit danych jest poprawną akcją `0` albo `1`
(`acs2-envs/src/multiplexer.rs:15-35`). Stąd:

| `a` | `k = a + 2^a` |
|---:|---:|
| 2 | 6 |
| 3 | 11 |
| 4 | 20 |
| 5 | 37 |
| 6 | 70 |
| 7 | 135 |

Dokładnie te aliasy instancjuje kod (`acs2-envs/src/multiplexer.rs:248-263`). Wartość
137 nie spełnia równania; prawidłowy duży rozmiar to 135.

Środowisko dodaje jeszcze jeden, końcowy bit walidacyjny, więc percepcja core ma
`N = k + 1`. Reset losuje `k` bitów wejścia i ustawia walidację na `0`
(`acs2-envs/src/multiplexer.rs:68-76`). Poprawna akcja zmienia tylko walidację
`0 -> 1` i daje nagrodę `1000`; błędna nie zmienia percepcji i daje `0`. Oba wyniki
kończą epizod po jednym kroku (`acs2-envs/src/multiplexer.rs:78-92`). Dzięki bitowi
walidacyjnemu zadanie ma rzeczywiste przejście do antycypowania, zamiast być tylko
statyczną klasyfikacją.

**Idealna reguła** funkcji multipleksera musi sprawdzić `a` bitów adresowych oraz
jeden wybrany bit danych, a pozostałe bity danych może zostawić jako wildcardy. Jej
docelowa specyficzność condition wynosi więc `a+1`. To własność matematyczna zadania,
nie funkcja obliczana w core; jest jawnie przyjęta jako metryka eksperymentalna w
`reports/MPX_final.md:8-16`.

Dlaczego MPX jest trudny dla LCS:

- przestrzeń wejść ma `2^k` elementów;
- ważny bit danych zależy od kombinacji bitów adresowych;
- reguła zbyt ogólna miesza adresy lub wartości wybranego bitu i antycypuje źle;
- reguła zbyt szczegółowa pasuje do wykładniczo małej niszy i rzadko wraca do action
  setu, więc ALP i GA mają mało okazji ją poprawiać.

Ostatni punkt jest empirycznie widoczny w projekcie jako wyścig specjalizacji z
generalizacją (`reports/MPX_final.md:60-70`). Nie wynika on z jednej instrukcji kodu,
lecz z połączenia dopasowania condition (`acs2-core/src/population.rs:59-63`), ALP na
poprzednim action secie (`acs2-core/src/agent.rs:77-89`) i progowego uruchamiania GA
(`acs2-core/src/ga.rs:8-27`).

## 12. Metryka `knowledge`

`knowledge` jest ułamkiem przejść, dla których istnieje **co najmniej jeden reliable
klasyfikator**, który jednocześnie:

1. ma tę samą akcję,
2. jego condition pasuje do `p0`,
3. jego effect poprawnie przewiduje `p0 -> p1`.

Predykat jest w `acs2-core/src/knowledge.rs:17-24`, filtrowanie reliable i obliczenie
ułamka w `acs2-core/src/knowledge.rs:26-55`. Zakres wynosi od `0` do `1`; dla pustej
listy przejść funkcja zwraca `0`.

Knowledge uwzględnia zarówno zmianę, jak i poprawnie przewidziany brak zmiany.
All-wildcard effect zalicza przejście identycznościowe, ponieważ dla każdej pozycji
wymaga `before == after` (`acs2-core/src/effect.rs:9-15`). Test pokazuje, że populacja
z samymi regułami zmiany zatrzymuje się na `0.5`, a komplet reguł zmiany i braku zmiany
osiąga `1.0` (`acs2-core/src/knowledge.rs:113-139`).

Dla MPX wyczerpujący zbiór zawiera wszystkie `2^k` wejścia i obie akcje
(`acs2-envs/src/multiplexer.rs:119-130`). Liczba par rośnie wykładniczo. Port ma szybką
dokładną implementację, która dla reliable reguł zaznacza pokryte komórki
`(input, action)` i liczy ich udział (`acs2-envs/src/multiplexer.rs:158-229`), ale sama
tablica nadal ma rozmiar proporcjonalny do `2^(k+1)`.

Dlatego `evaluate_knowledge` używa dokładnego wyniku dla `k <= 20`, a dla większych
`k` przechodzi na próbkę (`acs2-envs/src/multiplexer.rs:231-245`). Próbkowanie losuje
każdy bit wejścia i dla każdego wylosowanego wejścia dodaje obie akcje; losowanie jest
ze zwracaniem, bo kolejne wejścia są generowane niezależnie
(`acs2-envs/src/multiplexer.rs:132-148`). Standardowe binarium MPX przekazuje 50 000
wylosowanych wejść i stały seed metryki (`acs2-bench/src/bin/mpx.rs:11-13`). Wynik dla
dużych `k` jest estymatorem, nie wyczerpującym dowodem pokrycia.

## 13. Jedna próba eksploracyjna od początku do końca

Poniższa kolejność jest ważna, zwłaszcza słowo **poprzedni**.

1. Środowisko wykonuje `reset`; agent ma bieżące `state`, licznik kroków `0` i brak
   poprzedniego przejścia (`acs2-core/src/agent.rs:69-72`).
2. Agent skanuje populację i tworzy `[M]` dla bieżącego stanu
   (`acs2-core/src/agent.rs:74-75`).
3. Jeżeli istnieje poprzedni krok, agent ma już jego `p0`, akcję, reward i obecne
   `p1`. Uruchamia ALP na poprzednim `[A]`
   (`acs2-core/src/agent.rs:77-88`). ALP aktualizuje doświadczenie i mark, rozróżnia
   expected/unexpected, może utworzyć dzieci, wykonać covering i usunąć inadequate.
4. Ponieważ w Rust indeksy klasyfikatorów mogą zmienić się po usunięciu, match set jest
   wyliczany ponownie z aktualnej populacji (`acs2-core/src/agent.rs:89`).
5. Estymator oblicza bootstrap następnego stanu. RL aktualizuje `r` i `ir` reguł
   poprzedniego `[A]` (`acs2-core/src/agent.rs:90-98`).
6. Jeśli `do_ga`, GA może zadziałać na poprzednim `[A]`; potem `[M]` znów jest
   wyliczany (`acs2-core/src/agent.rs:99-110`).
7. Selektor wybiera akcję z aktualnego `[M]`, a populacja tworzy nowe `[A]`
   (`acs2-core/src/agent.rs:113-115`).
8. Środowisko wykonuje akcję i zwraca `p1`, reward oraz znaczniki końca. Agent zwiększa
   liczbę kroków (`acs2-core/src/agent.rs:117-120`).
9. Jeżeli epizod trwa, nowy `[A]`, `p0`, akcja i reward zostają zapamiętane jako
   `PreviousStep`; uczenie tego przejścia nastąpi na początku następnej iteracji, gdy
   znany jest już bieżący stan (`acs2-core/src/agent.rs:157-162`).
10. Jeżeli epizod się kończy, agent nie czeka na następną iterację: wykonuje terminalny
    ALP, RL z bootstrapem `0` i opcjonalny GA, po czym kończy próbę
    (`acs2-core/src/agent.rs:122-154`).

Dla single-step MPX kroki 3-6 na początku nie zachodzą, bo nie ma poprzedniego kroku.
Całe uczenie przejścia odbywa się w terminalnym kroku 10.

Próba eksploatacyjna jest krótsza: tworzy `[M]`, aktualizuje RL poprzedniego `[A]`,
wybiera `BestAction`, wykonuje krok i na końcu robi terminalny RL. Nie ma ALP, coveringu,
GA ani zmian strukturalnych populacji (`acs2-core/src/agent.rs:190-230`).

## 14. Pułapki i nieoczywistości

### 14.1 Port jest zgodny z pyalcs, nie automatycznie z opisem Butza

Wariant `Pyalcs` generalizacji expected case zmienia **rodzica**, licząc tylko ustalone
atrybuty o wildcardowym efekcie. Wariant `Butz` zmienia **potomka**, licząc pełną
specyficzność (`acs2-core/src/alp.rs:93-153`). Nie wolno połączyć ich w jeden
„kanoniczny” opis. `u_max` trzeba wyprowadzać osobno dla wariantu.

### 14.2 `does_anticipate_change` to nie `does_anticipate_correctly` — NAMED HAZARD

Pierwsze pytanie sprawdza tylko, czy effect ma jakiś symbol konkretny
(`acs2-core/src/classifier.rs:123-125`). Służy do filtrowania kandydatów przy wyborze
akcji i bootstrapie (`acs2-core/src/action_selection.rs:32-40`,
`acs2-core/src/population.rs:73-80`). Drugie porównuje cały efekt z konkretnym
przejściem (`acs2-core/src/classifier.rs:119-121`) i jest używane przez ALP oraz
knowledge. Reguły poprawnie przewidujące brak zmiany są niewidoczne dla greedy action
selection, ale muszą być zaliczone przez knowledge. Pomylenie funkcji sztucznie obcina
MPX knowledge do `0.5`; hazard jest udokumentowany w
`docs/ARCHITECTURE.md:348-362`.

### 14.3 `seed = base_seed + repeat` — NAMED HAZARD

Powtórzenie `r` nie używa ponownie base seeda, tylko `base_seed + r`. Zatem nagłówek
`seed=42`, `n_exp=3` oznacza seedy 42, 43 i 44. To dotyczy interpretacji logów i może
zmienić wniosek o wariantach przy silnej zmienności między seedami. Pełna pułapka
formatu logów jest w `docs/ARCHITECTURE.md:779-790`.

### 14.4 Wildcard ma dwie semantyki

`#` w condition to „pasuje do wszystkiego”; `#` w effect to „przewiduję brak zmiany”.
To najłatwiejszy sposób, by przypadkiem opisać model przejścia jako „nieznany”. Kod
rozdziela te znaczenia w `acs2-core/src/condition.rs:10-23` i
`acs2-core/src/effect.rs:9-25`.

### 14.5 Covering nie znaczy tylko „pusty match set”

Warunkiem jest brak jakiegokolwiek expected case w poprzednim action secie
(`acs2-core/src/alp.rs:188-223`). Covering może zajść mimo istniejących dopasowanych
reguł.

### 14.6 Trafna antycypacja nie zawsze zwiększa `q` rodzica

Niepusty `difference` tworzy potomka i omija prosty wzrost jakości rodzica
(`acs2-core/src/alp.rs:68-90`). Wzrost może dostać subsumer albo identyczna reguła przy
scalaniu (`acs2-core/src/alp.rs:41-54`).

### 14.7 `u_max = 100000` i `do_ga = false` dają specialize-only na maze

Domyślne wartości są w `acs2-core/src/config.rs:41-53`. Przy `u_max` znacznie większym
od `N` pętle generalizacji expected case nie działają, a wyłączony GA usuwa drugie
źródło generalizacji. Jest to celowo zachowana semantyka protokołu pyalcs, nie brak
implementacji; odnotowuje ją `docs/ARCHITECTURE.md:248-266`.

### 14.8 `u_max` nie jest limitem populacji ani parametrem coveringu

Jest używane wyłącznie w generalizacji expected case
(`acs2-core/src/alp.rs:77-153`). W szczególności `theta_as` również nie jest globalnym
limitem populacji: usuwa tylko z bieżącego action setu i tylko podczas GA
(`acs2-core/src/ga.rs:138-176`). Poza usuwaniem inadequate przez ALP i presją GA kod
nie ma globalnego limitu liczby makroklasyfikatorów.

### 14.9 Krzyżowanie nie jest monotoniczną generalizacją

Mutacja wyłącznie zamienia symbole na wildcardy (`acs2-core/src/ga.rs:74-84`), lecz
krzyżowanie zamienia fragmenty warunku (`acs2-core/src/ga.rs:86-105`). Opisywanie
każdego dziecka GA jako „bardziej ogólnego” jest zbyt mocne.

### 14.10 Flagi PEE i action planning nie oznaczają gotowych funkcji

`do_pee` ma tylko częściowe podłączenie w unexpected case, a `do_action_planning` nie
ma czytelnika w wykonywalnym core. Domyślnie obie są `false`
(`acs2-core/src/config.rs:49-52`). Nie należy raportować eksperymentu z tymi funkcjami
na podstawie samego istnienia pól konfiguracyjnych.

### 14.11 `do_subsumption = false` nie wyłącza subsumpcji ALP

Flaga jest sprawdzana w GA (`acs2-core/src/ga.rs:179-209`), ale ścieżka ALP zawsze
szuka subsumera (`acs2-core/src/alp.rs:21-43`).

### 14.12 Eksploatacja ignoruje epsilon i nadal wykonuje RL

`run_exploit_trial` konstruuje `BestAction` bez odczytu epsilon i nie uruchamia ALP ani
GA, ale aktualizuje `r` oraz `ir` (`acs2-core/src/agent.rs:171-230`). „Frozen
population” oznacza tu brak zmian struktury i `q`, nie całkowity brak uczenia liczb.

### 14.13 Rust celowo nie kopiuje błędu usuwania pyalcs

pyalcs usuwał element action setu podczas iterowania i pomijał następny element. Rust
klonuje `original_action_set`, zbiera ofiary i usuwa je po pętli
(`acs2-core/src/alp.rs:188-218`, `acs2-core/src/alp.rs:234-240`). Jest to świadoma,
udokumentowana różnica, w której Rust jest stroną poprawną
(`docs/ARCHITECTURE.md:60-82`).

### 14.14 Truncation jest końcem z bootstrapem zero

Agent nie ma własnego limitu kroków; czeka na `terminated || truncated`
(`acs2-core/src/agent.rs:74-163`). Maze musi samo
ustawić truncation (`acs2-envs/src/maze.rs:122-130`). Oba rodzaje końca dostają
bootstrap `0`, co zachowuje parytet z bazowym Gym/pyalcs. Nie wolno „naprawić” tylko
truncation do niezerowego bootstrapu bez świadomej zmiany semantyki eksperymentu.

### 14.15 Najlepsza akcja i RL nie używają tych samych wag co selekcja rodziców GA

- wybór akcji: `q * r * num` (`acs2-core/src/action_selection.rs:44-54`);
- bootstrap: maksimum `q * r`, bez `num` (`acs2-core/src/population.rs:73-80`);
- rodzice GA: `q^3 * num`, bez `r` (`acs2-core/src/ga.rs:56-71`).

Zastępowanie wszystkich trzech pojęciem „fitness” ukrywa realne różnice kodu.

### 14.16 `ir` i `ee` są obecnie słabiej podłączone, niż sugerują nazwy

`ir` jest uczone, lecz nie wpływa na fitness ani selekcję akcji
(`acs2-core/src/rl.rs:23-25`, `acs2-core/src/classifier.rs:80-82`). `ee` jest
inicjalizowane i zerowane, ale nigdy ustawiane na `true`
(`acs2-core/src/classifier.rs:26-40`, `acs2-core/src/classifier.rs:131-134`).

### 14.17 Reprezentacja symboli to bajty ASCII

Maze i MPX emitują `Token(b'0' + value)`, a nie numeryczne `Token(0)` i `Token(1)`
(`acs2-envs/src/maze.rs:75-83`, `acs2-envs/src/multiplexer.rs:68-75`). Zmieszanie tych
reprezentacji kompiluje się, ale condition nie będzie pasował.

### 14.18 Mark jest kosztowny pamięciowo przy dużym `N`

Każdy klasyfikator ma tablicę `N` struktur `BTreeSet`, nawet gdy zbiory są puste
(`acs2-core/src/mark.rs:1-17`). Pomiary projektu pokazują, że dla dużych MPX jest to
dominująca część rozmiaru klasyfikatora; to obserwacja pomiarowa opisana w
`docs/ARCHITECTURE.md:524-543`, nie wniosek wyłącznie z nazwy typu.

### 14.19 Asymetria subsumpcji nie pochodzi z samego `Condition::subsumes`

`Condition::subsumes` używa tego samego symetrycznego testu zgodności symboli co
dopasowanie: wildcard po dowolnej stronie wystarcza
(`acs2-core/src/condition.rs:10-12`, `acs2-core/src/condition.rs:25-27`). Kierunek
„starsza reguła pochłania bardziej szczegółową” powstaje dopiero przez osobny, ścisły
warunek mniejszej specyficzności w `does_subsume`
(`acs2-core/src/subsumption.rs:13-18`, `acs2-core/src/classifier.rs:115-117`). Czytanie
samej metody `Condition::subsumes` daje więc mylący obraz relacji.

## 15. Pomysły z ExSTraCS warte przetestowania w ACS2

1. **Lokalna ważność atrybutów** — osobny ranking dla akcji lub niszy; podnosić
   ważność pozycji, które Mark wskazał jako potrzebne do rozróżnienia przejść.
2. **Kierowane uogólnianie** — gdy trzeba wstawić `#`, usuwać najpierw najmniej ważny
   atrybut przewidujący brak zmiany, zamiast wybierać pozycję losowo.
3. **Osobny limit GA** — pozostawić `alp_u_max` dla potomków ALP, a dodać
   `ga_max_condition_specificity` kontrolowany po mutacji i crossoverze.
4. **Cel miękki i limit twardy** — lekko karać reguły ponad wartością docelową, ale
   wymuszać skrócenie dopiero ponad limitem; nie usuwa to reguł przejściowych zbyt wcześnie.
5. **Ochrona młodych reguł** — mocno uogólniać dopiero reguły z wysokim `q`, odpowiednim
   `exp` i pustym Markiem.

Sugerowana kolejność wdrażania: ranking lokalny → kierowane uogólnianie → limit po
crossoverze → cel miękki/limit twardy → bramka `q/exp/mark`.

Nie przenosić wprost: mutacji GA `# → symbol`, losowego coveringu z dokładnie `u_max`
symbolami, jednego globalnego rankingu dla MPX ani jednego wspólnego limitu dla ALP i GA.
W ACS2 specjalizację wykonuje już ALP; GA powinno przede wszystkim bezpiecznie
generalizować.

## 16. Skrót do odświeżenia przed rozmową

1. `condition` mówi **kiedy**, `action` mówi **co zrobić**, `effect` mówi **co się
   zmieni**, `mark` pomaga znaleźć **po czym rozdzielić konteksty**.
2. Wildcard condition znaczy „dowolnie”; wildcard effect znaczy „bez zmiany”.
3. `q` ocenia antycypację, `r` przewiduje zdyskontowaną wypłatę, `ir` nagrodę
   natychmiastową; fitness to `q*r`.
4. ALP specjalizuje model na podstawie realnych przejść; RL uczy wartości; GA daje
   presję generalizującą głównie przez usuwającą symbole mutację.
5. Reliable to `q > theta_r`; inadequate to `q < theta_i`; oba progi są ostre.
6. Covering następuje, gdy żaden klasyfikator poprzedniej akcji nie przewidział
   przejścia poprawnie, nie tylko wtedy, gdy `[M]` był pusty.
7. Knowledge mierzy pokrycie przejść przez reliable reguły, włącznie z brakiem zmiany.
8. Na maze `u_max=100000` i domyślne GA-off oznaczają specialize-only. Dla MPX aktywne
   `u_max` jest wyprowadzone osobno dla Pyalcs i Butz, nie zaczerpnięte z literatury.
9. Całe uczenie strukturalne zachodzi w eksploracji; eksploatacja robi tylko RL i
   zawsze używa `BestAction`.
10. Przy interpretacji wyników patrz razem na knowledge, liczbę reliable i ich
    specyficzność. Żadna z tych liczb osobno nie opisuje pełnego stanu uczenia.
