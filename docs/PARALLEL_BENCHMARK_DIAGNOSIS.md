# Parallel Benchmark Diagnosis

> Final evaluation of the current Windows parallelization in `plagih`.
>
> Sources:
> - `plagih/test/benchmarks/bench_output.txt`
> - `plagih/test/benchmarks/bench_resources_output.txt`
>
> Stand: 2026-03-16

---

## Setup

- OS: Windows
- CPU: 8 physische Kerne / 16 Threads
- Hauptbenchmark: `plagih/test/benchmarks/bench_diagnose_full.py`
- Ressourcen-Profiler: `plagih/test/benchmarks/bench_parallel_resources.py`
- Hauptkonfiguration:
  - `pop=1000`
  - `gens=5`
  - `compare_pops=(1000, 10000)`
  - `batch_sizes=(1, 32, 128, 0)`
- Ressourcen-Snapshot:
  - `pops=(1000, 10000)`
  - `gens=1`
  - `workers=(0, 8)`

---

## Executive Summary

### Wichtigste Befunde

1. **Die aktuelle Parallelisierung funktioniert jetzt gut und skaliert real.**
   - Beste steady-state-Konfiguration im fertigen Lauf ist **`parallel(8w)`**.
   - Steady-state-Speedup:
     - `pop=1000`: **2.98×**
     - `pop=10000`: **3.49×**

2. **Die größten historischen IPC-Bremsen sind weitgehend gelöst.**
   - Shared Memory für `df_train`: **1.68×** schnellerer Worker-Startup
   - Pre-selection statt Legacy-Update: **4.8×** weniger IPC-Kosten

3. **Die aktuelle Batch-Zone für die Runtime ist grob `32..128` Tasks pro Batch.**
   - `pop=1000`: **32** ist klar am besten
   - `pop=10000`: **128** ist knapp vor `32`
   - Große Auto-Batches (`~1 batch/worker`) sind nicht optimal

4. **`gen_create_initial()` ist ein großer fixer Kostenblock und immer noch sequentiell.**
   - `pop=10000` Initialisierung kostet pro Konfiguration ~**44–47 s**
   - dieser Block fällt für sequential und parallel fast identisch an

5. **RAM ist ein echter Parallelisierungsfaktor, aber aktuell noch kein unmittelbarer Killer.**
   - Peak RSS bei `parallel(8w)`:
     - `pop=1000`: **1.29 GB**
     - `pop=10000`: **1.50 GB**
   - Auffällig: Der Großteil des zusätzlichen Parallel-RAMs liegt in den Worker-Prozessen (**~1.13–1.18 GB Child RSS**)
   - Der Schritt von `pop=1000` zu `pop=10000` vergrößert den Parallel-Peak weniger stark als erwartet, weil der größte RAM-Block offenbar fixer Worker-/Interpreter-/Import-/Pool-Overhead ist

---

## 1. Faktorübersicht für die aktuelle Parallelisierung

| Faktor | Messgröße | Einfluss | Status |
|---|---|---|---|
| Worker-Startup / Spawn | Pool-Init-Zeit | Mittel | verbessert durch Shared Memory |
| `df_train`-Transport | Init-Payload / Pool-Init | Klein bis mittel | verbessert |
| Population-IPC | `_update_worker_state` vs. pre-selection | Hoch | stark verbessert |
| Pre-selection im Main-Prozess | `pre_select_for_tasks(...)` | Mittel | aktueller Hauptblock im IPC-Pfad |
| Task-Granularität | Avg Task Time ~3.9 ms | Hoch | weiterhin limitierend |
| Batchgröße | 1 / 32 / 128 / auto | Hoch | Sweet Spot gefunden |
| Result-Rücktransport | Result-pickle dumps | Niedrig | kein Bottleneck |
| Worker-Zahl | 2 / 4 / 8 | Hoch | 8 derzeit am besten |
| Initialpopulation | `gen_create_initial()` | Hoch | weiter sequentiell |
| RAM (Main-Prozess) | RSS | Mittel | skaliert mit Population |
| RAM (Worker-Prozesse) | Child RSS | Hoch | fixer Parallel-Overhead |
| CPU-Auslastung | system CPU avg/peak | Mittel | klar erhöht in parallel, aber nicht voll gesättigt |
| SymPy-Pathologien | Recursion / Hänger | Hoches Risiko | durch Guards entschärft |
| Debug-Logging | eager f-string / `str_as_expr()` | Hoches Risiko | gefixt |

---

## 2. Transport- und IPC-Faktoren

### 2.1 Pickle-Größen (`pop=1000`)

| Objekt | Größe |
|---|---:|
| `evolve` | 1.0 KB |
| `df_train` | 58.1 KB |
| `pop_genepool` | 388.3 KB |
| `paretofront` | 1.6 KB |
| Gesamt | 449.6 KB |

**Interpretation:**
- `df_train` ist klein.
- `pop_genepool` dominiert das Datenvolumen.
- Shared Memory für `df_train` ist sinnvoll, aber **nicht** der größte Hebel.

### 2.2 `df_train`: Pickle vs. Shared Memory

| Metrik | Wert |
|---|---:|
| Pickle-Payload | 58.1 KB |
| Shared-Memory-Raw-Buffer | 57.4 KB |
| Shared-Memory-Metadaten | 75 B |
| Pickled DataFrame Init | 3558.7 ms |
| Shared-Memory Attach | 2123.1 ms |
| Startup-Speedup | 1.68× |

**Interpretation:**
- Shared Memory spart beim Pool-Startup messbar Zeit.
- Der Effekt ist real, aber verglichen mit den gesamten Generationkosten sekundär.

### 2.3 Legacy-IPC vs. Pre-selection

| Metrik | Legacy | Pre-selection | Faktor |
|---|---:|---:|---:|
| IPC-Zeit (`4w`) | 1072.7 ms | 221.7 ms | 4.8× |

Zerlegung der neuen Variante:

| Teil | Zeit |
|---|---:|
| `pre_select_for_tasks(...)` | 197.6 ms |
| Batch dumps | 24.2 ms |
| Gesamt | 221.7 ms |

**Interpretation:**
- Die alte Population-IPC war ein echter Killer.
- Heute ist der große Restblock in diesem Teil nicht mehr Pickle, sondern **Pre-selection im Main-Prozess**.

---

## 3. Task-Granularität und Batching

### 3.1 Per-Task-Compute (`pop=1000`)

| Metrik | Wert |
|---|---:|
| Tasks | 900 |
| Successful candidates | 946 |
| Total | 3531.2 ms |
| Avg per task | 3.9 ms |

**Interpretation:**
- 3.9 ms pro Task ist weiterhin kurz.
- Genau dadurch bleiben Scheduling-, Queue- und Submission-Overhead relevant.
- Parallelisierung braucht deshalb **Chunking**, keine Mini-Tasks.

### 3.2 Batch-Vergleich `pop=1000`

| Batch | #Batches | Payload | Avg time | Speedup vs `1` |
|---|---:|---:|---:|---:|
| `1` | 900 | 567.1 KB | 3328.2 ms | 1.00× |
| `32` | 29 | 320.1 KB | 1338.1 ms | 2.49× |
| `128` | 8 | 319.6 KB | 1500.1 ms | 2.22× |
| `auto(225)` | 4 | 310.4 KB | 1510.3 ms | 2.20× |

### 3.3 Batch-Vergleich `pop=10000`

| Batch | #Batches | Payload | Avg time | Speedup vs `1` |
|---|---:|---:|---:|---:|
| `1` | 9000 | 5.5 MB | 14288.5 ms | 1.00× |
| `32` | 282 | 3.1 MB | 11903.3 ms | 1.20× |
| `128` | 71 | 3.0 MB | 11625.6 ms | 1.23× |
| `auto(2250)` | 4 | 3.0 MB | 12676.9 ms | 1.13× |

### 3.4 Kreuztabelle: Sweet Spot nach Population

| Population | Best Batch | Zweitbester | Auto-Batch relativ zum Besten |
|---|---|---|---|
| `1000` | `32` | `128` | 1.13× langsamer |
| `10000` | `128` | `32` | 1.09× langsamer |

**Interpretation:**
- Die Heuristik „ungefähr ein Batch pro Worker“ ist aktuell **zu grob**.
- Der stabile Arbeitsbereich liegt bei **32 bis 128 Tasks pro Batch**.
- Das passt zur aktuellen Runtime-Änderung in `parallel.py`, mehrere kleinere Chunks statt eines Großbatches zu verwenden.

---

## 4. Result-Rücktransport

| Batch | Payload | Dumps |
|---|---:|---:|
| `1` | 1.2 KB | 0.1 ms |
| `32` | 15.3 KB | 1.1 ms |
| `128` | 23.2 KB | 1.6 ms |
| `auto(63)` | 22.9 KB | 1.5 ms |

**Interpretation:**
- Der Rücktransport der `TaskResult`s ist klein.
- Das ist **kein** Prioritätshebel.

---

## 5. End-to-End-Skalierung

## 5.1 `pop=1000`, steady-state (gen 2+)

| Config | Avg/Gen | Speedup | Effizienz |
|---|---:|---:|---:|
| sequential | 3969.4 ms | 1.00× | - |
| parallel(2w) | 2788.3 ms | 1.42× | 71% |
| parallel(4w) | 1817.9 ms | 2.18× | 55% |
| parallel(8w) | 1333.0 ms | 2.98× | 37% |

## 5.2 `pop=10000`, steady-state (gen 2+)

| Config | Avg/Gen | Speedup | Effizienz |
|---|---:|---:|---:|
| sequential | 41612.2 ms | 1.00× | - |
| parallel(2w) | 28550.6 ms | 1.46× | 73% |
| parallel(4w) | 17097.8 ms | 2.43× | 61% |
| parallel(8w) | 11921.6 ms | 3.49× | 44% |

## 5.3 Kreuztabelle: Worker-Skalierung vs Population

| Population | 2 Worker | 4 Worker | 8 Worker | Best |
|---|---:|---:|---:|---|
| `1000` | 1.42× | 2.18× | **2.98×** | `8w` |
| `10000` | 1.46× | 2.43× | **3.49×** | `8w` |

## 5.4 Kreuztabelle: Effizienz vs Population

| Population | 2 Worker | 4 Worker | 8 Worker |
|---|---:|---:|---:|
| `1000` | 71% | 55% | 37% |
| `10000` | 73% | 61% | 44% |

**Interpretation:**
- Die aktuelle Parallelisierung skaliert bei großer Population **besser** als bei kleiner.
- Mehr Arbeit pro Generation amortisiert die Parallel-Overheads besser.
- `8w` ist auf diesem 8-Core-System im aktuellen Stand die beste Konfiguration.
- Die Effizienz bleibt unter linear, ist aber für `pop=10000` schon klar brauchbar.

---

## 6. Initialpopulation als eigener Kostenblock

`gen_create_initial()` läuft weiterhin sequentiell.

| Population | Sequential init | Parallel(8w) init | Befund |
|---|---:|---:|---|
| `1000` | 4288.7 ms | 4493.1 ms | praktisch gleich |
| `10000` | 44366.6 ms | 45096.9 ms | praktisch gleich |

**Interpretation:**
- Der Init-Block profitiert aktuell **nicht** von `parallel=`.
- Für `pop=10000` kostet allein die Initialpopulation pro Konfiguration ~45 s.
- Das erklärt einen relevanten Teil der langen Gesamtlaufzeit von Punkt 11.

---

## 7. CPU- und RAM-Profiling

Die direkten Ressourcenwerte stammen aus `bench_parallel_resources.py`.

**Wichtig:** Die Ressourcenläufe wurden mit `gens=1` aufgenommen. Sie sind
damit ideal für CPU-/RAM-Vergleiche und relative Worker-Kosten, aber **nicht**
für steady-state-Rankings gedacht. Für Performance-Rankings gilt weiterhin
Abschnitt 5 mit den `gens=5`-Messungen aus `bench_output.txt`.

### 7.1 Ressourcen-Kreuztabelle (vollständig: `0/2/4/8` Worker)

| Pop | Config | Avg/Gen | Peak RSS | Child RSS | Peak CPU |
|---|---|---:|---:|---:|---:|
| `1000` | sequential | 3521.6 ms | 161.4 MB | 0 B | 100.8% |
| `1000` | parallel(2w) | 4397.8 ms | 457.7 MB | 293.4 MB | 100.2% |
| `1000` | parallel(4w) | 3397.8 ms | 750.0 MB | 581.1 MB | 106.2% |
| `1000` | parallel(8w) | 3659.2 ms | 1.29 GB | 1.13 GB | 104.2% |
| `10000` | sequential | 36846.3 ms | 290.2 MB | 0 B | 106.1% |
| `10000` | parallel(2w) | 25739.3 ms | 632.9 MB | 319.4 MB | 106.2% |
| `10000` | parallel(4w) | 16716.7 ms | 940.9 MB | 622.5 MB | 106.2% |
| `10000` | parallel(8w) | 12007.6 ms | 1.49 GB | 1.18 GB | 106.2% |

### 7.2 Workerzahl × Peak RSS / Child RSS

| Worker | `pop=1000` Peak RSS | `pop=1000` Child RSS | `pop=10000` Peak RSS | `pop=10000` Child RSS |
|---|---:|---:|---:|---:|
| `0` | 161.4 MB | 0 B | 290.2 MB | 0 B |
| `2` | 457.7 MB | 293.4 MB | 632.9 MB | 319.4 MB |
| `4` | 750.0 MB | 581.1 MB | 940.9 MB | 622.5 MB |
| `8` | 1.29 GB | 1.13 GB | 1.49 GB | 1.18 GB |

### 7.3 Workerzahl × `gen_1` System-CPU-Auslastung

| Worker | `pop=1000` sys CPU avg | `pop=10000` sys CPU avg |
|---|---:|---:|
| `0` | 29.2% | 26.9% |
| `2` | 43.2% | 31.7% |
| `4` | 37.6% | 55.7% |
| `8` | 58.7% | 57.5% |

### 7.4 Population × RAM pro Kandidat

Hier ist nur grob `Peak RSS / pop_size` betrachtet. Das ist kein "reiner"
Speicher pro Kandidat, aber ein nützlicher Dichteindikator.

| Pop | Config | Peak RSS pro Kandidat |
|---|---|---:|
| `1000` | sequential | ~161 KB |
| `1000` | parallel(2w) | ~458 KB |
| `1000` | parallel(4w) | ~750 KB |
| `1000` | parallel(8w) | ~1.29 MB |
| `10000` | sequential | ~29 KB |
| `10000` | parallel(2w) | ~63 KB |
| `10000` | parallel(4w) | ~94 KB |
| `10000` | parallel(8w) | ~149 KB |

### 7.5 Speedup pro zusätzlichem GB RAM (`gens=1`, relativ zu sequential)

Formel:

```text
(speedup_vs_sequential - 1.0) / extra_peak_rss_in_GiB
```

| Pop | Config | Speedup vs seq | zusätzlicher Peak-RAM | Speedup-Gewinn pro GiB |
|---|---|---:|---:|---:|
| `1000` | parallel(2w) | 0.80× | ~0.29 GiB | ~-0.69×/GiB |
| `1000` | parallel(4w) | 1.04× | ~0.57 GiB | ~0.06×/GiB |
| `1000` | parallel(8w) | 0.96× | ~1.10 GiB | ~-0.03×/GiB |
| `10000` | parallel(2w) | 1.43× | ~0.33 GiB | ~1.29×/GiB |
| `10000` | parallel(4w) | 2.20× | ~0.64 GiB | ~1.90×/GiB |
| `10000` | parallel(8w) | 3.07× | ~1.17 GiB | ~1.71×/GiB |

### 7.6 Interpretation CPU

- Sequential nutzt im Wesentlichen **einen Kern voll**.
- Parallel erhöht die System-CPU-Last klar, aber nicht perfekt monoton:
  - bei `pop=1000` springt `4w -> 8w` deutlich in der CPU-Last, ohne im `gens=1`-Ressourcenlauf noch klar schneller zu werden
  - bei `pop=10000` steigt die CPU-Last zusammen mit der Workerzahl sinnvoll an und korreliert deutlich besser mit dem Speedup
- Das spricht dafür, dass kleine Populationen im ersten Generationenlauf noch relativ stark von Setup-/Orchestrierungsanteilen dominiert werden.

### 7.7 Interpretation RAM

- Der Main-Prozess skaliert mit der Population sichtbar:
  - sequential `1000`: 161.4 MB
  - sequential `10000`: 290.2 MB
- Der große Parallel-RAM-Block sitzt aber in den **Worker-Prozessen**.
- Auffällig ist die fast lineare Child-RSS-Zunahme mit der Workerzahl:
  - `pop=1000`: 293 MB → 581 MB → 1.13 GB
  - `pop=10000`: 319 MB → 623 MB → 1.18 GB

**Das heißt praktisch:**
- Ein großer Teil des Parallel-RAM ist fixer Worker-/Interpreter-/Import-/Pool-State.
- Die Populationserhöhung `1000 -> 10000` erhöht den Worker-RAM nur moderat.
- Der Speicheraufwand der Workerzahl ist damit für die aktuelle Architektur der wichtigere Hebel als die reine Populationsgröße.

### 7.8 Robustheitsbeobachtung

Im Ressourcenlauf `pop=10000`, `parallel(4w)` trat einmal ein `SympyError`
aus der bekannten `Ifte`/`Piecewise`-Pfadologie auf (`random_new`, `task_index=6584`).

Wichtig dabei:
- der Fehler wurde **sauber abgefangen**,
- mit Debug-Kontext ausgegeben,
- und der Lauf wurde **nicht** blockiert oder zum Hänger.

Das bestätigt, dass die neu eingebaute Fehlerdiagnostik in `parallel.py`
und `trees.py` ihren Zweck erfüllt.

---

## 8. Einflussfaktoren, priorisiert

### Größter positiver Einfluss
1. **Pre-selection statt Legacy Population IPC**
2. **Shared Memory für `df_train`**
3. **Chunked Batching statt Mini-Tasks oder Riesenchunks**
4. **8 Worker auf 8 physischen Kernen**

### Größte verbleibende Bremsen
1. **`gen_create_initial()` bleibt sequentiell**
2. **Pre-selection läuft im Main-Prozess und kostet ~200 ms pro Generation (`pop=1000`)**
3. **Taskzeiten bleiben kurz (~3.9 ms)**
4. **RAM-Overhead der Worker ist hoch (~0.29 GB bei `2w`, ~0.58–0.62 GB bei `4w`, ~1.13–1.18 GB bei `8w`)**
5. **System-CPU wird nicht vollständig ideal ausgenutzt**

---

## 9. Praktische Schlussfolgerungen für die aktuelle Architektur

### Was man aus den finalen Zahlen sicher sagen kann

- Die aktuelle Parallelisierung ist **klar erfolgreich**.
- Der frühere Zustand „Parallelisierung lohnt kaum“ gilt **nicht mehr**.
- Auf diesem System ist für die aktuelle Architektur **`parallel(8w)`** die beste Konfiguration.
- Größere Populationen verbessern die Parallel-Effizienz.
- Die Batch-Zone **32..128** ist die derzeit sinnvollste Granularität.

### Was CPU/RAM über den nächsten Hebel sagen

- **CPU-seitig** ist die Auslastung erhöht, aber nicht ideal → es gibt noch Scheduling-/Orchestrierungsverluste.
- **RAM-seitig** ist der größte Sprung nicht `1000 -> 10000`, sondern `sequential -> parallel(8w)`.
- Damit ist der wichtigste Ressourcenhebel aktuell eher:
  - Worker-State verkleinern
  - oder Worker-Zahl/Batching bewusst gegen RAM budgetieren

### Konkrete Empfehlungen

1. **Default-Batching nicht auf „ein Batch pro Worker“ zurückdrehen.**
2. **`parallel(8w)` als bevorzugte Benchmark-Konfiguration auf diesem Host betrachten.**
3. **Bei RAM-sensitiven Maschinen einen zweiten Preset-Pfad dokumentieren**, z. B. `parallel(4w)`.
4. **Wenn weiter optimiert wird, dann zuerst dort:**
   - Initialpopulation parallelisieren oder amortisieren
   - Pre-selection effizienter machen
   - Worker-State/RAM reduzieren

---

## 10. Offene Aufgaben

- [ ] **Parallelisierung weiter optimieren:** speziell Worker-RAM und Main-Process-Orchestrierung (`gen_create_initial()`, Pre-selection, Worker-State) erneut untersuchen. Die aktuellen Messungen zeigen, dass hier wahrscheinlich noch Potenzial steckt.
- [ ] **Ressourcenprofiling mit `gens>=2` wiederholen**, um CPU-/RAM-Kreuztabellen nicht nur für den ersten Generationslauf, sondern auch für steady-state zu bekommen.
- [ ] **Batch-Sweet-Spot unter RAM-Budget testen**, z. B. `32`, `64`, `128` gegen `2w/4w/8w`, um einen besseren Perf/RAM-Preset abzuleiten.

---

## 11. Relevante Dateien

- Benchmark-Gesamtlauf: `plagih/test/benchmarks/bench_diagnose_full.py`
- Finaler Output: `plagih/test/benchmarks/bench_output.txt`
- Ressourcen-Profiler: `plagih/test/benchmarks/bench_parallel_resources.py`
- Ressourcen-Output: `plagih/test/benchmarks/bench_resources_output.txt`
- Runtime-Parallellogik: `plagih/parallel.py`
- Pool-/GP-Lifecycle: `plagih/trees.py`


