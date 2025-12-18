# Sistema di Puntamento Stereo (C + Python)

Questo progetto implementa un sistema di visione stereoscopica ibrido. Il "core" computazionale per l'elaborazione delle immagini è scritto in **C** per massimizzare le performance, mentre l'interfaccia, l'acquisizione video e i calcoli fisici (velocità, accelerazione) sono gestiti in **Python**.

## Struttura

- `src/stereo_core.c`: Libreria C che analizza i pixel, trova il centroide dell'oggetto scuro su sfondo chiaro e calcola la disparità.
- `main.py`: Script Python che carica la libreria C, simula (o acquisisce) i flussi video e calcola la fisica del movimento.
- `Makefile`: Script per compilare la libreria condivisa.

## Prerequisiti

- GCC
- Python 3
- OpenCV per Python (`pip install opencv-python numpy`)

## Compilazione

Prima di eseguire il programma, è necessario compilare il codice C in una libreria condivisa (`.so`).

```bash
make
```

Questo genererà il file `libstereo.so`.

## Esecuzione

Esegui lo script Python:

```bash
python3 main.py
```

## Funzionamento

1. **Simulazione**: Attualmente il codice usa una classe `MockCamera` che genera due immagini bianche con un pallino nero che si muove, simulando un oggetto che si avvicina e allontana.
2. **Elaborazione C**: Python passa i puntatori dei buffer raw delle immagini alla funzione C `process_stereo_frame`.
3. **Triangolazione**: Il C calcola la distanza $Z$ usando la formula:
   $$ Z = \frac{f \cdot b}{d} $$
   Dove $b = 2.0m$ (baseline) e $d$ è la disparità in pixel.
4. **Fisica**: Python deriva velocità e accelerazione basandosi sulla variazione della distanza nel tempo.

## Adattamento a Telecamere Reali

Per usare telecamere reali, modifica `main.py`:

1. Sostituisci `MockCamera` con `cv2.VideoCapture(0)` e `cv2.VideoCapture(1)`.
2. Assicurati di calibrare `FOCAL_LENGTH` in base alle tue lenti specifiche.