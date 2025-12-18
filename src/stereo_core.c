#include <stdio.h>
#include <stdlib.h>
#include <math.h>

// Struttura per restituire i risultati al wrapper
typedef struct {
    double distance;      // Distanza in metri
    double disparity;     // Disparità in pixel
    int object_found;     // 1 se l'oggetto è stato trovato in entrambi i frame, 0 altrimenti
    int left_x;           // Coordinata X centroide sinistra
    int left_y;           // Coordinata Y centroide sinistra
    int right_x;          // Coordinata X centroide destra
    int right_y;          // Coordinata Y centroide destra
} StereoResult;

// Funzione helper per trovare il centroide di un oggetto scuro su sfondo chiaro
// Assumiamo input in scala di grigi (1 byte per pixel) per semplicità e velocità
int find_centroid(const unsigned char* img, int width, int height, int threshold, int* out_x, int* out_y) {
    long sum_x = 0;
    long sum_y = 0;
    long count = 0;

    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            // L'immagine è un array 1D. Indice = y * width + x
            unsigned char pixel_val = img[y * width + x];

            // Logica di soglia:
            // Se il pixel è PIÙ SCURO della soglia, è parte dell'oggetto (cielo è chiaro)
            if (pixel_val < threshold) {
                sum_x += x;
                sum_y += y;
                count++;
            }
        }
    }

    if (count > 0) {
        *out_x = (int)(sum_x / count);
        *out_y = (int)(sum_y / count);
        return 1; // Trovato
    }

    return 0; // Non trovato
}

// Funzione principale esportata
// img_left, img_right: puntatori ai buffer raw delle immagini (grayscale)
// width, height: dimensioni immagini
// baseline: distanza tra le camere in metri (es. 2.0)
// focal_length: lunghezza focale in pixel (dipende dalla camera e risoluzione)
// threshold: valore 0-255 per distinguere oggetto da cielo
void process_stereo_frame(
    const unsigned char* img_left, 
    const unsigned char* img_right, 
    int width, 
    int height, 
    double baseline, 
    double focal_length, 
    int threshold,
    StereoResult* result
) {
    int lx, ly, rx, ry;
    
    // 1. Trova l'oggetto nell'immagine sinistra
    int found_l = find_centroid(img_left, width, height, threshold, &lx, &ly);
    
    // 2. Trova l'oggetto nell'immagine destra
    int found_r = find_centroid(img_right, width, height, threshold, &rx, &ry);

    if (found_l && found_r) {
        result->object_found = 1;
        result->left_x = lx;
        result->left_y = ly;
        result->right_x = rx;
        result->right_y = ry;

        // 3. Calcola Disparità
        // Disparità = (X_sinistra - X_destra)
        // Nota: Assumiamo che le camere siano rettificate. 
        // Se l'oggetto è all'infinito, disparità è 0. Più è vicino, più è alta.
        double disparity = (double)(lx - rx);

        // Gestione casi limite (disparità negativa o zero)
        if (disparity <= 0.1) {
            disparity = 0.1; // Evita divisione per zero, oggetto molto lontano
        }
        
        result->disparity = disparity;

        // 4. Calcola Distanza (Triangolazione)
        // Z = (f * b) / d
        result->distance = (focal_length * baseline) / disparity;

    } else {
        result->object_found = 0;
        result->distance = -1.0;
        result->disparity = 0.0;
    }
}
