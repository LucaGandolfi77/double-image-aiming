import ctypes
import cv2
import numpy as np
import time
import os

# --- Configurazione ---
LIB_PATH = "./libstereo.so"
WIDTH = 640
HEIGHT = 480
BASELINE = 2.0  # Metri
FOCAL_LENGTH = 800.0 # Pixel (valore ipotetico, va calibrato nella realtà)
THRESHOLD = 100 # Soglia luminosità (0-255). Sotto = oggetto, Sopra = cielo

# --- Definizione Struttura C ---
class StereoResult(ctypes.Structure):
    _fields_ = [
        ("distance", ctypes.c_double),
        ("disparity", ctypes.c_double),
        ("object_found", ctypes.c_int),
        ("left_x", ctypes.c_int),
        ("left_y", ctypes.c_int),
        ("right_x", ctypes.c_int),
        ("right_y", ctypes.c_int)
    ]

# --- Caricamento Libreria C ---
def load_c_lib():
    if not os.path.exists(LIB_PATH):
        print(f"Errore: {LIB_PATH} non trovato. Esegui 'make' prima.")
        exit(1)
    
    lib = ctypes.CDLL(LIB_PATH)
    
    # Definizione argomenti funzione
    lib.process_stereo_frame.argtypes = [
        ctypes.POINTER(ctypes.c_ubyte), # img_left
        ctypes.POINTER(ctypes.c_ubyte), # img_right
        ctypes.c_int,                   # width
        ctypes.c_int,                   # height
        ctypes.c_double,                # baseline
        ctypes.c_double,                # focal_length
        ctypes.c_int,                   # threshold
        ctypes.POINTER(StereoResult)    # result struct
    ]
    return lib

# --- Classe Fisica ---
class PhysicsTracker:
    def __init__(self):
        self.prev_distance = None
        self.prev_velocity = 0.0
        self.prev_time = None
        
        self.velocity = 0.0      # m/s
        self.acceleration = 0.0  # m/s^2

    def update(self, current_distance):
        current_time = time.time()
        
        if self.prev_distance is None:
            self.prev_distance = current_distance
            self.prev_time = current_time
            return

        dt = current_time - self.prev_time
        if dt <= 0: return

        # Calcolo Velocità: v = (d2 - d1) / t
        # Nota: Se l'oggetto si avvicina, la distanza diminuisce, velocità negativa.
        # Qui calcoliamo la velocità scalare di avvicinamento/allontanamento
        new_velocity = (current_distance - self.prev_distance) / dt
        
        # Calcolo Accelerazione: a = (v2 - v1) / t
        self.acceleration = (new_velocity - self.prev_velocity) / dt
        self.velocity = new_velocity

        # Aggiorna stato precedente
        self.prev_distance = current_distance
        self.prev_velocity = new_velocity
        self.prev_time = current_time

# --- Generatore Simulazione (Mock) ---
# Crea due immagini con un pallino nero che si sposta per simulare avvicinamento
class MockCamera:
    def __init__(self):
        self.frame_count = 0
        
    def get_frames(self):
        # Crea sfondo bianco (cielo)
        img_l = np.ones((HEIGHT, WIDTH), dtype=np.uint8) * 255
        img_r = np.ones((HEIGHT, WIDTH), dtype=np.uint8) * 255
        
        # Simula oggetto che si avvicina (disparità aumenta)
        # Frame 0: Disparità 10px -> Lontano
        # Frame 100: Disparità 100px -> Vicino
        
        base_x = WIDTH // 2
        y = HEIGHT // 2
        
        # Simuliamo un movimento sinusoidale
        shift = abs(np.sin(self.frame_count * 0.05)) * 50 + 10 
        
        # Disparità = shift * 2 (un po' a sx, un po' a dx)
        lx = int(base_x + shift)
        rx = int(base_x - shift)
        
        # Disegna oggetto (cerchio nero)
        cv2.circle(img_l, (lx, y), 20, (50), -1) # Grigio scuro
        cv2.circle(img_r, (rx, y), 20, (50), -1)
        
        self.frame_count += 1
        time.sleep(0.05) # Simula 20fps
        
        return img_l, img_r

# --- Main ---
def main():
    lib = load_c_lib()
    tracker = PhysicsTracker()
    
    # Usa MockCamera per testare senza hardware. 
    # Per usare vere webcam: capL = cv2.VideoCapture(0), capR = cv2.VideoCapture(1)
    camera = MockCamera() 
    
    print("Avvio tracciamento stereo...")
    print(f"Baseline: {BASELINE}m")
    
    try:
        while True:
            # 1. Acquisizione
            frame_l, frame_r = camera.get_frames()
            
            # Assicurarsi che i dati siano C-contiguous (importante per ctypes)
            if not frame_l.flags['C_CONTIGUOUS']: frame_l = np.ascontiguousarray(frame_l)
            if not frame_r.flags['C_CONTIGUOUS']: frame_r = np.ascontiguousarray(frame_r)

            # 2. Preparazione dati per C
            c_res = StereoResult()
            p_frame_l = frame_l.ctypes.data_as(ctypes.POINTER(ctypes.c_ubyte))
            p_frame_r = frame_r.ctypes.data_as(ctypes.POINTER(ctypes.c_ubyte))
            
            # 3. Chiamata alla libreria C (Heavy lifting)
            lib.process_stereo_frame(
                p_frame_l, p_frame_r, 
                WIDTH, HEIGHT, 
                BASELINE, FOCAL_LENGTH, 
                THRESHOLD, 
                ctypes.byref(c_res)
            )
            
            # 4. Elaborazione Fisica (Python)
            if c_res.object_found:
                tracker.update(c_res.distance)
                
                # Output Console
                print(f"Dist: {c_res.distance:.2f}m | "
                      f"Vel: {tracker.velocity:.2f}m/s | "
                      f"Acc: {tracker.acceleration:.2f}m/s^2 | "
                      f"Disp: {c_res.disparity:.1f}px")
                
                # Visualizzazione (Disegna info sul frame sinistro)
                display_img = cv2.cvtColor(frame_l, cv2.COLOR_GRAY2BGR)
                cv2.putText(display_img, f"Dist: {c_res.distance:.2f}m", (10, 30), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                cv2.circle(display_img, (c_res.left_x, c_res.left_y), 5, (0, 255, 0), 2)
                
                # Nota: In un ambiente headless (senza monitor), cv2.imshow potrebbe fallire.
                # Commentare le righe sotto se si esegue su server remoto.
                # cv2.imshow("Stereo Tracker (Left)", display_img)
                # if cv2.waitKey(1) & 0xFF == ord('q'):
                #    break
            else:
                print("Oggetto non trovato.")

    except KeyboardInterrupt:
        print("\nArresto...")
    
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
