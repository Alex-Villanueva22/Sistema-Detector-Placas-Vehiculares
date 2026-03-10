# IMPORTAR LIBRERÍAS NECESARIAS 
import cv2
import numpy as np
import pytesseract
import argparse
import re
import os

# CONFIGURACIÓN
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

class DetectorPlacas:
    
    def __init__(self, video_path):
        self.video_path = video_path
        self.matriculas_detectadas = []
        
    def validar_placa(self, texto):
        """Valida formato ABC-123"""
        texto = texto.strip().upper()
        texto = re.sub(r'[^A-Z0-9-]', '', texto)
        texto = texto.replace('O', '0').replace('I', '1').replace('S', '5')
        
        if len(texto) >= 6:
            if re.match(r'^[A-Z]{3}\d{3}$', texto):
                return texto[:3] + '-' + texto[3:]
            elif re.match(r'^[A-Z]{3}-\d{3}$', texto):
                return texto
        
        return None
    
    def preprocesar_placa(self, roi):
        """Preprocesa la región de la placa"""
        h, w = roi.shape[:2]
        
        # Redimensionar para mejor OCR
        if h < 50:
            scale = 50 / h
            roi = cv2.resize(roi, (int(w * scale), 50))
        
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        
        # Método 1: Otsu
        _, thresh1 = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Método 2: Threshold alto (para placas muy blancas)
        _, thresh2 = cv2.threshold(gray, 160, 255, cv2.THRESH_BINARY)
        
        # Método 3: Adaptativo
        thresh3 = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                        cv2.THRESH_BINARY, 11, 2)
        
        return [thresh1, thresh2, thresh3]
    
    def detectar_placas(self, frame):
        """Detecta placas usando múltiples métodos combinados"""
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        h, w = frame.shape[:2]
        
        candidatos = []
        
        # MÉTODO 1: Detección por regiones blancas
        _, thresh_white = cv2.threshold(gray, 170, 255, cv2.THRESH_BINARY)
        
        # Dilatar horizontalmente (las placas son horizontales)
        kernel_h = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 5))
        dilated = cv2.dilate(thresh_white, kernel_h, iterations=1)
        
        contours1, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for cnt in contours1:
            x, y, w, h = cv2.boundingRect(cnt)
            
            # Filtros muy permisivos
            if w < 70 or h < 18:
                continue
            if w > 600 or h > 200:
                continue
            
            aspect = w / float(h)
            if aspect < 1.8 or aspect > 7.0:
                continue
            
            candidatos.append((x, y, w, h, 'white'))
        
        # MÉTODO 2: Detección por bordes
        edges = cv2.Canny(gray, 50, 150)
        
        kernel_rect = cv2.getStructuringElement(cv2.MORPH_RECT, (21, 5))
        closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel_rect)
        
        contours2, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for cnt in contours2:
            x, y, w, h = cv2.boundingRect(cnt)
            
            if w < 70 or h < 18:
                continue
            if w > 600 or h > 200:
                continue
            
            aspect = w / float(h)
            if aspect < 1.8 or aspect > 7.0:
                continue
            
            candidatos.append((x, y, w, h, 'edge'))
        
        # MÉTODO 3: Sobel horizontal (detecta líneas de texto)
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobelx = np.absolute(sobelx)
        sobelx = np.uint8(sobelx)
        
        _, sobel_thresh = cv2.threshold(sobelx, 50, 255, cv2.THRESH_BINARY)
        
        kernel_sobel = cv2.getStructuringElement(cv2.MORPH_RECT, (17, 5))
        sobel_closed = cv2.morphologyEx(sobel_thresh, cv2.MORPH_CLOSE, kernel_sobel)
        
        contours3, _ = cv2.findContours(sobel_closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for cnt in contours3:
            x, y, w, h = cv2.boundingRect(cnt)
            
            if w < 70 or h < 18:
                continue
            if w > 600 or h > 200:
                continue
            
            aspect = w / float(h)
            if aspect < 1.8 or aspect > 7.0:
                continue
            
            candidatos.append((x, y, w, h, 'sobel'))
        
        # Eliminar duplicados
        candidatos_unicos = self.eliminar_duplicados(candidatos)
        
        return candidatos_unicos
    
    def eliminar_duplicados(self, candidatos):
        """Elimina candidatos que se solapan"""
        if not candidatos:
            return []
        
        unicos = []
        
        for cand in candidatos:
            x1, y1, w1, h1 = cand[:4]
            duplicado = False
            
            for unico in unicos:
                x2, y2, w2, h2 = unico[:4]
                
                # Calcular intersección
                x_left = max(x1, x2)
                y_top = max(y1, y2)
                x_right = min(x1 + w1, x2 + w2)
                y_bottom = min(y1 + h1, y2 + h2)
                
                if x_right > x_left and y_bottom > y_top:
                    overlap = (x_right - x_left) * (y_bottom - y_top)
                    area1 = w1 * h1
                    area2 = w2 * h2
                    
                    if overlap > 0.5 * min(area1, area2):
                        duplicado = True
                        break
            
            if not duplicado:
                unicos.append(cand)
        
        return unicos
    
    def ocr_placa(self, roi):
        """Aplica OCR - VERSION OPTIMIZADA"""
        versiones = self.preprocesar_placa(roi)
        
        # OPTIMIZACIÓN: Probar solo los 2 mejores métodos
        configs = [
            '--psm 7 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789',
            '--psm 8 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789',
        ]
        
        # Probar primero el método más efectivo (Otsu)
        for config in configs:
            try:
                texto = pytesseract.image_to_string(versiones[0], config=config)
                placa = self.validar_placa(texto)
                if placa:
                    return placa
            except:
                pass
        
        # Si falló con Otsu, probar con threshold alto
        try:
            texto = pytesseract.image_to_string(versiones[1], config=configs[0])
            placa = self.validar_placa(texto)
            if placa:
                return placa
        except:
            pass
        
        return None
    
    def procesar_video(self):
        """Procesa el video - OPTIMIZADO"""
        cap = cv2.VideoCapture(self.video_path)
        
        if not cap.isOpened():
            print(f"❌ No se pudo abrir el video")
            return False
        
        fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        print(f"📹 Video: {width}x{height} @ {fps} FPS")
        print(f"⏱️  Total: {total_frames} frames")
        print(f"⚡ MODO RÁPIDO: Procesando cada 3 frames\n")
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter('video_salida.mp4', fourcc, fps, (width, height))
        
        detecciones = {}
        frame_num = 0
        frames_procesados = 0
        
        # Cache para mantener detecciones entre frames
        ultimo_frame_con_placas = {}
        
        cv2.namedWindow('Detector Placas Peru', cv2.WINDOW_NORMAL)
        
        import time
        tiempo_inicio = time.time()
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_num += 1
            tiempo = frame_num / fps
            frame_output = frame.copy()
            
            # OPTIMIZACIÓN 1: Procesar solo cada 3 frames (pero escribir todos)
            if frame_num % 3 == 0:
                frames_procesados += 1
                
                # Detectar autos candidatos
                candidatos = self.detectar_placas(frame)
                
                if frame_num % 30 == 0:  # Mostrar progreso cada 30 frames
                    elapsed = time.time() - tiempo_inicio
                    fps_proceso = frames_procesados / elapsed if elapsed > 0 else 0
                    print(f"Frame {frame_num}/{total_frames} | {len(candidatos)} candidatos | {fps_proceso:.1f} fps")
                
                # OPTIMIZACIÓN 2: Solo hacer OCR a candidatos en tercio inferior
                candidatos_filtrados = [c for c in candidatos if c[1] > height * 0.3]
                
                # Limpiar cache anterior
                ultimo_frame_con_placas.clear()
                
                # Procesar candidatos
                for item in candidatos_filtrados:
                    x, y, w, h = item[:4]
                    metodo = item[4] if len(item) > 4 else 'unknown'
                    
                    margin = 5
                    y1 = max(0, y - margin)
                    y2 = min(frame.shape[0], y + h + margin)
                    x1 = max(0, x - margin)
                    x2 = min(frame.shape[1], x + w + margin)
                    
                    roi = frame[y1:y2, x1:x2]
                    
                    texto = self.ocr_placa(roi)
                    
                    if texto:
                        # Placa Detectada
                        ultimo_frame_con_placas[texto] = (x, y, w, h)
                        
                        if texto not in detecciones:
                            detecciones[texto] = []
                        detecciones[texto].append((frame_num, tiempo))
                        
                        print(f"  ✅ {texto}")
                    else:
                        # Candidato
                        cv2.rectangle(frame_output, (x, y), (x+w, y+h), (0, 255, 255), 1)
            
            # Dibujar placas detectadas (usar cache si no se procesó este frame)
            for texto, (x, y, w, h) in ultimo_frame_con_placas.items():
                cv2.rectangle(frame_output, (x, y), (x+w, y+h), (0, 255, 0), 4)
                cv2.rectangle(frame_output, (x, y-35), (x+220, y), (0, 255, 0), -1)
                cv2.putText(frame_output, texto, (x+5, y-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 2)
            
            # Panel de información
            progreso = (frame_num / total_frames) * 100
            cv2.rectangle(frame_output, (10, 10), (550, 60), (0, 0, 0), -1)
            cv2.putText(frame_output, f"Frame: {frame_num}/{total_frames} ({progreso:.1f}%)", 
                       (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            cv2.putText(frame_output, f"Placas detectadas: {len(detecciones)}", 
                       (15, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
            out.write(frame_output)
            cv2.imshow('Detector Placas Peru', frame_output)
            
            if cv2.waitKey(1) & 0xFF == 27:
                break
        
        cap.release()
        out.release()
        cv2.destroyAllWindows()
        
        tiempo_total = time.time() - tiempo_inicio
        print(f"\n⏱️  Tiempo total: {tiempo_total:.1f}s")
        print(f"⚡ Velocidad: {frames_procesados/tiempo_total:.1f} fps procesados")
        
        # Consolidar
        self.matriculas_detectadas = []
        for texto, frames in detecciones.items():
            frame_num, tiempo = frames[0]
            self.matriculas_detectadas.append((texto, frame_num, tiempo))
        
        self.matriculas_detectadas.sort(key=lambda x: x[2])
        
        print(f"🎯 Placas únicas: {len(self.matriculas_detectadas)}")
        return True
    
    def guardar_resultados(self):
        """Guarda los resultados"""
        with open('matriculas_detectadas.txt', 'w', encoding='utf-8') as f:
            f.write("="*70 + "\n")
            f.write("MATRÍCULAS DETECTADAS - ESTACIONAMIENTO FACULTAD\n")
            f.write("="*70 + "\n\n")
            
            if self.matriculas_detectadas:
                for i, (texto, frame, tiempo) in enumerate(self.matriculas_detectadas, 1):
                    f.write(f"{i}. [{texto}] – Detectada en el segundo {tiempo:.2f} (Frame {frame})\n")
            else:
                f.write("No se detectaron matrículas válidas\n")
            
            f.write(f"\n{'='*70}\n")
            f.write(f"TOTAL: {len(self.matriculas_detectadas)} matrículas\n")
            f.write("="*70 + "\n")
        
        print("\n📄 Archivo generado: matriculas_detectadas.txt")
    
    def mostrar_resumen(self):
        """Muestra resumen en consola"""
        print("\n" + "="*70)
        print("RESUMEN FINAL")
        print("="*70 + "\n")
        
        if self.matriculas_detectadas:
            print(f"✅ Total de placas detectadas: {len(self.matriculas_detectadas)}\n")
            for i, (texto, _, tiempo) in enumerate(self.matriculas_detectadas, 1):
                print(f"  {i}. {texto} ({tiempo:.2f}s)")
        else:
            print("❌ No se detectaron placas válidas")
        
        print("\n" + "="*70 + "\n")


def main():
    parser = argparse.ArgumentParser(description='Detector de placas peruanas')
    parser.add_argument('--video', required=True, help='Video de entrada')
    args = parser.parse_args()
    
    if not os.path.exists(args.video):
        print(f"❌ El archivo '{args.video}' no existe")
        return
    
    print("\n" + "="*70)
    print("DETECTOR DE MATRÍCULAS PERUANAS")
    print("Métodos: Regiones blancas + Bordes + Sobel")
    print("="*70 + "\n")
    
    detector = DetectorPlacas(args.video)
    
    if detector.procesar_video():
        detector.guardar_resultados()
        detector.mostrar_resumen()
        print("📁 Archivos generados:")
        print("  ✓ matriculas_detectadas.txt")
        print("  ✓ video_salida.mp4\n")


if __name__ == "__main__":
    main()