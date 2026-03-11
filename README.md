# 🚗 Sistema Detector de Placas Vehiculares (Classic Computer Vision)

> Un sistema de reconocimiento de matrículas vehiculares (ALPR) desarrollado exclusivamente con técnicas clásicas de procesamiento de imágenes, logrando alta precisión sin el uso de Deep Learning.

---

## 🎯 El Desafío
El objetivo principal fue demostrar la eficacia de los métodos tradicionales de visión computacional frente a problemas complejos de seguridad ciudadana.
* **Problema:** Detectar y leer placas en video bajo condiciones variables de iluminación y perspectiva.
* **Solución:** Implementación de un pipeline de procesamiento multicanal que utiliza detección por bordes, gradientes y regiones para maximizar la tasa de éxito.

## 🛠️ Tecnologías Utilizadas
* **Lenguaje:** Python 3.x
* **Visión Computacional:** OpenCV (`cv2`)
* **Procesamiento Numérico:** NumPy
* **OCR:** Tesseract (pytesseract)
* **Gestión de Archivos:** OS, Argparse, Re (RegEx)

## 🔬 Metodología Técnica (Pipeline de Detección)

El sistema ejecuta tres métodos en paralelo para localizar las regiones candidatas (ROI):
1. **Detección por Regiones Blancas:** Identificación de áreas con alta densidad de color claro.
2. **Detección por Bordes (Canny):** Localización de formas rectangulares basadas en discontinuidades de intensidad.
3. **Gradientes Horizontales (Sobel):** Resalte de la textura vertical característica de los caracteres de la placa.

### Optimización del OCR
Cada región detectada se somete a tres tipos de umbralización antes de pasar por Tesseract para asegurar la lectura:
* **Otsu Thresholding:** Cálculo automático del umbral óptimo.
* **High Threshold (160):** Optimizado para placas con alta reflectancia (muy blancas).
* **Adaptive Thresholding:** Ajuste local según variaciones de brillo.

## ✨ Características Principales
* ✅ **Procesamiento en Tiempo Real:** Genera un video de salida con los cuadros delimitadores (bounding boxes).
* ✅ **Filtro de Ruido:** Uso de expresiones regulares (RegEx) para validar el formato de las placas detectadas.
* ✅ **Registro Automático:** Exportación de todas las matrículas reconocidas a un archivo de texto (.txt).
* ✅ **Zero Neural Networks:** Funcionamiento basado 100% en lógica de visión computacional clásica.
