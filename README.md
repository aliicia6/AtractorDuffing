# AtractorDuffing

Simulación, comparación y visualización animada de dos atractores de Duffing mediante Python. El programa integra numéricamente el sistema con Runge-Kutta de cuarto orden (RK4), calcula una estimación acumulada del mayor exponente de Lyapunov y muestra la evolución de ambas trayectorias en el espacio de fases.

Además, permite exportar puntos discretos de las trayectorias a archivos JSON compatibles con la aplicación **Fractal Music**, filtrando los puntos mediante el conjunto de Mandelbrot.

## Características

- Simulación simultánea de dos sistemas de Duffing.
- Integración numérica mediante RK4.
- Comparación visual de las trayectorias en el espacio de fases `(x, y)`.
- Cálculo acumulado del mayor exponente de Lyapunov mediante el método de Benettin.
- Detección aproximada de una separación dinámica entre las dos trayectorias.
- Animación interactiva con controles para mostrar u ocultar:
  - Información de la simulación.
  - Atractor 1.
  - Atractor 2.
  - Línea amarilla de comparación.
  - Fondo del conjunto de Mandelbrot.
- Generación opcional de un archivo GIF.
- Exportación de los puntos válidos a archivos JSON compatibles con **Fractal Music**.
- Compresión configurable de los puntos exportados.

## Requisitos

- Python 3.9 o superior.
- NumPy.
- Matplotlib.
- Pillow, únicamente si se desea guardar la animación como GIF.

Instala las dependencias con:

```bash
pip install numpy matplotlib pillow
```

## Uso

Ejecuta el programa desde la raíz del repositorio:

```bash
python duffing_comparacion_animada.py
```

El programa solicitará interactivamente los parámetros comunes y los parámetros de cada atractor.

### Parámetros comunes

- **Número de pasos:** cantidad de iteraciones de la simulación.
- **Paso temporal (`dt`):** tamaño del paso utilizado por el integrador RK4.

### Parámetros de cada atractor

- **`x0`:** posición inicial.
- **`y0`:** velocidad inicial.
- **`delta` (`δ`):** amortiguamiento.
- **`alpha` (`α`):** parámetro lineal del sistema.
- **`beta` (`β`):** coeficiente no lineal cúbico.
- **`f0` (`F₀`):** amplitud de la fuerza externa.
- **`frecuencia` (`ω`):** frecuencia angular de la fuerza externa.
- **Compresión:** conserva un punto de cada `N` puntos al representar y exportar datos discretos.

Los valores predeterminados están definidos en el propio programa. Por ejemplo, el sistema comienza con `delta = 0.2`, `alpha = -1.0`, `beta = 1.0`, `f0 = 0.3` y `frecuencia = 1.0`.

## Guardar la animación como GIF

Para guardar la animación, utiliza la opción `--guardar-gif`:

```bash
python duffing_comparacion_animada.py --guardar-gif comparacion.gif
```

La velocidad de la animación puede configurarse con `--fps`:

```bash
python duffing_comparacion_animada.py --guardar-gif comparacion.gif --fps 30
```

El valor predeterminado es de 30 fotogramas por segundo.

## Exportación para Fractal Music

El programa genera archivos JSON preparados para ser cargados o procesados por la aplicación **Fractal Music**. Para evitar exportar datos innecesarios, cada punto de la trayectoria se valida previamente y solo se incluyen los puntos que pertenecen al conjunto de Mandelbrot.

Por defecto, los archivos JSON se guardan en la misma carpeta que el programa. Puedes cambiar la carpeta de salida con `--carpeta-salida`:

```bash
python duffing_comparacion_animada.py --carpeta-salida resultados
```

Se crean directorios con nombres similares a:

```text
resultados/
├── duffing_atractor_1/
│   ├── duffing_atractor_1_001.json
│   └── ...
└── duffing_atractor_2/
    ├── duffing_atractor_2_001.json
    └── ...
```

Los JSON utilizan la estructura de sesión visual esperada por **Fractal Music** e incluyen:

- Una lista de `cases` con los puntos exportados.
- Un identificador único para cada caso.
- Las coordenadas reales e imaginarias del parámetro complejo `c`.
- La recurrencia `quadratic_complex`, utilizada para representar la dinámica `z² + c`.
- Metadatos de discretización, análisis y visualización.
- La configuración necesaria para trabajar con sucesiones fractales y generar resultados musicales.

Cada archivo se divide en bloques de hasta 80 casos para facilitar su carga y procesamiento en **Fractal Music**. La cantidad de puntos exportados depende de la duración de la simulación, del factor de compresión y de la pertenencia de cada punto al conjunto de Mandelbrot.

## Modelo matemático

El sistema de Duffing utilizado es:

```text
x' = y

y' = F₀ cos(ωt) - δy - αx - βx³
```

La simulación representa la posición `x` y la velocidad `y` en el espacio de fases. Para comparar ambos atractores se calcula la distancia euclídea entre sus estados en cada instante:

```text
d(t) = √((x₁ - x₂)² + (y₁ - y₂)²)
```

La aplicación utiliza como referencia de separación un umbral dinámico igual a `1000 × d(0)`, donde `d(0)` es la distancia inicial entre ambos estados.

## Estructura del repositorio

```text
.
├── duffing_comparacion_animada.py
├── README.md
└── LICENSE
```

## Notas

- La animación puede requerir más tiempo y memoria cuando se utilizan muchos pasos.
- Aunque todos los pasos se integran, la animación limita el número de fotogramas a 1200 para mantener una visualización manejable.
- La estimación del exponente de Lyapunov mejora cuando la simulación es suficientemente larga y el paso temporal es adecuado.
- Los valores de los parámetros pueden producir comportamientos periódicos, cuasiperiódicos o caóticos.
- Los archivos JSON están pensados para su uso con **Fractal Music**; la aplicación puede aplicar sus propios análisis, visualizaciones y procesos de sonificación sobre los datos importados.

## Licencia

Consulta el archivo [LICENSE](LICENSE) para conocer los términos de uso y distribución del proyecto.
