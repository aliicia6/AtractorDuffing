"""Muestra cómo se forman simultáneamente dos atractores de Duffing."""

import argparse
import json
from pathlib import Path

import matplotlib.animation as animation
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.widgets import Button
import numpy as np


TAMANO_BLOQUE = 80
MAX_ITERACIONES_MANDELBROT = 1000


def pedir_valor(mensaje, tipo, defecto):
    while True:
        respuesta = input(f"{mensaje} [{defecto}]: ").strip()
        if not respuesta:
            return defecto
        try:
            return tipo(respuesta)
        except ValueError:
            print("Valor no válido. Inténtalo de nuevo.")


def pedir_si_no(mensaje, defecto=True):
    respuesta_defecto = "s" if defecto else "n"
    while True:
        respuesta = input(f"{mensaje} (s/n) [{respuesta_defecto}]: ").strip().lower()
        if not respuesta:
            return defecto
        if respuesta in {"s", "si", "sí"}:
            return True
        if respuesta in {"n", "no"}:
            return False
        print("Responde s o n.")


def integrar_duffing(pasos, dt, p):
    estado = np.array([p["x0"], p["y0"]], dtype=float)
    trayectoria = []
    for paso in range(pasos):
        t = paso * dt
        trayectoria.append((t, *estado))
        def f(ti, s):
            x, y = s
            return np.array([y, p["f0"] * np.cos(p["frecuencia"] * ti) - p["delta"] * y - p["alpha"] * x - p["beta"] * x**3])
        k1 = f(t, estado); k2 = f(t + dt / 2, estado + dt * k1 / 2)
        k3 = f(t + dt / 2, estado + dt * k2 / 2); k4 = f(t + dt, estado + dt * k3)
        estado += dt * (k1 + 2 * k2 + 2 * k3 + k4) / 6
    return np.asarray(trayectoria)


def calcular_lyapunov(trayectoria, dt, p):
    """Calcula el mayor exponente de Lyapunov acumulado mediante Benettin.

    La trayectoria principal y la perturbación tangente se integran con RK4.
    La perturbación se renormaliza en cada paso y se acumulan sus factores de
    crecimiento. El resultado converge al mayor exponente de Lyapunov cuando
    la simulación es suficientemente larga.
    """
    perturbacion = np.array([1.0, 0.0])
    suma_logaritmos = 0.0
    exponentes = np.zeros(len(trayectoria))
    for indice in range(1, len(trayectoria)):
        t = trayectoria[indice - 1, 0]
        estado = trayectoria[indice - 1, 1:3]

        def derivada_estado(ti, s):
            x, y = s
            return np.array([
                y,
                p["f0"] * np.cos(p["frecuencia"] * ti)
                - p["delta"] * y
                - p["alpha"] * x
                - p["beta"] * x**3,
            ])

        def derivada_tangente(s, vector):
            x = s[0]
            dx, dy = vector
            return np.array([
                dy,
                (-p["alpha"] - 3 * p["beta"] * x**2) * dx - p["delta"] * dy,
            ])

        # Etapas RK4 de la trayectoria principal del paso actual.
        k1 = derivada_estado(t, estado)
        k2 = derivada_estado(t + dt / 2, estado + dt * k1 / 2)
        k3 = derivada_estado(t + dt / 2, estado + dt * k2 / 2)
        k4 = derivada_estado(t + dt, estado + dt * k3)
        estado_k2 = estado + dt * k1 / 2
        estado_k3 = estado + dt * k2 / 2
        estado_k4 = estado + dt * k3

        # Etapas RK4 de la ecuación variacional, evaluadas en cada estado
        # intermedio de la trayectoria y no únicamente en el estado inicial.
        q1 = derivada_tangente(estado, perturbacion)
        q2 = derivada_tangente(estado_k2, perturbacion + dt * q1 / 2)
        q3 = derivada_tangente(estado_k3, perturbacion + dt * q2 / 2)
        q4 = derivada_tangente(estado_k4, perturbacion + dt * q3)
        perturbacion += dt * (q1 + 2 * q2 + 2 * q3 + q4) / 6

        norma = np.linalg.norm(perturbacion)
        if norma == 0 or not np.isfinite(norma):
            perturbacion = np.array([1.0, 0.0])
            continue
        suma_logaritmos += np.log(norma)
        exponentes[indice] = suma_logaritmos / (indice * dt)
        perturbacion /= norma
    return exponentes


def pedir_parametros(numero):
    print(f"\n--- Parámetros del atractor {numero} ---")
    return {
        "x0": pedir_valor("Posición inicial (x₀)", float, 0.0),
        "y0": pedir_valor("Velocidad inicial (y₀)", float, 0.0),
        "delta": pedir_valor("Amortiguamiento (δ)", float, 0.2),
        "alpha": pedir_valor("Parámetro (α)", float, -1.0),
        "beta": pedir_valor("Parámetro (β)", float, 1.0),
        "f0": pedir_valor("Fuerza máxima (F₀)", float, 0.3),
        "frecuencia": pedir_valor("Frecuencia angular (ω)", float, 1.0),
        "compresion": pedir_valor("Compresión: 1 de cada N puntos", int, 100),
    }


def pertenece_a_mandelbrot(c):
    z = 0j
    for _ in range(MAX_ITERACIONES_MANDELBROT):
        z = z * z + c
        if abs(z) > 2:
            return False
    return True


def crear_fondo_mandelbrot(resolucion=700):
    eje_x = np.linspace(-2.0, 2.0, resolucion)
    eje_y = np.linspace(-2.0, 2.0, resolucion)
    xx, yy = np.meshgrid(eje_x, eje_y)
    z = np.zeros_like(xx, dtype=complex)
    escape = np.zeros_like(xx, dtype=float)
    activos = np.ones_like(xx, dtype=bool)
    for iteracion in range(1, MAX_ITERACIONES_MANDELBROT + 1):
        z[activos] = z[activos] ** 2 + xx[activos] + 1j * yy[activos]
        nuevos = activos & (np.abs(z) > 2)
        escape[nuevos] = iteracion
        activos[nuevos] = False
        if not activos.any():
            break
    escape[activos] = MAX_ITERACIONES_MANDELBROT
    return escape


def crear_json_atractor(trayectoria, parametros, pasos, dt, numero):
    muestreados = trayectoria[::parametros["compresion"]]
    cases = []
    for t, x, y in muestreados:
        c = complex(float(x), float(y))
        # El punto se valida antes de entrar en el JSON. Los puntos que
        # escapan del conjunto de Mandelbrot se descartan completamente.
        if not pertenece_a_mandelbrot(c):
            continue
        indice = len(cases) + 1
        cases.append({
            "identifier": f"case-{indice:03d}",
            "label": f"Q{indice}: c={c.real:.4f}{c.imag:+.4f}j",
            "metadata": {},
            "recurrence": {"name": "quadratic_complex", "parameters": {
                "c_imag": c.imag, "c_real": c.real, "z0_imag": 0.0, "z0_real": 0.0,
            }},
            "source": "mandelbrot_click",
        })
    configuracion = {
        "analysis": {"attractor_window": None, "cluster_tolerance": None,
                      "divergence_limit": 1_000_000.0, "max_period": 16,
                      "max_terms": pasos, "min_repetitions": 5, "stability_ratio": 0.98,
                      "tolerance": 1e-7, "transient_terms": 100},
        "discretization": {"coordinate_system": "cartesian",
                            "compression_factor": parametros["compresion"],
                            "ranges": {"x": {"maximum": 2.0, "minimum": -2.0, "steps": 16},
                                       "y": {"maximum": 2.0, "minimum": -2.0, "steps": 16}}},
        "recurrence": {"name": "quadratic_complex",
                        "parameters": {"c_imag": 0.0, "c_real": 0.0,
                                        "z0_imag": 0.0, "z0_real": 0.0}},
    }
    return {"cases": cases, "results": [], "schema": "fractal_sequences.visual_session", "version": 1,
            "view": {"active_case_id": None, "config": configuracion,
                     "map_limits": {"logistica": [0.0, 4.0, 0.0, 1.0],
                                     "z^2+c": [-2.0, 1.0, -1.35, 1.35]},
                     "music": {"criterion": "melody", "scale": "minor", "segmentation": "bands"},
                     "table_mode": "sucesion"}}


def guardar_json_atractor(trayectoria, parametros, pasos, dt, numero, carpeta_base):
    carpeta = carpeta_base / f"duffing_atractor_{numero}"
    carpeta.mkdir(parents=True, exist_ok=True)
    # Limpia únicamente los JSON de este atractor antes de exportar la nueva ejecución.
    json_global_anterior = carpeta / f"duffing_atractor_{numero}.json"
    if json_global_anterior.exists():
        json_global_anterior.unlink()
    for archivo_anterior in carpeta.glob(f"duffing_atractor_{numero}_*.json"):
        archivo_anterior.unlink()
    datos = crear_json_atractor(trayectoria, parametros, pasos, dt, numero)
    rutas_bloques = []
    for inicio in range(0, len(datos["cases"]), TAMANO_BLOQUE):
        bloque = datos["cases"][inicio:inicio + TAMANO_BLOQUE]
        numero_bloque = inicio // TAMANO_BLOQUE + 1
        ruta_bloque = carpeta / f"duffing_atractor_{numero}_{numero_bloque:03d}.json"
        with ruta_bloque.open("w", encoding="utf-8") as archivo:
            json.dump({**datos, "cases": bloque}, archivo, indent=2, ensure_ascii=False)
        rutas_bloques.append(ruta_bloque)
    return (rutas_bloques[0] if rutas_bloques else carpeta), len(datos["cases"])


def preparar_ejes(eje, trayectorias):
    todos_x = np.concatenate([trayectoria[:, 1] for trayectoria in trayectorias])
    todos_y = np.concatenate([trayectoria[:, 2] for trayectoria in trayectorias])
    limite = max(np.max(np.abs(todos_x)), np.max(np.abs(todos_y)), 1.0) * 1.08
    eje.set_xlim(-limite, limite)
    eje.set_ylim(-limite, limite)
    eje.set_aspect("equal")
    eje.set_xlabel("x (posición)", color="white")
    eje.set_ylabel("y (velocidad)", color="white")
    eje.set_title("Atractor de Duffing", color="white")
    eje.tick_params(colors="white")
    for borde in eje.spines.values():
        borde.set_color("white")
    eje.grid(True, color="white", alpha=0.2)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--guardar-gif",
        help="Ruta opcional para guardar la animación, por ejemplo comparacion.gif",
    )
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument(
        "--carpeta-salida",
        type=Path,
        default=Path(__file__).resolve().parent,
        help="Carpeta donde se guardan los JSON (por defecto, junto a este programa)",
    )
    args = parser.parse_args()

    mostrar_informacion = True
    mostrar_atractor1 = True
    mostrar_atractor2 = True
    mostrar_comparacion = True
    print("Animación de dos atractores de Duffing")
    pasos = pedir_valor("Número de pasos común", int, 10000)
    dt = pedir_valor("Paso temporal (dt) común", float, 0.01)
    parametros1 = pedir_parametros(1)
    parametros2 = pedir_parametros(2)

    if pasos <= 0 or dt <= 0 or args.fps <= 0:
        parser.error("pasos, dt y fps deben ser positivos")
    if parametros1["compresion"] <= 0 or parametros2["compresion"] <= 0:
        parser.error("la compresión debe ser positiva en ambos atractores")

    trayectoria1 = integrar_duffing(pasos, dt, parametros1)
    trayectoria2 = integrar_duffing(pasos, dt, parametros2)
    trayectorias = (trayectoria1, trayectoria2)
    parametros_animacion = (parametros1, parametros2)
    colores = ("#a6bcc9", "#1e3252")
    nombres = ("Atractor 1", "Atractor 2")
    distancias = np.sqrt(np.sum((trayectoria1[:, 1:3] - trayectoria2[:, 1:3]) ** 2, axis=1))
    lyapunov1 = calcular_lyapunov(trayectoria1, dt, parametros1)
    lyapunov2 = calcular_lyapunov(trayectoria2, dt, parametros2)
    rutas_json = (
        guardar_json_atractor(trayectoria1, parametros1, pasos, dt, 1, args.carpeta_salida),
        guardar_json_atractor(trayectoria2, parametros2, pasos, dt, 2, args.carpeta_salida),
    )
    print(f"JSON del atractor 1: {rutas_json[0][0]}")
    puntos_exportados_1 = rutas_json[0][1]
    print(f"Puntos reales guardados en el JSON del atractor 1: {puntos_exportados_1}")
    print(f"JSON del atractor 2: {rutas_json[1][0]}")
    puntos_exportados_2 = rutas_json[1][1]
    print(f"Puntos reales guardados en el JSON del atractor 2: {puntos_exportados_2}")
    total_puntos_exportados = puntos_exportados_1 + puntos_exportados_2
    print(f"Total real de puntos guardados en los JSON: {total_puntos_exportados}")
    print("JSON creados correctamente")
    distancia_inicial = distancias[0]
    # El umbral depende de la semilla inicial. Se usa la distancia completa
    # del espacio de fases y no solo la coordenada x.
    umbral_ruptura = 1000.0 * distancia_inicial
    if umbral_ruptura == 0:
        umbral_ruptura = np.finfo(float).eps
    indices_rotura = np.flatnonzero(distancias > umbral_ruptura)
    primer_indice_rotura = int(indices_rotura[0]) if len(indices_rotura) else None
    if mostrar_comparacion:
        if primer_indice_rotura is None:
            print(f"No se supera el límite dinámico {umbral_ruptura:g} durante la simulación.")
        else:
            print(
                f"Punto de ruptura: iteración {primer_indice_rotura}, "
                f"t={trayectoria1[primer_indice_rotura, 0]:.6f}, "
                f"distancia={distancias[primer_indice_rotura]:.6f} "
                f"> límite dinámico {umbral_ruptura:g} "
                f"(1000 × d0, d0={distancia_inicial:g})"
            )

    figura, eje = plt.subplots(figsize=(10, 8), facecolor="black")
    figura.subplots_adjust(right=0.78, bottom=0.16)
    eje.set_facecolor("black")
    preparar_ejes(eje, trayectorias)
    fondo_mandelbrot = eje.imshow(
        crear_fondo_mandelbrot(), extent=(-2, 2, -2, 2), origin="lower",
        cmap="copper", norm="log", interpolation="bilinear",
        alpha=0.48, zorder=0, visible=False,
    )
    informacion = eje.text(
        0.02, 0.98, "", transform=eje.transAxes, va="top", ha="left",
        color="white", fontsize=10,
        bbox={"facecolor": "black", "edgecolor": "white", "alpha": 0.75, "pad": 6},
    )
    lineas = []
    puntos = []
    posiciones_actuales = []
    inicios = []
    conexion_actual, = eje.plot([], [], color="#ffe600", linewidth=1.0, alpha=0.8, label="Comparación")
    for trayectoria, color, nombre in zip(trayectorias, colores, nombres):
        visible = mostrar_atractor1 if nombre == "Atractor 1" else mostrar_atractor2
        etiqueta = f"{nombre} — trayectoria" if visible else "_nolegend_"
        linea, = eje.plot([], [], color=color, linewidth=0.55, alpha=0.9, label=etiqueta)
        punto, = eje.plot([], [], linestyle="", marker="o", markersize=3.5, color=color,
                          label=f"{nombre} — puntos discretos" if visible else "_nolegend_")
        posicion_actual, = eje.plot([], [], linestyle="", marker="o", markersize=8,
                                    markerfacecolor="none", markeredgecolor="white",
                                    markeredgewidth=1.2)
        inicio, = eje.plot([trayectoria[0, 1]], [trayectoria[0, 2]],
                           linestyle="", marker="x", markersize=8, color="white")
        lineas.append(linea)
        puntos.append(punto)
        posiciones_actuales.append(posicion_actual)
        inicios.append(inicio)
        linea.set_visible(visible)
        punto.set_visible(visible)
        posicion_actual.set_visible(visible)
        inicio.set_visible(visible)
    conexion_actual.set_visible(mostrar_comparacion)

    leyenda_colores = figura.legend(
        handles=[
            Line2D([0], [0], color="#a6bcc9", linewidth=2, label="Atractor 1"),
            Line2D([0], [0], color="#1e3252", linewidth=2, label="Atractor 2"),
            Line2D([0], [0], color="#ffe600", linewidth=2, label="Comparación"),
        ],
        loc="lower center", bbox_to_anchor=(0.39, 0.015), ncol=2,
        facecolor="black", edgecolor="white", labelcolor="white", fontsize=8,
    )

    controles = {}
    nombres_controles = (
        "Información", "Atractor 1", "Atractor 2",
        "Comparación amarilla", "Fondo Mandelbrot",
    )
    estados = {
        "Información": True, "Atractor 1": True, "Atractor 2": True,
        "Comparación amarilla": True, "Fondo Mandelbrot": False,
    }
    grupos = {
        "Atractor 1": (lineas[0], puntos[0], posiciones_actuales[0], inicios[0]),
        "Atractor 2": (lineas[1], puntos[1], posiciones_actuales[1], inicios[1]),
    }

    def cambiar_control(nombre):
        estados[nombre] = not estados[nombre]
        boton = controles[nombre]
        boton.label.set_text(f"{nombre}: {'ON' if estados[nombre] else 'OFF'}")
        if nombre == "Información":
            mostrar_informacion = estados[nombre]
            informacion.set_visible(mostrar_informacion)
        elif nombre in grupos:
            for artista in grupos[nombre]:
                artista.set_visible(estados[nombre])
        elif nombre == "Comparación amarilla":
            conexion_actual.set_visible(estados[nombre])
        elif nombre == "Fondo Mandelbrot":
            fondo_mandelbrot.set_visible(estados[nombre])
        color = "#174d24" if estados[nombre] else "#7a1717"
        hover_color = "#286b38" if estados[nombre] else "#a52a2a"
        boton.color = color
        boton.hovercolor = hover_color
        boton.ax.set_facecolor(color)
        boton.ax.patch.set_facecolor(color)
        figura.canvas.draw_idle()

    for posicion, nombre in enumerate(nombres_controles):
        estado_inicial = estados[nombre]
        color_inicial = "#174d24" if estado_inicial else "#7a1717"
        hover_inicial = "#286b38" if estado_inicial else "#a52a2a"
        boton_ax = figura.add_axes([0.80, 0.56 - posicion * 0.075, 0.18, 0.055])
        boton_ax.set_facecolor(color_inicial)
        boton = Button(boton_ax, f"{nombre}: {'ON' if estado_inicial else 'OFF'}",
                       color=color_inicial, hovercolor=hover_inicial)
        boton.label.set_color("white")
        boton.label.set_fontsize(8)
        boton.on_clicked(lambda evento, nombre=nombre: cambiar_control(nombre))
        controles[nombre] = boton

    # Se limita el número de fotogramas para que la animación sea manejable,
    # aunque la integración conserve todos los pasos calculados.
    cantidad_fotogramas = min(1200, pasos)
    indices = np.linspace(1, pasos, cantidad_fotogramas, dtype=int)

    def actualizar(indice):
        for trayectoria, parametros, linea, punto, posicion_actual in zip(
            trayectorias, parametros_animacion, lineas, puntos, posiciones_actuales
        ):
            if not linea.get_visible():
                linea.set_data([], [])
                punto.set_data([], [])
                posicion_actual.set_data([], [])
                continue
            linea.set_data(trayectoria[:indice, 1], trayectoria[:indice, 2])
            seleccion = trayectoria[:indice: parametros["compresion"]]
            punto.set_data(seleccion[:, 1], seleccion[:, 2])
            posicion_actual.set_data([trayectoria[indice - 1, 1]], [trayectoria[indice - 1, 2]])
        if mostrar_comparacion:
            conexion_actual.set_data(
                [trayectoria1[indice - 1, 1], trayectoria2[indice - 1, 1]],
                [trayectoria1[indice - 1, 2], trayectoria2[indice - 1, 2]],
            )
        if mostrar_comparacion and primer_indice_rotura is None:
            caos_texto = "No se supera el límite dinámico"
        elif mostrar_comparacion and indice - 1 >= primer_indice_rotura:
            caos_texto = (
                f"Entrada al caos detectada: paso {primer_indice_rotura}, "
                f"t={trayectoria1[primer_indice_rotura, 0]:.3f}"
            )
        elif mostrar_comparacion:
            caos_texto = "Calculando entrada al caos..."
        tiempo = trayectoria1[indice - 1, 0]
        if mostrar_comparacion:
            informacion.set_text(
                f"Progreso: paso {indice}/{pasos}\n"
                f"Tiempo: t={tiempo:.3f}\n"
                f"Distancia: {distancias[indice - 1]:.6f}\n"
                f"Lyapunov 1: {lyapunov1[indice - 1]:.6f}\n"
                f"Lyapunov 2: {lyapunov2[indice - 1]:.6f}\n"
                f"{caos_texto}"
            )
        else:
            informacion.set_text(f"Progreso: paso {indice}/{pasos}\nTiempo: t={tiempo:.3f}")
        return (
            *lineas,
            *puntos,
            *posiciones_actuales,
            conexion_actual,
            informacion,
            fondo_mandelbrot,
        )

    animacion = animation.FuncAnimation(
        figura, actualizar, frames=indices, interval=1000 / args.fps,
        blit=True, repeat=False,
    )

    if args.guardar_gif:
        animacion.save(args.guardar_gif, writer=animation.PillowWriter(fps=args.fps))
        print(f"Animación guardada: {args.guardar_gif}")
    plt.show()


if __name__ == "__main__":
    main()
