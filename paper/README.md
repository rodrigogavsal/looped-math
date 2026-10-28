# Memoria (formato ACL, máximo 4 páginas sin referencias)

Compilar desde esta carpeta:

```bash
cd paper
latexmk -pdf main.tex
```

Requiere una distribución de LaTeX con `latexmk`. `acl.sty` y `acl_natbib.bst` son la plantilla oficial
(<https://github.com/acl-org/acl-style-files>).

`memoria.pdf` es la versión compilada de esta fuente (la que se entrega). Si se cambia algo, hay que
recompilar y sustituirla: `cp` del `main.pdf` generado a `memoria.pdf`.

## Estructura

- `main.tex`: título, autores, resumen y `\input` de las secciones.
- `sections/`: una sección por fichero (`01_introduccion.tex` ... `06_conclusiones.tex`).
- `references.bib`: bibliografía (solo se imprimen las obras citadas).
- `tablas/tabla_principal.tex`: tabla de resultados (se genera con `python -m src.analysis.run_all`).
- `figures/`: figuras en PDF usadas por la memoria (`loops_sweep.pdf`, `internal_main.pdf`).

## Convenciones

- `\pend{...}` marca en rojo un dato pendiente de confirmar y `\todo{...}`, un párrafo por redactar.
  Antes de entregar no debe quedar ninguno: `grep -rn "pend{\|todo{" main.tex sections`.
- Límite del enunciado: 4 páginas de contenido; las referencias van aparte. Comprobar que la sección
  «Conclusiones y limitaciones» termina en la página 4.
- En las comparaciones del texto, la primera cifra es la del Looped y la segunda la del GPT.
