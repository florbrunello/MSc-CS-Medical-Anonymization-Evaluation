# Trabajo Especial
## Evaluación de diferentes aproximaciones a la anonimización de textos médicos en español

**Resumen**

Este proyecto evalúa y compara diferentes enfoques para el Reconocimiento de Entidades Nombradas (NER) en documentos médicos en español, utilizando cuatro conjuntos de datos y tres tipos de modelos. Este repositorio contiene los conjuntos de datos, código y experimentos para el proyecto de investigación. El proyecto compara tres familias de soluciones para la de-identificación de texto clínico: reglas simbólicas basadas en conocimientos de dominio mediante expresiones regulares (REGEX), redes neuronales recurrentes BiLSTM-CRF y grandes modelos de lenguaje (Llama 3.1). Los experimentos se ejecutaron en cuatro conjuntos de datos (dos públicos y dos sintéticos). Se proporciona todo lo necesario para reproducir los experimentos.

## Estructura del repositorio

```
README.md                    <- archivo en inglés
README_es.md                 <- este archivo (versión en español)
Datasets/                    <- .txt, .ann, .xml (BRAT/XML)
  ├─ MEDDOCAN/
  ├─ SPG/
  ├─ SPGExtended/
  └─ CARMEN-I/   (restringido por licencia)
Models/
  ├─ REGEX/
  ├─ BiLSTM-CRF/
  └─ Llama3.1/
Results/
  ├─ results_dev.xlsx
  ├─ results_test.xlsx
  └─ plots_dev/
  └─ plots_dev/
Models/Scripts/
  └─ confusion_matrix_generator_global.py
LICENSE
```

## Conjuntos de datos (`Datasets/`)
Contiene cuatro conjuntos de datos médicos anotados en formato BRAT:

- **MEDDOCAN**: corpus público de 1.000 documentos médicos (500 entrenamiento, 250 desarrollo, 250 test). Utilizado en IberLEF, MEDDOCAN. 
- **SPG**: 1.000 registros clínicos sintéticos generados con el módulo Synthetic Patient Generator (500 entrenamiento, 250 desarrollo, 250 test).  
- **SPGExtended**: 448 registros sintéticos curados producidos mediante la extensión de las salidas de SPG y realizando curación automática/manual (358 entrenamiento, 45 desarrollo, 45 test).
- **CARMEN-I**: subconjunto de 1.697 documentos de casos clínicos en español (847 entrenamiento, 425 desarrollo, 425 test). Este conjunto de datos está restringido por licencia y por lo tanto no se incluye en este repositorio. El conjunto de datos se puede acceder a través de PhysioNet: https://physionet.org/content/carmen-i/1.0.1/


## Modelos (`Models/`)
Implementación y evaluación de tres enfoques diferentes:

##### 1. **REGEX** - Reglas simbólicas basadas en conocimiento de dominio usando expresiones regulares
- Patrones basados en reglas desarrollados por la comunidad ARPHAI.
- Las salidas son procesadas y convertidas al formato BRAT para evaluación.
- Fortalezas: interpretabilidad y bajo costo computacional. Debilidades: recall limitado y mal desempeño en texto variable.

##### 2. **BiLSTM-CRF** - Red Neuronal Recurrente
- La implementación sigue una arquitectura LSTM Bidireccional + CRF (inspirada en Ma & Hovy y Saluja et al.).
- Requiere embeddings de palabras/subpalabras, tokenización y entrenamiento supervisado.
- Ofrece un buen equilibrio entre rendimiento y eficiencia computacional.

##### 3. **Llama 3.1** - Modelo de Lenguaje Grande
- Utiliza el modelo Llama 3.1 para NER. Usado mediante prompting (one-/few-shot) para generar anotaciones in-line (XML) que se convierten a BRAT para evaluación.
- Los LLMs se desempeñan bien en textos similares a los de entrenamiento pero pueden mostrar menor robustez en registros clínicos de formatos ligeramente distintos. Las limitaciones incluyen alucinaciones y manejo inconsistente de offsets.

## Resultados
Los resultados se encuentran en la carpeta `Results/`:

### Métricas
- **`results_dev.xlsx`**: Métricas de evaluación en el conjunto de desarrollo
- **`results_test.xlsx`**: Métricas de evaluación en el conjunto de test
- Contiene métricas detalladas (Precisión, Recall, F1-Score) por modelo y conjunto de datos

### Visualizaciones
- **`plots_dev/`**: Matrices de confusión y gráficos para el conjunto de desarrollo
- **`plots_test/`**: Matrices de confusión y gráficos para el conjunto de test
- Comparación visual entre modelos (REGEX, BiLSTM-CRF, Llama3.1)

## Ejecución

Para ejecutar evaluaciones específicas, consulte los archivos `ejecucion.md` en cada modelo:
- `Models/REGEX/[dataset]/ejecucion.md`
- `Models/BiLSTM-CRF/ejecucion.md` 
- `Models/Llama3.1/[dataset]/ejecucion.md`

Para generar matrices de confusión globales:
```bash
cd Models/Scripts
python3 confusion_matrix_generator_global.py [MODEL] --partition [dev/test/both]
```

## Licencia

Este proyecto está licenciado bajo **CC0 1.0 Universal** (Creative Commons Zero) - consulte el archivo [LICENSE](LICENSE) para más detalles.

La licencia CC0 permite el uso libre, modificación y distribución del código sin restricciones, dedicando efectivamente el trabajo al dominio público.

## Agradecimientos

Este trabajo utilizó recursos computacionales de UNC Supercómputo (CCAD), que forman parte del SNCAD, Argentina y recursos de la comunidad ARPHAI.
