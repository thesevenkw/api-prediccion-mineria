# API de predicción Iris

Proyecto de Minería de Datos. FastAPI expone un clasificador que usa cuatro medidas en centímetros para estimar si una flor es setosa, versicolor o virginica.

- [Repositorio público en GitHub](https://github.com/thesevenkw/api-prediccion-mineria)
- [API pública en Render](https://api-prediccion-mineria.onrender.com)
- [Swagger UI público](https://api-prediccion-mineria.onrender.com/docs)

## Instalar y ejecutar

Usa Python 3.14. La versión fijada para Render se encuentra en `.python-version`.

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m uvicorn main:app
```

En Linux o macOS, usa `.venv/bin/python` en lugar de `.\.venv\Scripts\python.exe`.

Abre [Swagger UI local](http://127.0.0.1:8000/docs). En `POST /predecir`, pulsa **Try it out** y luego **Execute** para enviar el ejemplo.

## Cómo funciona el modelo

Al arrancar, la API carga las 150 observaciones de Iris incluidas en scikit-learn y entrena un `DecisionTreeClassifier` con `max_depth=3` y `random_state=42`. El árbol divide las observaciones según sus mediciones. Las proporciones de clases en la hoja a la que llega una flor determinan sus probabilidades estimadas. La confianza es la probabilidad que el modelo asigna a la clase elegida. Los valores se redondean a cuatro decimales.

## Endpoints

| Método y ruta | Descripción |
| --- | --- |
| `GET /` | Muestra información general y enlaces a las rutas. |
| `GET /health` | Confirma que la API arrancó y el modelo está entrenado. |
| `GET /modelo` | Describe el algoritmo, sus parámetros, las medidas y las clases. |
| `POST /predecir` | Devuelve especie, confianza y probabilidades por clase. |
| `GET /docs` | Abre la interfaz interactiva de Swagger. |
| `GET /openapi.json` | Devuelve la especificación OpenAPI. |

### Ejemplo de predicción

```json
{
  "largo_sepalo": 5.1,
  "ancho_sepalo": 3.5,
  "largo_petalo": 1.4,
  "ancho_petalo": 0.2
}
```

```powershell
curl.exe -i http://127.0.0.1:8000/predecir -H "Content-Type: application/json" --data-binary "@ejemplo_peticion.json"
```

Respuesta HTTP 200 para el ejemplo.

```json
{
  "especie_predicha": "setosa",
  "confianza": 1.0,
  "probabilidades": {"setosa": 1.0, "versicolor": 0.0, "virginica": 0.0}
}
```

Envía las cuatro medidas como números positivos y finitos. Un JSON inválido o con campos incorrectos recibe HTTP 422 y una lista `detail` que identifica el problema.

## Pruebas automatizadas

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-dev.txt
.\.venv\Scripts\python.exe -m pytest -q
```

## Despliegue en Render

`render.yaml` define un servicio web Python en el plan **Free**, con la rama `main` y la comprobación de salud en `/health`. El servicio instala las dependencias con `pip install -r requirements.txt` y arranca con `uvicorn main:app --host 0.0.0.0 --port $PORT`. Render proporciona `PORT`. [Guía oficial de FastAPI en Render](https://render.com/docs/deploy-fastapi)

El servicio se creó desde la URL pública del repositorio. Los cambios se publican con un despliegue manual en Render. [Documentación de despliegues](https://render.com/docs/deploys)
