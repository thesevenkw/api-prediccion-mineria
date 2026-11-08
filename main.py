"""API de clasificación de flores Iris para la tarea de Minería de Datos.

Ejecutar: python -m uvicorn main:app --reload
Documentación Swagger: http://127.0.0.1:8000/docs
"""

from typing import Annotated

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier


app = FastAPI(
    title="API de Predicción de Flores Iris",
    description=(
        "Proyecto de Minería de Datos: endpoint que usa un modelo "
        "de árbol de decisión para clasificar flores Iris."
    ),
    version="1.0.0",
)

# El conjunto de datos Iris viene incluido en scikit-learn.
datos_iris = load_iris()
modelo = DecisionTreeClassifier(max_depth=3, random_state=42)
modelo.fit(datos_iris.data, datos_iris.target)
especies = [str(datos_iris.target_names[indice]) for indice in modelo.classes_]

# scikit-learn convierte las entradas del árbol a float32.
Medida = Annotated[
    float,
    Field(strict=True, gt=0, le=3.4028234663852886e38, allow_inf_nan=False),
]


class DatosFlor(BaseModel):
    largo_sepalo: Medida = Field(description="Largo del sépalo (cm)")
    ancho_sepalo: Medida = Field(description="Ancho del sépalo (cm)")
    largo_petalo: Medida = Field(description="Largo del pétalo (cm)")
    ancho_petalo: Medida = Field(description="Ancho del pétalo (cm)")

    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={
            "examples": [
                {
                    "largo_sepalo": 5.1,
                    "ancho_sepalo": 3.5,
                    "largo_petalo": 1.4,
                    "ancho_petalo": 0.2,
                }
            ]
        }
    )


class Prediccion(BaseModel):
    especie_predicha: str = Field(description="Especie de Iris estimada")
    confianza: float = Field(ge=0, le=1, description="Probabilidad asignada a la especie predicha")
    probabilidades: dict[str, float] = Field(description="Probabilidad estimada para cada especie")


@app.exception_handler(RequestValidationError)
async def error_validacion(_request: Request, error: RequestValidationError):
    # Se omiten los valores recibidos, incluidos los números no serializables en JSON.
    detalle = [
        {campo: problema[campo] for campo in ("loc", "msg", "type")}
        for problema in error.errors()
    ]
    return JSONResponse(status_code=422, content={"detail": detalle})


@app.get("/", tags=["Estado"], summary="Página principal")
def inicio():
    return {
        "mensaje": "API de predicción Iris funcionando",
        "documentacion_swagger": "/docs",
        "endpoint_prediccion": "/predecir",
        "informacion_modelo": "/modelo",
        "estado_servicio": "/health",
    }


@app.get("/salud", include_in_schema=False)
@app.get("/health", tags=["Estado"], summary="Verificar que la API y el modelo están listos")
def salud():
    return {"estado": "ok", "modelo": "DecisionTreeClassifier"}


@app.get("/modelo", tags=["Predicción"], summary="Consultar cómo se entrenó el modelo")
def informacion_modelo():
    return {
        "algoritmo": type(modelo).__name__,
        "dataset": "Iris",
        "muestras_entrenamiento": len(datos_iris.target),
        "parametros": {"max_depth": modelo.max_depth, "random_state": modelo.random_state},
        "caracteristicas": list(DatosFlor.model_fields),
        "unidad": "cm",
        "clases": especies,
    }


@app.post(
    "/predecir",
    tags=["Predicción"],
    summary="Predecir la especie de flor",
    description="Envía cuatro medidas numéricas, positivas y finitas en centímetros.",
    response_model=Prediccion,
    responses={
        200: {
            "description": "Especie y probabilidades calculadas por el modelo.",
            "content": {
                "application/json": {
                    "example": {
                        "especie_predicha": "setosa",
                        "confianza": 1.0,
                        "probabilidades": {"setosa": 1.0, "versicolor": 0.0, "virginica": 0.0},
                    }
                }
            },
        },
        422: {
            "description": "JSON inválido o medidas faltantes, desconocidas o fuera de rango.",
            "content": {
                "application/json": {
                    "example": {
                        "detail": [
                            {
                                "loc": ["body", "largo_sepalo"],
                                "msg": "Input should be greater than 0",
                                "type": "greater_than",
                            }
                        ]
                    }
                }
            },
        },
    },
)
def predecir(flor: DatosFlor):
    # El orden corresponde a las cuatro columnas de Iris.
    entrada = [[
        flor.largo_sepalo,
        flor.ancho_sepalo,
        flor.largo_petalo,
        flor.ancho_petalo,
    ]]
    indice = int(modelo.predict(entrada)[0])
    probabilidades = modelo.predict_proba(entrada)[0]

    return {
        "especie_predicha": especies[indice],
        "confianza": round(float(probabilidades[indice]), 4),
        "probabilidades": {
            especie: round(float(probabilidad), 4)
            for especie, probabilidad in zip(especies, probabilidades, strict=True)
        },
    }
