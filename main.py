"""API de clasificación de flores Iris para la tarea de Minería de Datos.

Ejecutar: python -m uvicorn main:app --reload
Documentación Swagger: http://127.0.0.1:8000/docs
"""

import os
import sqlite3
from collections.abc import Iterator
from contextlib import asynccontextmanager, closing
from pathlib import Path as FilePath
from typing import Annotated

from fastapi import Depends, FastAPI, HTTPException, Path, Query, Request, Response
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier


@asynccontextmanager
async def preparar_base(app: FastAPI):
    ruta_local = FilePath(__file__).with_name("flores.sqlite3")
    app.state.ruta_flores = os.environ.get("FLORES_DB", str(ruta_local))
    with closing(sqlite3.connect(app.state.ruta_flores, autocommit=True)) as conexion:
        conexion.execute("""
            CREATE TABLE IF NOT EXISTS flores (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                largo_sepalo REAL NOT NULL,
                ancho_sepalo REAL NOT NULL,
                largo_petalo REAL NOT NULL,
                ancho_petalo REAL NOT NULL
            )
        """)
    yield


app = FastAPI(
    title="API de Predicción de Flores Iris",
    description=(
        "Proyecto de Minería de Datos. Guarda y administra mediciones de flores "
        "y predice su especie de Iris con un árbol de decisión."
    ),
    version="1.1.0",
    lifespan=preparar_base,
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
EJEMPLO_FLOR = {
    "largo_sepalo": 5.1,
    "ancho_sepalo": 3.5,
    "largo_petalo": 1.4,
    "ancho_petalo": 0.2,
}


class DatosFlor(BaseModel):
    largo_sepalo: Medida = Field(description="Largo del sépalo (cm)")
    ancho_sepalo: Medida = Field(description="Ancho del sépalo (cm)")
    largo_petalo: Medida = Field(description="Largo del pétalo (cm)")
    ancho_petalo: Medida = Field(description="Ancho del pétalo (cm)")

    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={"examples": [EJEMPLO_FLOR]},
    )


class Prediccion(BaseModel):
    especie_predicha: str = Field(description="Especie de Iris estimada")
    confianza: float = Field(ge=0, le=1, description="Probabilidad asignada a la especie predicha")
    probabilidades: dict[str, float] = Field(description="Probabilidad estimada para cada especie")


class FlorGuardada(DatosFlor):
    id: int = Field(description="Identificador asignado al crear el registro")
    model_config = ConfigDict(json_schema_extra={"examples": [{"id": 1, **EJEMPLO_FLOR}]})


def conectar_flores(request: Request) -> Iterator[sqlite3.Connection]:
    with closing(sqlite3.connect(
        request.app.state.ruta_flores, autocommit=True, check_same_thread=False
    )) as conexion:
        conexion.row_factory = sqlite3.Row
        yield conexion


ConexionFlores = Annotated[sqlite3.Connection, Depends(conectar_flores)]
IdFlor = Annotated[int, Path(ge=1, le=2**63 - 1, description="Identificador de la flor")]
FLOR_NO_ENCONTRADA = "No se encontró la flor."
RESPUESTA_FLOR_NO_ENCONTRADA = {
    404: {
        "description": FLOR_NO_ENCONTRADA,
        "content": {"application/json": {"example": {"detail": FLOR_NO_ENCONTRADA}}},
    }
}


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
        "registros_flores": "/flores",
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


@app.post(
    "/flores", tags=["Flores"], summary="Crear un registro de flor",
    response_model=FlorGuardada, status_code=201,
)
def crear_flor(flor: DatosFlor, conexion: ConexionFlores, respuesta: Response):
    medidas = flor.model_dump()
    cursor = conexion.execute("""
        INSERT INTO flores (largo_sepalo, ancho_sepalo, largo_petalo, ancho_petalo)
        VALUES (:largo_sepalo, :ancho_sepalo, :largo_petalo, :ancho_petalo)
    """, medidas)
    respuesta.headers["Location"] = f"/flores/{cursor.lastrowid}"
    return {"id": cursor.lastrowid, **medidas}


@app.get("/flores", tags=["Flores"], summary="Listar flores", response_model=list[FlorGuardada])
def listar_flores(
    conexion: ConexionFlores,
    limite: Annotated[int, Query(ge=1, le=100, description="Máximo de registros")] = 50,
    omitir: Annotated[int, Query(ge=0, le=2**63 - 1, description="Registros que se saltan")] = 0,
):
    filas = conexion.execute(
        "SELECT * FROM flores ORDER BY id LIMIT ? OFFSET ?", (limite, omitir)
    )
    return [dict(fila) for fila in filas]


@app.get(
    "/flores/{id_flor}", tags=["Flores"], summary="Consultar una flor por ID",
    response_model=FlorGuardada, responses=RESPUESTA_FLOR_NO_ENCONTRADA,
)
def consultar_flor(id_flor: IdFlor, conexion: ConexionFlores):
    fila = conexion.execute("SELECT * FROM flores WHERE id = ?", (id_flor,)).fetchone()
    if fila is None:
        raise HTTPException(status_code=404, detail=FLOR_NO_ENCONTRADA)
    return dict(fila)


@app.put(
    "/flores/{id_flor}", tags=["Flores"], summary="Actualizar una flor",
    description="Envía las cuatro medidas para reemplazar los valores guardados.",
    response_model=FlorGuardada, responses=RESPUESTA_FLOR_NO_ENCONTRADA,
)
def actualizar_flor(id_flor: IdFlor, flor: DatosFlor, conexion: ConexionFlores):
    registro = {"id": id_flor, **flor.model_dump()}
    cursor = conexion.execute("""
        UPDATE flores
        SET largo_sepalo = :largo_sepalo, ancho_sepalo = :ancho_sepalo,
            largo_petalo = :largo_petalo, ancho_petalo = :ancho_petalo
        WHERE id = :id
    """, registro)
    if cursor.rowcount == 0:
        raise HTTPException(status_code=404, detail=FLOR_NO_ENCONTRADA)
    return registro


@app.delete(
    "/flores/{id_flor}", tags=["Flores"], summary="Eliminar una flor",
    status_code=204, responses=RESPUESTA_FLOR_NO_ENCONTRADA,
)
def eliminar_flor(id_flor: IdFlor, conexion: ConexionFlores):
    cursor = conexion.execute("DELETE FROM flores WHERE id = ?", (id_flor,))
    if cursor.rowcount == 0:
        raise HTTPException(status_code=404, detail=FLOR_NO_ENCONTRADA)
    return Response(status_code=204)
