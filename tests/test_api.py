import pytest
from fastapi.testclient import TestClient

from main import app


cliente = TestClient(app)
CAMPOS = ("largo_sepalo", "ancho_sepalo", "largo_petalo", "ancho_petalo")
EJEMPLO = dict(zip(CAMPOS, (5.1, 3.5, 1.4, 0.2), strict=True))
ESPECIES = ["setosa", "versicolor", "virginica"]


def test_health():
    respuesta = cliente.get("/health")
    assert respuesta.status_code == 200
    assert respuesta.json() == {"estado": "ok", "modelo": "DecisionTreeClassifier"}


def test_modelo():
    respuesta = cliente.get("/modelo")
    assert respuesta.status_code == 200
    metadatos = respuesta.json()
    assert metadatos["dataset"] == "Iris"
    assert metadatos["algoritmo"] == "DecisionTreeClassifier"
    assert metadatos["caracteristicas"] == list(CAMPOS)
    assert metadatos["clases"] == ESPECIES
    assert metadatos["muestras_entrenamiento"] == 150


@pytest.mark.parametrize(
    ("medidas", "especie"),
    [
        ((5.1, 3.5, 1.4, 0.2), "setosa"),
        ((5.5, 2.3, 4.0, 1.3), "versicolor"),
        ((6.3, 3.3, 6.0, 2.5), "virginica"),
    ],
)
def test_prediccion(medidas, especie):
    respuesta = cliente.post("/predecir", json=dict(zip(CAMPOS, medidas, strict=True)))
    assert respuesta.status_code == 200
    datos = respuesta.json()
    assert datos["especie_predicha"] == especie
    probabilidades = datos["probabilidades"]
    assert set(probabilidades) == set(ESPECIES)
    assert all(0 <= valor <= 1 for valor in probabilidades.values())
    assert sum(probabilidades.values()) == pytest.approx(1, abs=0.0002)
    assert datos["confianza"] == probabilidades[especie] == max(probabilidades.values())


@pytest.mark.parametrize(
    ("entrada", "campo"),
    [
        ({}, "largo_sepalo"),
        ({**EJEMPLO, "largo_sepalo": 0}, "largo_sepalo"),
        ({**EJEMPLO, "ancho_sepalo": -1}, "ancho_sepalo"),
        ({**EJEMPLO, "largo_petalo": "1.4"}, "largo_petalo"),
        ({**EJEMPLO, "ancho_petalo": True}, "ancho_petalo"),
        ({**EJEMPLO, "largo_sepalo": 1e50}, "largo_sepalo"),
        ({**EJEMPLO, "color": "amarillo"}, "color"),
    ],
)
def test_medidas_invalidas(entrada, campo):
    respuesta = cliente.post("/predecir", json=entrada)
    assert respuesta.status_code == 422
    assert any(error["loc"] == ["body", campo] for error in respuesta.json()["detail"])


@pytest.mark.parametrize(
    "contenido",
    ['{"largo_sepalo":', '{"largo_sepalo": 1e400, "ancho_sepalo": 1, "largo_petalo": 1, "ancho_petalo": 1}'],
)
def test_json_invalido_o_no_finito(contenido):
    respuesta = cliente.post(
        "/predecir", content=contenido, headers={"content-type": "application/json"}
    )
    assert respuesta.status_code == 422
    assert respuesta.json()["detail"]


def test_documentacion():
    inicio = cliente.get("/")
    assert inicio.status_code == 200
    assert inicio.json()["documentacion_swagger"] == "/docs"
    swagger = cliente.get("/docs")
    assert swagger.status_code == 200
    assert "Swagger UI" in swagger.text
    especificacion = cliente.get("/openapi.json")
    assert especificacion.status_code == 200
    post = especificacion.json()["paths"]["/predecir"]["post"]
    assert {"200", "422"} <= set(post["responses"])
