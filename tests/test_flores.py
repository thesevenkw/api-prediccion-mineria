import pytest
from fastapi.testclient import TestClient

from main import app


MEDIDAS = {
    "largo_sepalo": 5.1,
    "ancho_sepalo": 3.5,
    "largo_petalo": 1.4,
    "ancho_petalo": 0.2,
}


@pytest.fixture
def cliente(monkeypatch, tmp_path):
    monkeypatch.setenv("FLORES_DB", str(tmp_path / "flores.sqlite3"))
    with TestClient(app) as cliente:
        yield cliente


def test_ciclo_crud(cliente):
    creacion = cliente.post("/flores", json=MEDIDAS)
    assert creacion.status_code == 201
    registro = creacion.json()
    ruta = creacion.headers["location"]
    assert ruta == f"/flores/{registro['id']}"
    assert registro == {"id": registro["id"], **MEDIDAS}
    assert cliente.get("/flores").json() == [registro]
    assert cliente.get(ruta).json() == registro

    medidas_nuevas = {**MEDIDAS, "largo_sepalo": 6.2}
    actualizacion = cliente.put(ruta, json=medidas_nuevas)
    assert actualizacion.status_code == 200
    assert cliente.get(ruta).json() == {"id": registro["id"], **medidas_nuevas}

    eliminacion = cliente.delete(ruta)
    assert eliminacion.status_code == 204
    assert eliminacion.content == b""
    assert cliente.get("/flores").json() == []
    assert cliente.get(ruta).status_code == 404
    assert cliente.put(ruta, json=MEDIDAS).status_code == 404
    assert cliente.delete(ruta).status_code == 404


def test_validacion_flores(cliente):
    assert cliente.post("/flores", json={**MEDIDAS, "largo_sepalo": -1}).status_code == 422
    assert cliente.get("/flores/0").status_code == 422
    assert cliente.get("/flores?limite=101").status_code == 422


def test_persistencia_y_paginacion(monkeypatch, tmp_path):
    monkeypatch.setenv("FLORES_DB", str(tmp_path / "flores.sqlite3"))
    with TestClient(app) as cliente:
        primera = cliente.post("/flores", json=MEDIDAS).json()
        segunda = cliente.post("/flores", json=MEDIDAS).json()

    with TestClient(app) as cliente:
        assert cliente.get("/flores?limite=1").json() == [primera]
        assert cliente.get("/flores?limite=1&omitir=1").json() == [segunda]
