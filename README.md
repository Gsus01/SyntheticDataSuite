# SyntheticDataSuite

Este repositorio contiene componentes para la generación y procesamiento de datos sintéticos. El microservicio principal se encuentra en `components/preprocessing`.

Las pruebas están organizadas bajo `tests/` siguiendo un enfoque por capas (unitarias, de integración y de caja negra). Para ejecutar todo el conjunto de tests basta con usar Docker Compose desde la raíz del proyecto:

```bash
docker compose run --rm test-preprocessing
```

Consulta la documentación específica de cada componente para más detalles.
