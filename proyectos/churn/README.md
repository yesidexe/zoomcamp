# Churn prediction

- `train.py`

- La idea original era usar LogisticRegression, pero no se pudo, daba muchos errores, así que lo cambié, igualmente da el mismo rendimiento.
- No lo probé con randomforest
- Principalmente lo probé con train, test y val, pero el final lo haré sin val
- `custom_transformers.py` lo hice sin este archivo y me dio el mismo rendimiento, así que por ahora no lo voy a usar, pero lo dejaré por si se quiere usar.

## Uso de docker
- Abrir el Docker Desktop
- Hago el build, `docker build -t churn-prediction .`
- Lo corro de esta manera para poder ver los archivos, `docker run -it --rm --entrypoint=bash churn-prediction:latest`
- Y para ejecutar el server de waitress, `docker run -it --rm -p 9696:9696 churn-prediction:latest`

## AWS
