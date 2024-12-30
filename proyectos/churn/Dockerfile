FROM python:3.10-slim 

RUN pip install pipenv

WORKDIR /app
COPY Pipfile /app/
COPY Pipfile.lock /app/

RUN pipenv install --system --deploy

COPY src/predict.py /app/
COPY data/modelo_churn.joblib /app/

EXPOSE 9696

ENTRYPOINT ["waitress-serve", "--port=9696", "predict:app"]
