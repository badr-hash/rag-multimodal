import os

import ollama
import psycopg2

from config import PGVECTOR_DB, PGVECTOR_HOST, PGVECTOR_PASSWORD, PGVECTOR_PORT, PGVECTOR_USER

conn = psycopg2.connect(
    host=PGVECTOR_HOST,
    port=PGVECTOR_PORT,
    dbname=PGVECTOR_DB,
    user=PGVECTOR_USER,
    password=PGVECTOR_PASSWORD,
)
print("PostgreSQL OK")
print(ollama.list())