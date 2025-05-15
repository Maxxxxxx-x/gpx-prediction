from psycopg2.extensions import connection
from typing_extensions import List, Tuple, Any
import psycopg2
import os


def connect_database() -> connection | None:
    host = os.getenv("DB_HOST", default=None)
    port = os.getenv("DB_PORT", default=None)
    user = os.getenv("DB_USER", default=None)
    password = os.getenv("DB_PASS", default=None)
    dbName = os.getenv("DB_NAME", default=None)

    if not (host and port and user and password and dbName):
        print("Missing ENV variable")
        return None
    conn_string = f'host={host} port={port} user={
        user} password={password} dbname={dbName}'
    return psycopg2.connect(conn_string)


def fetch_record_for_training(
        connection: connection,
        distance: int,
        query_limit: int,
        offset: int
) -> List[Tuple[Any, ...]]:
    query = "SELECT id, rawdata FROM records WHERE distance >= %s LIMIT %s OFFSET %s;"
    cursor = connection.cursor()
    cursor.execute(query, (distance, query_limit, offset))
    data = cursor.fetchall()
    cursor.close()
    return data


def fetch_record_for_prediction(
        connection: connection,
        id=""
) -> Tuple[Any, ...] | None:
    cursor = connection.cursor()
    if id != "":
        query = "SELECT id, rawdata FROM records WHERE id = %s LIMIT 1;"
        cursor.execute(query, (id,))
    else:
        query = "SELECT id, rawdata FROM records ORDER BY RANDOM() LIMIT 1;"
        cursor.execute(query)
    data = cursor.fetchone()
    cursor.close()
    return data
