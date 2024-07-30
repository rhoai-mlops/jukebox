import kfp
import kfp.dsl as dsl
from kfp.dsl import (
    component,
    Input,
    Output,
    Dataset,
    Metrics,
)

@component(packages_to_install=["psycopg2", "pandas"])
def fetch_transactionsdb_data(
    datastore: dict, 
    dataset: Output[Dataset]
):
    """
    Fetches data from the transactionsdb datastore
    """
    
    import psycopg2
    import pandas as pd
    
    query = f"select * from {datastore['table']} limit 10000"
    with psycopg2.connect(host='transactionsdb.mlops-transactionsdb.svc.cluster.local', port=5432, dbname='transactionsdb', user='transactionsdb', password='transactionsdb') as connection:
        data = pd.read_sql_query(query, connection)
    print(data.head())
    
    dataset.path += ".csv"
    data.to_csv(dataset.path, index=False, header=True)