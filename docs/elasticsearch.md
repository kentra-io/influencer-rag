# Switching to ElasticSearch DB
1. Run:
```shell
cd docker/elasticsearch
docker compose up -d
```
to start ElasticSearch and Kibana

2. Modify `config.py` set `default_vector_db` to `VectorDbType.ELASTICSEARCH`

## Access Kibana
Open URL: `http://localhost:5601/` to access Kibana.

Navigate to `Dev Tools` -> `Console` to run ES queries.

# Example Queries

### Query all embeddings
```
GET youtube_transcripts/_search
{
  "query": {
    "match_all": {}
  }
}
```

### Delete index
```
DELETE youtube_transcripts/
```