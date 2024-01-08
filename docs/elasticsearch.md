## Access Kibana
Open URL: `http://localhost:5601/` to access Kibana.

Navigate to `Dev Tools` -> `Console` to run ES queries.

## Example Queries

### Query all embeddings
```json
GET test_index3/_search
{
  "query": {
    "match_all": {}
  }
}
```