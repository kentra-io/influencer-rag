## Access Kibana
Open URL: `http://localhost:5601/` to access Kibana.

Navigate to `Dev Tools` -> `Console` to run ES queries.

## Example Queries

### Query all embeddings
```json
GET youtube_transcripts/_search
{
  "query": {
    "match_all": {}
  }
}
```

### Delete index
```json
DELETE youtube_transcripts/
```