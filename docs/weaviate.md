# Switching to Weaviate DB

1. Run:
```shell
cd docker/weaviate
docker compose up -d
```

2. Modify `config.py` set `default_vector_db` to `VectorDbType.WEAVIATE`


# Weaviate REST/GraphQL API examples

### Get first 10 objects
```
GET http://localhost:8080/v1/objects?class=LangChain&limit=10
```

### Count objects
```
POST http://localhost:8080/v1/graphql
{
    Aggregate {
      LangChain {
        meta {
          count
        }
      }
    }
}
```

### Hybrid search example
```
{
  Get {
    LangChain(limit: 4, hybrid: { query: "who is sam altman?", alpha: 0.5 }) {
      title
      channel
      url
      file
      paragraph
      text
      _additional {
        explainScore
        score
      }
    }
  }
}
```