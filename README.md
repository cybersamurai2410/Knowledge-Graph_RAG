# Knowledge Graphs with RAG
Automatically build knowledge graphs from text documents with embeddings stored in a graph database for RAG (retrieval augemented generation) with LLMs to retrieve context for question-answering based on the relationships between the data by generating Cypher queries. 

- Use Neo4j query language Cypher to manage and retrieve data stored in knowledge graphs.
- Write knowledge graph queries that find and format text data to provide more relevant context to LLMs for Retrieval Augmented Generation.
- Build a question-answering system using Neo4j and LangChain to chat with a knowledge graph of structured text documents.

**Graph Schema:**
```bash
Node properties are the following: Chunk {textEmbedding:
LIST, f10kItem: STRING, chunkSeqId: INTEGER, text: STRING,
cik: STRING, cusip6: STRING, names: LIST, formId: STRING,
source: STRING, chunkId: STRING},Form {cusip6: STRING,
names: LIST, formId: STRING, source: STRING},Company
{location: POINT, cusip: STRING, names: LIST,
companyAddress: STRING, companyName: STRING, cusip6:
STRING},Manager {location: POINT, managerName: STRING,
managerCik: STRING, managerAddress: STRING},Address
{location: POINT, country: STRING, city: STRING, state:
STRING} Relationship properties are the following: SECTION
{f10kItem: STRING},OWNS_STOCK_IN {shares: INTEGER,
reportCalendarOrQuarter: STRING, value: FLOAT} The
relationships are the following: (:Chunk)-[:NEXT]-
>(:Chunk),(:Chunk)-[:PART_OF]->(:Form),(:Form)-[:SECTION]-
>(:Chunk),(:Company)-[:FILED]->(:Form),(:Company)-
[:LOCATED_AT]->(:Address),(:Manager)-[:LOCATED_AT]-
>(:Address),(:Manager)-[:OWNS_STOCK_IN]->(:Company)
```

**Inference:**
```python
cypherChain.run("What investment firms are in San Francisco?")
```
```bash
> Entering new GraphCypherQAChain chain...
Generated Cypher:
MATCH (mgr:Manager)-[:LOCATED_AT]->(mgrAddress:Address)
WHERE mgrAddress.city = 'San Francisco'
RETURN mgr.managerName
Full Context:
[{'mgr.managerName': 'PARNASSUS INVESTMENTS, LLC'}, {'mgr.managerName': 'SKBA CAPITAL MANAGEMENT LLC'}, {'mgr.managerName': 'ROSENBLUM SILVERMAN SUTTON S F INC /CA'}, {'mgr.managerName': 'CHARLES SCHWAB INVESTMENT MANAGEMENT INC'}, {'mgr.managerName': 'WELLS FARGO & COMPANY/MN'}, {'mgr.managerName': 'Dodge & Cox'}, {'mgr.managerName': 'Strait & Sound Wealth Management LLC'}, {'mgr.managerName': 'Sonoma Private Wealth LLC'}, {'mgr.managerName': 'Fund Management at Engine No. 1 LLC'}, {'mgr.managerName': 'SELDON CAPITAL LP'}]

> Finished chain.
PARNASSUS INVESTMENTS, LLC, ROSENBLUM SILVERMAN SUTTON S F
INC /CA, and Dodge & Cox are investment firms located in San
Francisco.
```
