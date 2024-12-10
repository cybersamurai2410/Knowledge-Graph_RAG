from dotenv import load_dotenv
import os
import json
import textwrap
import csv

from langchain_community.graphs import Neo4jGraph
from langchain_community.vectorstores import Neo4jVector
from langchain_openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQAWithSourcesChain
from langchain_openai import ChatOpenAI

import warnings
warnings.filterwarnings("ignore")

load_dotenv('.env', override=True)
NEO4J_URI = os.getenv('NEO4J_URI')
NEO4J_USERNAME = os.getenv('NEO4J_USERNAME')
NEO4J_PASSWORD = os.getenv('NEO4J_PASSWORD')
NEO4J_DATABASE = os.getenv('NEO4J_DATABASE')
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
OPENAI_ENDPOINT = os.getenv('OPENAI_BASE_URL') + '/embeddings'

VECTOR_INDEX_NAME = 'form_10k_chunks'
VECTOR_NODE_LABEL = 'Chunk'
VECTOR_SOURCE_PROPERTY = 'text'
VECTOR_EMBEDDING_PROPERTY = 'textEmbedding'

kg = Neo4jGraph(
    url=NEO4J_URI, username=NEO4J_USERNAME, password=NEO4J_PASSWORD, database=NEO4J_DATABASE
)

# SEC (Securities and Exchange Commission) form 10-k  
first_file_name = "./data/form10k/0000950170-23-027948.json"
first_file_as_object = json.load(open(first_file_name))

# Split form 10-k sections into chunks 
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size = 2000,
    chunk_overlap  = 200,
    length_function = len,
    is_separator_regex = False,
)

# Actual function used to split text data from json file and add metadata 
def split_form10k_data_from_file(file):
    chunks_with_metadata = [] # use this to accumlate chunk records
    file_as_object = json.load(open(file)) # open the json file

    for item in ['item1','item1a','item7','item7a']: # pull these keys from the json
        print(f'Processing {item} from {file}') 
        item_text = file_as_object[item] # grab the text of the item
        item_text_chunks = text_splitter.split_text(item_text) # split the text into chunks
        chunk_seq_id = 0

        for chunk in item_text_chunks[:20]: # only take the first 20 chunks
            form_id = file[file.rindex('/') + 1:file.rindex('.')] # extract form id from file name
            # finally, construct a record with metadata and the chunk text
            chunks_with_metadata.append({
                'text': chunk, 
                # metadata from looping...
                'f10kItem': item,
                'chunkSeqId': chunk_seq_id,
                # constructed metadata...
                'formId': f'{form_id}', # pulled from the filename
                'chunkId': f'{form_id}-{item}-chunk{chunk_seq_id:04d}',
                # metadata from file...
                'names': file_as_object['names'],
                'cik': file_as_object['cik'],
                'cusip6': file_as_object['cusip6'],
                'source': file_as_object['source'],
            })
            chunk_seq_id += 1

        print(f'\tSplit into {chunk_seq_id} chunks')

    return chunks_with_metadata

first_file_chunks = split_form10k_data_from_file(first_file_name)

# Merge chunks to graph with Chunk label for node with these properties 
merge_chunk_node_query = """
MERGE(mergedChunk:Chunk {chunkId: $chunkParam.chunkId})
    ON CREATE SET 
        mergedChunk.names = $chunkParam.names,
        mergedChunk.formId = $chunkParam.formId, 
        mergedChunk.cik = $chunkParam.cik, 
        mergedChunk.cusip6 = $chunkParam.cusip6, 
        mergedChunk.source = $chunkParam.source, 
        mergedChunk.f10kItem = $chunkParam.f10kItem, 
        mergedChunk.chunkSeqId = $chunkParam.chunkSeqId, 
        mergedChunk.text = $chunkParam.text
RETURN mergedChunk
"""

kg = Neo4jGraph(
    url=NEO4J_URI, username=NEO4J_USERNAME, password=NEO4J_PASSWORD, database=NEO4J_DATABASE
)

# Create constraint to ensure chunk id unique for all nodes
kg.query("""
CREATE CONSTRAINT unique_chunk IF NOT EXISTS 
    FOR (c:Chunk) REQUIRE c.chunkId IS UNIQUE
""")
kg.query("SHOW INDEXES")

node_count = 0
for chunk in first_file_chunks:
    print(f"Creating `:Chunk` node for chunk ID {chunk['chunkId']}")

    kg.query(merge_chunk_node_query, 
            params={
                'chunkParam': chunk
            })
    node_count += 1
    
print(f"Created {node_count} nodes")

# Create vector index
kg.query("""
         CREATE VECTOR INDEX `form_10k_chunks` IF NOT EXISTS
          FOR (c:Chunk) ON (c.textEmbedding) 
          OPTIONS { indexConfig: {
            `vector.dimensions`: 1536,
            `vector.similarity_function`: 'cosine'    
         }}
""")

# Calculate embedding vectors for chunks and populate index
kg.query("""
    MATCH (chunk:Chunk) WHERE chunk.textEmbedding IS NULL
    WITH chunk, genai.vector.encode(
      chunk.text, 
      "OpenAI", 
      {
        token: $openAiApiKey, 
        endpoint: $openAiEndpoint
      }) AS vector
    CALL db.create.setNodeVectorProperty(chunk, "textEmbedding", vector)
    """, 
    params={"openAiApiKey":OPENAI_API_KEY, "openAiEndpoint": OPENAI_ENDPOINT} )

def neo4j_vector_search(question):
  """Search for similar nodes using the Neo4j vector index"""
  vector_search_query = """
    WITH genai.vector.encode(
      $question, 
      "OpenAI", 
      {
        token: $openAiApiKey,
        endpoint: $openAiEndpoint
      }) AS question_embedding
    CALL db.index.vector.queryNodes($index_name, $top_k, question_embedding) yield node, score
    RETURN score, node.text AS text
  """
  similar = kg.query(vector_search_query, 
                     params={
                      'question': question, 
                      'openAiApiKey':OPENAI_API_KEY,
                      'openAiEndpoint': OPENAI_ENDPOINT,
                      'index_name':VECTOR_INDEX_NAME, 
                      'top_k': 10})
  return similar

search_results = neo4j_vector_search(
    'In a single sentence, tell me about Netapp.'
)

neo4j_vector_store = Neo4jVector.from_existing_graph(
    embedding=OpenAIEmbeddings(),
    url=NEO4J_URI,
    username=NEO4J_USERNAME,
    password=NEO4J_PASSWORD,
    index_name=VECTOR_INDEX_NAME,
    node_label=VECTOR_NODE_LABEL,
    text_node_properties=[VECTOR_SOURCE_PROPERTY],
    embedding_node_property=VECTOR_EMBEDDING_PROPERTY,
)

retriever = neo4j_vector_store.as_retriever()

chain = RetrievalQAWithSourcesChain.from_chain_type(
    ChatOpenAI(temperature=0), 
    chain_type="stuff", 
    retriever=retriever
)

def prettychain(question: str) -> str:
    """Pretty print the chain's response to a question"""
    response = chain({"question": question},
        return_only_outputs=True,)
    print(textwrap.fill(response['answer'], 60))

prettychain("What is Netapp's primary business?")

cypher = """
  MATCH (anyChunk:Chunk) 
  WITH anyChunk LIMIT 1
  RETURN anyChunk { .names, .source, .formId, .cik, .cusip6 } as formInfo
"""
form_info_list = kg.query(cypher)

cypher = """
    MERGE (f:Form {formId: $formInfoParam.formId })
      ON CREATE 
        SET f.names = $formInfoParam.names
        SET f.source = $formInfoParam.source
        SET f.cik = $formInfoParam.cik
        SET f.cusip6 = $formInfoParam.cusip6
"""
kg.query(cypher, params={'formInfoParam': form_info})

cypher = """
  MATCH (from_same_section:Chunk)
  WHERE from_same_section.formId = $formIdParam
    AND from_same_section.f10kItem = $f10kItemParam
  WITH from_same_section
    ORDER BY from_same_section.chunkSeqId ASC
  WITH collect(from_same_section) as section_chunk_list
    CALL apoc.nodes.link(
        section_chunk_list, 
        "NEXT", 
        {avoidDuplicates: true}
    )
  RETURN size(section_chunk_list)
"""

for form10kItemName in ['item1', 'item1a', 'item7', 'item7a']:
  kg.query(cypher, params={'formIdParam':form_info['formId'], 
                           'f10kItemParam': form10kItemName})

cypher = """
  MATCH (c:Chunk), (f:Form)
    WHERE c.formId = f.formId
  MERGE (c)-[newRelationship:PART_OF]->(f) 
  RETURN count(newRelationship)
"""
kg.query(cypher)

cypher = """
  MATCH (first:Chunk), (f:Form)
  WHERE first.formId = f.formId
    AND first.chunkSeqId = 0
  WITH first, f
    MERGE (f)-[r:SECTION {f10kItem: first.f10kItem}]->(first) 
  RETURN count(r)
"""
kg.query(cypher) 

cypher = """
    MATCH window = (c1:Chunk)-[:NEXT]->(c2:Chunk)-[:NEXT]->(c3:Chunk) 
        WHERE c1.chunkId = $chunkIdParam
    RETURN length(window) as windowPathLength
    """
kg.query(cypher,
         params={'chunkIdParam': next_chunk_info['chunkId']})

cypher = """
  MATCH window=
      (:Chunk)-[:NEXT*0..1]->(c:Chunk)-[:NEXT*0..1]->(:Chunk) // *0..1 variable length where 0 is minimum number of relationships and 1 is maximum
    WHERE c.chunkId = $chunkIdParam
  RETURN length(window)
  """
kg.query(cypher,
         params={'chunkIdParam': first_chunk_info['chunkId']})

vector_store_extra_text = Neo4jVector.from_existing_index(
    embedding=OpenAIEmbeddings(),
    url=NEO4J_URI,
    username=NEO4J_USERNAME,
    password=NEO4J_PASSWORD,
    database="neo4j",
    index_name=VECTOR_INDEX_NAME,
    text_node_property=VECTOR_SOURCE_PROPERTY,
    retrieval_query=retrieval_query_extra_text, 
)

retriever_extra_text = vector_store_extra_text.as_retriever()

chain_extra_text = RetrievalQAWithSourcesChain.from_chain_type(
    ChatOpenAI(temperature=0), 
    chain_type="stuff", 
    retriever=retriever_extra_text
)

all_form13s = []
with open('./data/form13.csv', mode='r') as csv_file:
    csv_reader = csv.DictReader(csv_file)
    for row in csv_reader: 
      all_form13s.append(row)

cypher = """
  MATCH (com:Company), (form:Form)
    WHERE com.cusip6 = form.cusip6
  RETURN com.companyName, form.names
"""
kg.query(cypher)

cypher = """
  MATCH (com:Company), (form:Form)
    WHERE com.cusip6 = form.cusip6
  SET com.names = form.names
"""
kg.query(cypher)

kg.query("""
  MATCH (com:Company), (form:Form)
    WHERE com.cusip6 = form.cusip6
  MERGE (com)-[:FILED]->(form)
""")

cypher = """
  MERGE (mgr:Manager {managerCik: $managerParam.managerCik})
    ON CREATE
        SET mgr.managerName = $managerParam.managerName,
            mgr.managerAddress = $managerParam.managerAddress
"""
kg.query(cypher, params={'managerParam': first_form13})

kg.query("""
CREATE CONSTRAINT unique_manager 
  IF NOT EXISTS
  FOR (n:Manager) 
  REQUIRE n.managerCik IS UNIQUE
""")

kg.query("""
CREATE FULLTEXT INDEX fullTextManagerNames
  IF NOT EXISTS
  FOR (mgr:Manager) 
  ON EACH [mgr.managerName]
""")

cypher = """
  MERGE (mgr:Manager {managerCik: $managerParam.managerCik}) // create new node if exists 
    ON CREATE
        SET mgr.managerName = $managerParam.managerName,
            mgr.managerAddress = $managerParam.managerAddress
"""
for form13 in all_form13s:
  kg.query(cypher, params={'managerParam': form13 })

cypher = """
  MATCH (mgr:Manager {managerCik: $investmentParam.managerCik}), 
        (com:Company {cusip6: $investmentParam.cusip6})
  RETURN mgr.managerName, com.companyName, $investmentParam as investment
"""
kg.query(cypher, params={ 
    'investmentParam': first_form13 
})

cypher = """
MATCH (mgr:Manager {managerCik: $ownsParam.managerCik}), 
        (com:Company {cusip6: $ownsParam.cusip6})
MERGE (mgr)-[owns:OWNS_STOCK_IN { 
    reportCalendarOrQuarter: $ownsParam.reportCalendarOrQuarter
}]->(com)
ON CREATE // set extra values when relationship is created 
    SET owns.value  = toFloat($ownsParam.value), 
        owns.shares = toInteger($ownsParam.shares)
RETURN mgr.managerName, owns.reportCalendarOrQuarter, com.companyName
"""
kg.query(cypher, params={ 'ownsParam': first_form13 })

kg.query("""
MATCH (mgr:Manager {managerCik: $ownsParam.managerCik})
-[owns:OWNS_STOCK_IN]->
        (com:Company {cusip6: $ownsParam.cusip6})
RETURN owns { .shares, .value }
""", params={ 'ownsParam': first_form13 })

cypher = """
MATCH (mgr:Manager {managerCik: $ownsParam.managerCik}), 
        (com:Company {cusip6: $ownsParam.cusip6})
MERGE (mgr)-[owns:OWNS_STOCK_IN { 
    reportCalendarOrQuarter: $ownsParam.reportCalendarOrQuarter 
    }]->(com)
  ON CREATE
    SET owns.value  = toFloat($ownsParam.value), 
        owns.shares = toInteger($ownsParam.shares)
"""
for form13 in all_form13s:
  kg.query(cypher, params={'ownsParam': form13 })

cypher = """
    MATCH (:Chunk {chunkId: $chunkIdParam})-[:PART_OF]->(f:Form),
        (com:Company)-[:FILED]->(f),
        (mgr:Manager)-[owns:OWNS_STOCK_IN]->(com)
    RETURN mgr.managerName + " owns " + owns.shares + 
        " shares of " + com.companyName + 
        " at a value of $" + 
        apoc.number.format(toInteger(owns.value)) AS text
    LIMIT 10
    """
kg.query(cypher, params={
    'chunkIdParam': ref_chunk_id
})

vector_store = Neo4jVector.from_existing_graph(
    embedding=OpenAIEmbeddings(),
    url=NEO4J_URI,
    username=NEO4J_USERNAME,
    password=NEO4J_PASSWORD,
    index_name=VECTOR_INDEX_NAME,
    node_label=VECTOR_NODE_LABEL,
    text_node_properties=[VECTOR_SOURCE_PROPERTY],
    embedding_node_property=VECTOR_EMBEDDING_PROPERTY,
)
retriever = vector_store.as_retriever()
plain_chain = RetrievalQAWithSourcesChain.from_chain_type(
    ChatOpenAI(temperature=0), 
    chain_type="stuff", 
    retriever=retriever
)

investment_retrieval_query = """
MATCH (node)-[:PART_OF]->(f:Form),
    (f)<-[:FILED]-(com:Company),
    (com)<-[owns:OWNS_STOCK_IN]-(mgr:Manager)
WITH node, score, mgr, owns, com 
    ORDER BY owns.shares DESC LIMIT 10
WITH collect (
    mgr.managerName + 
    " owns " + owns.shares + 
    " shares in " + com.companyName + 
    " at a value of $" + 
    apoc.number.format(toInteger(owns.value)) + "." 
) AS investment_statements, node, score
RETURN apoc.text.join(investment_statements, "\n") + 
    "\n" + node.text AS text,
    score,
    { 
      source: node.source
    } as metadata
"""

vector_store_with_investment = Neo4jVector.from_existing_index(
    OpenAIEmbeddings(),
    url=NEO4J_URI,
    username=NEO4J_USERNAME,
    password=NEO4J_PASSWORD,
    database="neo4j",
    index_name=VECTOR_INDEX_NAME,
    text_node_property=VECTOR_SOURCE_PROPERTY,
    retrieval_query=investment_retrieval_query,
)
retriever_with_investments = vector_store_with_investment.as_retriever()
investment_chain = RetrievalQAWithSourcesChain.from_chain_type(
    ChatOpenAI(temperature=0), 
    chain_type="stuff", 
    retriever=retriever_with_investments
)
investment_chain(
    {"question": "In a single sentence, tell me about Netapp investors."},
    return_only_outputs=True,
)
