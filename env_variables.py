import os
from dotenv import load_dotenv

def load_env():
  load_dotenv('.env', override=True)

  env_vars = {
      'NEO4J_URI': os.getenv('NEO4J_URI'),
      'NEO4J_USERNAME': os.getenv('NEO4J_USERNAME'),
      'NEO4J_PASSWORD': os.getenv('NEO4J_PASSWORD'),
      'NEO4J_DATABASE': os.getenv('NEO4J_DATABASE') or 'neo4j',
      'OPENAI_API_KEY': os.getenv('OPENAI_API_KEY'),
      'OPENAI_ENDPOINT': os.getenv('OPENAI_BASE_URL') + '/embeddings'
  }
  return env_vars
  
