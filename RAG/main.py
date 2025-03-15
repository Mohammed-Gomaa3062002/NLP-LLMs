from llama_index.llms.ollama import Ollama
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.core.settings import Settings

# Global Configuration
class GlobalConfig:
    llm = None
    embed_model = None

    @staticmethod
    def initialize():
        # LLM (Ollama) with increased timeout
        GlobalConfig.llm = Ollama(
            model="llama2:latest",
            base_url="http://localhost:11434",
            request_timeout=300.0  # Increase timeout to 300 seconds (adjust as needed)
        )

        # HuggingFace Embeddings (Optional)
        GlobalConfig.embed_model = HuggingFaceEmbedding(
            model_name="BAAI/bge-small-en-v1.5"
        )


# Initialize Config
GlobalConfig.initialize()

# Load documents
documents = SimpleDirectoryReader("data").load_data()

# Update settings with the new LLM and embedding model
Settings.llm = GlobalConfig.llm
Settings.embed_model = GlobalConfig.embed_model

# Index and Query Engine
index = VectorStoreIndex.from_documents(documents)

# Query Engine using the updated settings
query_engine = index.as_query_engine()

# Test with a simpler query
response = query_engine.query("What is NLP?")
print("What is NLP?", '\n')
print(response)

