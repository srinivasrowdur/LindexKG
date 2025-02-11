import os
from llama_index.core import SimpleDirectoryReader, PropertyGraphIndex
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI
from dotenv import load_dotenv

# Load API key from environment
load_dotenv()

def create_sample_text():
    """Create a sample text file for demonstration"""
    sample_text = """
    Interleaf was a company that had smart people and built impressive technology. 
    They created software for document creation and added a scripting language that was a dialect of Lisp.
    However, they eventually got crushed by Moore's law.

    Viaweb was started by Paul Graham. It had a code editor where users could create web pages.
    Dan Giffin worked for Viaweb. The software worked via the web, which was innovative at the time.
    """
    
    # Create data directory if it doesn't exist
    os.makedirs("data", exist_ok=True)
    
    # Write sample text to file
    with open("data/sample.txt", "w") as f:
        f.write(sample_text)

def main():
    # Create sample data
    create_sample_text()
    
    # Load documents
    documents = SimpleDirectoryReader("data").load_data()
    
    # Create the property graph index
    index = PropertyGraphIndex.from_documents(
        documents,
        llm=OpenAI(model="gpt-4o-mini", temperature=0.3),
        embed_model=OpenAIEmbedding(model_name="text-embedding-3-small"),
        show_progress=True,
    )
    
    # Save the graph visualization
    index.property_graph_store.save_networkx_graph(name="knowledge_graph.html")
    
    # Example queries
    print("\nQuerying without source text:")
    retriever = index.as_retriever(include_text=False)
    nodes = retriever.retrieve("What happened at Interleaf and Viaweb?")
    for node in nodes:
        print(node.text)
    
    print("\nQuerying with summarization:")
    query_engine = index.as_query_engine(include_text=True)
    response = query_engine.query("What happened at Interleaf and Viaweb?")
    print(str(response))
    
    # Save the index for later use
    index.storage_context.persist(persist_dir="./storage")

if __name__ == "__main__":
    main() 