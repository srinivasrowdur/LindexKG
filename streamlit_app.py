import streamlit as st
import os
from dotenv import load_dotenv
from llama_index.core import SimpleDirectoryReader, PropertyGraphIndex, Document, StorageContext
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI
from llama_index.graph_stores.neo4j import Neo4jPropertyGraphStore
import tempfile
from neo4j import GraphDatabase

# Load environment variables
load_dotenv()

# Initialize OpenAI API key
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

# Neo4j connection settings
NEO4J_URI = os.getenv("NEO4J_URI")
NEO4J_USERNAME = os.getenv("NEO4J_USERNAME")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")

def initialize_graph_store():
    """Initialize Neo4j graph store"""
    try:
        graph_store = Neo4jPropertyGraphStore(
            username=NEO4J_USERNAME,
            password=NEO4J_PASSWORD,
            url=NEO4J_URI,
            database="neo4j"
        )
        # Show existing data stats
        driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USERNAME, NEO4J_PASSWORD))
        with driver.session() as session:
            result = session.run("""
                MATCH (n)
                RETURN count(n) as node_count
            """)
            node_count = result.single()["node_count"]
            st.sidebar.write(f"Found {node_count} existing nodes in Neo4j")
        driver.close()
        
        st.sidebar.success("Neo4j Graph Store initialized successfully")
        return graph_store
    except Exception as e:
        st.sidebar.error(f"Error initializing Neo4j Graph Store: {str(e)}")
        return None

def initialize_index(graph_store):
    """Initialize the PropertyGraphIndex"""
    try:
        # Create new index if no data exists, otherwise load existing
        index = PropertyGraphIndex.from_documents(
            documents=[],  # Empty list as we'll add documents later
            property_graph_store=graph_store,
            llm=OpenAI(model="gpt-4o-mini", temperature=0.3),
            embed_model=OpenAIEmbedding(model_name="text-embedding-3-small"),
            show_progress=False  # Disable progress bars
        )
        st.sidebar.success("Index initialized successfully")
        return index
    except Exception as e:
        st.sidebar.error(f"Error initializing index: {type(e).__name__} - {str(e)}")
        return None

def clear_neo4j_database():
    """Clear all data from Neo4j database"""
    try:
        driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USERNAME, NEO4J_PASSWORD))
        with driver.session() as session:
            session.run("MATCH (n) DETACH DELETE n")
        driver.close()
        st.sidebar.success("Neo4j database cleared")
        # Clear all session state
        for key in list(st.session_state.keys()):
            del st.session_state[key]
    except Exception as e:
        st.sidebar.error(f"Error clearing database: {str(e)}")

def process_document(index, file_content):
    """Process a document and add it to the index"""
    if index is None:
        st.error("Index not properly initialized")
        return

    try:
        document = Document(text=file_content)
        # Use insert_nodes to properly process the document
        with st.spinner('Processing document...'):
            index.insert_nodes([document])
        st.sidebar.success("Document processed and added to knowledge graph")
    except Exception as e:
        st.sidebar.error(f"Error processing document: {str(e)}")

def initialize_query_engine(index):
    """Initialize or refresh the query engine"""
    try:
        return PropertyGraphIndex.from_existing(
            property_graph_store=index.property_graph_store,
            llm=OpenAI(model="gpt-4o-mini", temperature=0.3),
            embed_model=OpenAIEmbedding(model_name="text-embedding-3-small"),
            show_progress=False  # Disable progress bars
        ).as_query_engine(
            include_text=True,
            response_mode="compact",
            streaming=False
        )
    except Exception as e:
        st.sidebar.error(f"Error initializing query engine: {str(e)}")
        return None

def main():
    st.title("Chat with Contracts using Knowledge Graph")
    
    # Debug information
    st.sidebar.write("### Debug Information")
    st.sidebar.write(f"Neo4j URI: {NEO4J_URI}")
    st.sidebar.write(f"Neo4j Username: {NEO4J_USERNAME}")
    
    # Add sidebar options
    with st.sidebar:
        st.write("### Database Controls")
        if st.button("Clear Database"):
            clear_neo4j_database()
            st.rerun()
        
        if st.button("Show Database Stats"):
            driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USERNAME, NEO4J_PASSWORD))
            with driver.session() as session:
                result = session.run("""
                    MATCH (n)
                    WITH labels(n) as labels, count(*) as count
                    RETURN labels, count
                    ORDER BY count DESC
                """)
                st.write("### Node Types")
                for record in result:
                    st.write(f"{record['labels']}: {record['count']}")
            driver.close()

    # Initialize graph store
    if 'graph_store' not in st.session_state:
        st.sidebar.write("Initializing graph store...")
        st.session_state.graph_store = initialize_graph_store()
        if st.session_state.graph_store is None:
            st.error("Failed to initialize graph store. Check Neo4j connection details.")
            return
    
    # Initialize index
    if ('index' not in st.session_state or st.session_state.index is None) and st.session_state.graph_store is not None:
        st.sidebar.write("Initializing index...")
        st.session_state.index = initialize_index(st.session_state.graph_store)
        if st.session_state.index is None:
            st.error("Failed to initialize index. Please check Neo4j connection and try again.")
            return
        # Initialize query engine
        st.session_state.query_engine = initialize_query_engine(st.session_state.index)
        if st.session_state.query_engine is None:
            st.error("Failed to initialize query engine.")
            return
    
    # Add a state to track if we're processing a document
    if 'processing' not in st.session_state:
        st.session_state.processing = False
    
    # Initialize chat history
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    
    # File uploader
    if not st.session_state.processing:
        uploaded_file = st.file_uploader("Upload a PDF file", type=['pdf'])
    
    if uploaded_file and not st.session_state.processing:
        # Check if index is properly initialized
        if st.session_state.index is None:
            st.error("Index not initialized. Please check Neo4j connection and try again.")
            return

        st.session_state.processing = True
        # Read the PDF content
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_file_path = os.path.join(temp_dir, uploaded_file.name)
            with open(temp_file_path, "wb") as f:
                f.write(uploaded_file.getvalue())
            documents = SimpleDirectoryReader(temp_dir).load_data()
            
            # Process each document
            for doc in documents:
                process_document(st.session_state.index, doc.text)
        
        st.success('PDF processed successfully!')
        # Refresh query engine after all documents are processed
        with st.spinner('Initializing query engine...'):
            st.session_state.query_engine = initialize_query_engine(st.session_state.index)
    
    # Reset processing state after file upload widget is shown
    if st.session_state.processing:
        st.session_state.processing = False
        # Clear the uploaded file from memory
        st.session_state.uploaded_file = None
    
    # Chat interface
    if st.session_state.index is not None:
        # Display chat history
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
        
        # Chat input
        if prompt := st.chat_input("Ask a question about your documents"):
            # Add user message
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)
            
            # Generate response
            with st.chat_message("assistant"):
                with st.spinner('Retrieving answer...'):
                    try:
                        response = st.session_state.query_engine.query(prompt)
                        st.markdown(str(response))
                        st.session_state.messages.append(
                            {"role": "assistant", "content": str(response)}
                        )
                    except Exception as e:
                        st.error(f"Error generating response: {str(e)}")
        
        # Control buttons
        col1, col2 = st.columns(2)
        with col1:
            if st.button('Clear Chat History'):
                st.session_state.messages = []
                st.rerun()
        
        with col2:
            if st.button('Reset Knowledge Graph'):
                st.session_state.index = initialize_index(st.session_state.graph_store)
                st.session_state.messages = []
                st.rerun()

if __name__ == "__main__":
    main() 