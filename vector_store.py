from langchain_core.vectorstores import InMemoryVectorStore

session_vector_stores = {}

def init_vector_store(session_id, embedding_model):
    """Initializes a vector store for a session."""
    global session_vector_stores

    if session_id not in session_vector_stores:
        session_vector_stores[session_id] = InMemoryVectorStore(embedding_model)

    return session_vector_stores[session_id]

def add_json_to_vector(session_id, data):
    global session_vector_stores

    if session_id in session_vector_stores:
        print("🔍 Storing Data in Vector Store:")
        for doc in data:
            print(doc)  # This will print stored documents in console

        session_vector_stores[session_id].add_documents(data)


def retrieve_json_from_vector(session_id, question, k=4):
    global session_vector_stores

    if session_id in session_vector_stores:
        retrieved_docs = session_vector_stores[session_id].similarity_search(question, k=k)

        if retrieved_docs:
            return "\n\n".join([doc.page_content for doc in retrieved_docs])
    
    return "No relevant information found in the JSON data."



