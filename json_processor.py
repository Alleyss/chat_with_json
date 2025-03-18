from langchain_core.documents import Document

def process_json(json_data):
    processed_data = []
    
    for entry in json_data:
        # Convert JSON to text format that embeds well
        text_representation = (
            f"Name: {entry['name']}\n"
            f"Department: {entry['department']}\n"
            f"Subjects: {', '.join(entry['subjects'])}\n"
            f"Cabin Number: {entry['cabin_number']}\n"
            f"Email: {entry['email']}\n"
            f"Phone: {entry['phone']}"
        )
        processed_data.append(text_representation)
    
    return processed_data

