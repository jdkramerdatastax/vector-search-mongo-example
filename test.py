from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi
import requests
from dotenv import dotenv_values

config = dotenv_values('.env')

uri = config['MONGODB_URI']

# Create a new client and connect to the server
client = MongoClient(uri, server_api=ServerApi('1'))
# Send a ping to confirm a successful connection
try:
    client.admin.command('ping')
    print("Pinged your deployment. You successfully connected to MongoDB!")
except Exception as e:
    print(e)
    
db = client.sample_mflix
collection = db.movies

hf_token = config['HUGGINGFACE_API_KEY']

embedding_url = "https://api-inference.huggingface.co/pipeline/feature-extraction/sentence-transformers/all-MiniLM-L6-v2"

def generate_embedding(text: str) -> list[float]:
    response = requests.post(
	    embedding_url,
		headers={"Authorization": f"Bearer {hf_token}"},
		json={"inputs": text})
    if response.status_code != 200:
        raise ValueError(f"Request failed with status code {response.status_code}: {response.text}")
    return response.json()

for doc in collection.find({'plot':{"$exists": True}}).limit(50):
      doc['plot_embedding_hf'] = generate_embedding(doc['plot'])
      collection.replace_one({'_id': doc['_id']}, doc)
 
query = "imaginary characters from outer space at war"

results = collection.aggregate([
    {
        '$search': {
            "index": "PlotSemanticSearch",
            "knnBeta": {
                "vector": generate_embedding(query),
                "k": 4,
                "path": "plot_embedding_hf"}
        }
    }
])

for document in results:
    print(f'Movie Name: {document["title"]}\nMovie Plot: {document["plot"]}\n')