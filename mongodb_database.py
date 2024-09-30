from pymongo import MongoClient

mongo_uri = "<MONGO URI GOES HERE>"

client = MongoClient(mongo_uri)

db = client["<CLIENT NAME GOES HERE>"]

collection = db["<COLLECTION NAME GOES HERE>"]