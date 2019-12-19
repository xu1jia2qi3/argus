import redis


db = redis.Redis(host="localhost", port=6379)
model = None

 

db.flushdb()