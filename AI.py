#A test file to run AI models for a base-line NOT INVOLVED IN FINAL CODE, just for testing and exploration of different models and approaches to sentence similarity.

from sentence_transformers import SentenceTransformer, util, CrossEncoder

# SBERT captures topical similarity well but is weak at understanding negation and antonyms
model = SentenceTransformer('all-MiniLM-L6-v2') 
#cross-encoder reads both sentences together rather than independently, making it much better at understanding structural differences like passive/active voice.
cross_encoder = CrossEncoder('cross-encoder/stsb-roberta-base') 


s1 = "The cat sat on the mat"
s2 = "A feline rested on a rug"
s3 = "Stock markets crashed today"
s4= "The team completed the project successfully before the game"
s5 = "The project was successfully completed before the game."
s6  = "The company increased revenue this year"
s7 = "The company reduced revenue this year"
s8 = "During the meeting, the manager announced the policy"
s9 = "The manager announced the policy during the meeting"
s10 = "I created a robot"
s11 = "I built a robot"
s12 = "I designed a robot"
s13 = "I wrote the letter"
s14 = "The letter was written by me"

emb1 = model.encode(s1)
emb2 = model.encode(s2)
emb3 = model.encode(s3)
emb4 = model.encode(s4)
emb5 = model.encode(s5)
em6 = model.encode(s6)
emb7 = model.encode(s7)
emb8 = model.encode(s8)
emb9 = model.encode(s9)
em10 = model.encode(s10)
em11 = model.encode(s11)
em12 = model.encode(s12)
em13 = model.encode(s13)
em14 = model.encode(s14)

print(util.cos_sim(emb1, emb2))  # ~0.56 (somewhat meaning)
print(util.cos_sim(emb1, emb3))  # ~0.07 (unrelated)
print(util.cos_sim(emb4, emb5))  # 0.88 (Very Similar meaning) 
print(util.cos_sim(em6, emb7))  # ~0.72 (Related) #look into why this is not higher, maybe the model is not good at understanding the difference between increase and decrease in revenue
print(util.cos_sim(emb8, emb9))  # ~0.98 (Near identical meaning)
print(util.cos_sim(em10, em11))  # ~0.92 (Near identical meaning)
print(util.cos_sim(em10, em12))  # ~0.90 (Near identical meaning)
print(util.cos_sim(em13, em14))  # ~0.84 (Very similar meaning)


#cross encoder scores
print(cross_encoder.predict([(s1, s2)]))  # ~0.69 (somewhat meaning)
print(cross_encoder.predict([(s1, s3)]))  # ~0.002 (unrelated)
print(cross_encoder.predict([(s13, s14)]))  # ~0.92 (Very similar meaning)

#ScoreMeaning
# 0.9 – 1.0 Near identical meaning
# 0.7 – 0.9 Very similar
# 0.5 – 0.7 Related / somewhat similar
# 0.3 – 0.5 Loosely related
# 0.0 – 0.3 Unrelated