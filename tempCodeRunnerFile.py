import nltk
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer
from Phase_1.preprocessor import Preprocessor

nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')
nltk.download('averaged_perceptron_tagger_eng')

class SynonymMatcher:
    def __init__(self):
        self.preprocessor = Preprocessor()
        self.lemmatizer = WordNetLemmatizer()
        self.MATCH_THRESHOLD = 0.85 # Threshold for considering two words as synonyms based on Wu-Palmer similarity
        self.IRREGULAR_VERBS = {
    "sat": "sit",
    "went": "go",
    "ran": "run",
    "ate": "eat",
    "saw": "see",
    "took": "take",
    "came": "come",
    "said": "say",
    "got": "get",
    "made": "make",
    "knew": "know",
    "thought": "think",
    "told": "tell",
    "became": "become",
    "showed": "show",
    "felt": "feel",
    "left": "leave",
    "kept": "keep",
    "brought": "bring",
    "began": "begin",
    "grown": "grow",
    "drawn": "draw",
    "worn": "wear",
    "chosen": "choose",
    "spoken": "speak",
    "stolen": "steal",
    "broken": "break",
    "forgotten": "forget",
    "hidden": "hide",
    "risen": "rise",
    "fallen": "fall",
    "driven": "drive",
    "ridden": "ride",
    "rung": "ring",
    "sung": "sing",
    "sunk": "sink",
    "swum": "swim",
    "thrown": "throw",
    "blown": "blow",
    "grown": "grow",
    "known": "know",
    "shown": "show",
    "flown": "fly",
    "drew": "draw",
    "drove": "drive",
    "rode": "ride",
    "rose": "rise",
    "fell": "fall",
    "rang": "ring",
    "sang": "sing",
    "sank": "sink",
    "swam": "swim",
    "threw": "throw",
    "blew": "blow",
    "grew": "grow",
    "flew": "fly",
    "wore": "wear",
    "spoke": "speak",
    "broke": "break",
    "chose": "choose",
    "stole": "steal",
    "forgot": "forget",
    "hid": "hide",
}

    #Step 1: get POS tag (noun,verb,adjective,adverb)
    #Wordnet needs POS tag to look up the right synset
    def get_pos_tag(self,word):
        tag = nltk.pos_tag([word])[0][1]

        #Convert nltk POS tags to wordnet POS tags
        if tag.startswith('J'):
            return wordnet.ADJ
        elif tag.startswith('V'):
            return wordnet.VERB
        elif tag.startswith('R'):
            return wordnet.ADV
        else:
            return wordnet.NOUN
        
    #lemmatize before WordNet lookup
    def lemmatize(self, word):
        # Check irregular verbs first
     if word.lower() in self.IRREGULAR_VERBS:
        return self.IRREGULAR_VERBS[word.lower()]
    
     pos = self.get_pos_tag(word)
     lemmatized = self.lemmatizer.lemmatize(word, pos=pos)
    
     # If unchanged try forcing VERB pos
     if lemmatized == word:
        verb_lemma = self.lemmatizer.lemmatize(word, pos=wordnet.VERB)
        if verb_lemma != word:
            return verb_lemma
            
     return lemmatized

    #Step 2: Get all sysnets for a word
    def get_synsets(self, word):
        lemmatized_word = self.lemmatize(word)
        pos = self.get_pos_tag(lemmatized_word)
        synsets = wordnet.synsets(lemmatized_word, pos=pos)

        if not synsets:
            # If no synsets found with POS, try without POS
            synsets = wordnet.synsets(lemmatized_word)

        return synsets
    
    #antonym detection
    def are_antonyms(self, word_a, word_b):
        # Lemmatize first so WordNet finds the right entries
        word_a = self.lemmatize(word_a)
        word_b = self.lemmatize(word_b)

        synsets = wordnet.synsets(word_a)
        for syn in synsets:
            for lemma in syn.lemmas():
                antonyms = [ant.name() for ant in lemma.antonyms()]
                if word_b in antonyms:
                    return True
        return False


    #Step 3: Wu-palmer similarity between two words
    def wu_palmer_similarity(self, word1, word2):
        word_a = self.lemmatize(word1)
        word_b = self.lemmatize(word2)

        synsets_1 = self.get_synsets(word_a)
        synsets_2 = self.get_synsets(word_b)
    

        #if either word is not found in Wordnet, return 0 similarity
        if not synsets_1 or not synsets_2:
            return 0.0  
        
        #find the best similarity between any pair of synsets
        max_similarity = 0.0
        for syn1 in synsets_1:
            for syn2 in synsets_2:
                similarity = syn1.wup_similarity(syn2)
                if similarity is not None and similarity > max_similarity:
                    max_similarity = similarity

        return round(max_similarity, 4) # Round to 4 decimal places for better readability
    
    #Step 4: token to token best match
    #for each token in A find the best matching token in B
    def best_token_match(self, tokens_a, tokens_b):
        if not tokens_a or not tokens_b:
            return 0.0
        
        total_similarity = 0.0

        for token_a in tokens_a:
            best_match_score = 0.0
            best_match_word = None

            for token_b in tokens_b:
                #check exact mathch first
                if token_a == token_b:
                    score = 1.0
                else:
                    if self.are_antonyms(token_a, token_b):
                        score = 0.0
                    else:
                        score = self.wu_palmer_similarity(token_a, token_b)
                        if score < self.MATCH_THRESHOLD:
                             score = 0.0 # Consider as no match if below threshold
                if score > best_match_score:
                    best_match_score = score
                    best_match_word = token_b
            
            print(f"  '{token_a}' best match → '{best_match_word}' = {best_match_score}")
            total_similarity += best_match_score

        return round(total_similarity / len(tokens_a), 4) # Average similarity across all tokens in A
    
    #Step 5: Symmetric matching
    #Run matching both ways A -> B and B -> A and average the scores
    #This ensures the score is the same regardless of input order
    def compare(self, sentence_a, sentence_b):
        tokens_a = self.preprocessor.process(sentence_a)
        tokens_b = self.preprocessor.process(sentence_b)

        print(f"\nTokens A: {tokens_a}")
        print(f"Tokens B: {tokens_b}")

        print("\nA → B matches:")
        score_ab = self.best_token_match(tokens_a, tokens_b)

        print("\nB → A matches:")
        score_ba = self.best_token_match(tokens_b, tokens_a)

        # Symmetric average
        final_score = round((score_ab + score_ba) / 2, 4)
        return final_score


#testing
if __name__ == "__main__":
    matcher = SynonymMatcher()

    

    pairs = [
        ("The Cat sat on the Mat",          "A feline rested on a rug"),
        ("The Cat sat on the Mat",          "Stock markets crashed today"),
        ("The company increased revenue",   "The company reduced revenue"),
        ("I created a robot",               "I built a robot"),
        ("I wrote the letter",              "The letter was written by me"),
        ("The manager announced the policy during the meeting",
         "During the meeting the manager announced the policy"),
    ]

    print(matcher.lemmatize("sat"))      # expecting "sit"    → probably returns "sat"
    print(matcher.lemmatize("rested"))   # expecting "rest"   → probably returns "rested"
    print(matcher.lemmatize("created"))  # expecting "create" → probably returns "created"
    print(matcher.lemmatize("built"))    # expecting "build"  → probably returns "built"
    print(matcher.lemmatize("wrote"))    # expecting "write"  → probably returns "wrote"
    print(matcher.lemmatize("written"))  #

   
    print(matcher.wu_palmer_similarity("create", "build"))  
    # likely ~0.89 but let's verify
    print(matcher.wu_palmer_similarity("sit", "rest"))      
    # likely ~0.89 but let's verify

    print("=" * 70)
    for sentence_a, sentence_b in pairs:
        print(f"\nSentence A: {sentence_a}")
        print(f"Sentence B: {sentence_b}")
        score = matcher.compare(sentence_a, sentence_b)
        print(f"Final WordNet Score: {score}")
        print("-" * 70)




