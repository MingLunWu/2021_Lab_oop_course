class Tokenizer:
    def __init__(self) -> None:
        self.dictionary = {
            "I": 0,
            "am": 1,
            "a": 2,
            "good": 3,
            "boy": 4,
            "how": 5,
            "about": 6,
            "you": 7,
            "[UNK]": 8
        }
        
    
if __name__ == "__main__":
    tokenizer = Tokenizer()
    # Test Case 1
    indexes = tokenizer.encode(["I", "am", "a", "good", "boy"])
    print(indexes) # 答案應該要是 [0, 1, 2, 3, 4]
    words = tokenizer.decode(indexes) # 答案應該要是 ['I', 'am', 'a', 'good', 'boy']
    print(words)

    # Test Case 2
    indexes = tokenizer.encode(["Allen", "how", "about", "you", "?"])
    print(indexes) # 答案應該要是 [8, 5, 6, 7, 8]
    words = tokenizer.decode(indexes)
    print(words) # 答案應該要是['[UNK]', 'how', 'about', 'you', '[UNK]']

