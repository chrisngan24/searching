class Tokenizer:
    def __init__(self):
        pass

    def tokenize_str(self, string):
        str_buff = []
        for c in string:
            if c.isalnum():
                str_buff.append(c.lower())
            elif len(str_buff) > 0:
                yield ''.join(str_buff)
                str_buff = []
        if len(str_buff) >0:
            yield ''.join(str_buff)

if __name__ == '__main__':
    a = 'The cat. The cat is fat'
    t = Tokenizer()
    print list(t.tokenize_str(a))
