class Lexicon:
    def __init__(self):
        self.mapper = {}
        self.inv_mapper = [] # all mapped values are keys

    def has_key(self, k):
        return self.mapper.has_key(k)

    def map_k_to_v(self,k):
        if not self.mapper.has_key(k):
            self.mapper[k] = len(self.inv_mapper)
            self.inv_mapper.append(k)
        return self.mapper[k]

    def map_v_to_k(self, v):
        if v > len(self.inv_mapper) - 1:
            raise Exception("Invalid mapped value")
        else:
            return self.inv_mapper[v]
        

if __name__ == '__main__':
    le = Lexicon()
    print le.map_k_to_v('dog')
    print le.map_k_to_v('cat')
    print le.map_v_to_k(0)
    print le.map_v_to_k(1)
    
