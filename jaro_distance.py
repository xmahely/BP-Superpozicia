import math


class JaroDistance:

    def __init__(self, tensor1, tensor2):
        self.tensor1 = tensor1
        self.m1 = 0
        self.tensor2 = tensor2
        self.m2 = 0
        self.t = 0
        self.bool = True

    def getDistance(self):
        self.findMatchingChars(self.tensor1, self.tensor2)
        self.findMatchingChars(self.tensor2, self.tensor1)

        if self.m1 == 0 or self.m2 == 0:
            jaro = 0
        else:
            self.t = math.floor(self.t / 2)
            jaro = (1 / 3) * ((self.m1 / self.getSize(self.tensor1)) + (self.m2 / self.getSize(self.tensor2))
                              + ((min(self.m1, self.m2) - self.t) / min(self.m1, self.m2)))
        return jaro

    def getSize(self, tensor):
        return list(tensor.size())[0]

    def getBorder(self):
        return math.floor((min(self.getSize(self.tensor1), self.getSize(self.tensor2)) / 2))

    def addM(self):
        if self.bool:
            self.m1 = self.m1 + 1
        else:
            self.m2 = self.m2 + 1

    def findMatchingChars(self, t1, t2):
        border = self.getBorder()
        size = self.getSize(t2)

        for i, t in enumerate(list(t1)):
            if i < self.getSize(t2):
                if t1[i] == t2[i]:
                    self.addM()
            else:
                for j in range(max(0, i - border), min(size, 1 + i + border)):
                    if t1[i] == t2[j]:
                        if i != j:
                            self.t = self.t + (1 / 2)
                        self.addM()
                        break

        self.bool = False

