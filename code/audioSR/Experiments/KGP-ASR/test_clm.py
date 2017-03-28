import decoder
import sys
def func( string ):
	dic = decoder.predictWithCLM(string)
	return sorted(dic, key = dic.get)[-1]
length = int(sys.argv[1])
s = str(sys.argv[2])
for i in range(length):
	s = s+ func(s)

print(s)
