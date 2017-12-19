import matplotlib

PATH = "./session1/ECHO-2017_12_19_12_18_15.txt"

file = open(PATH,"r")
for line in file:
	print (line)