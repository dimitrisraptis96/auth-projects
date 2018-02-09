import csv
import sys
import string

txt_file = sys.argv[1]
csv_file = string.replace(sys.argv[1],".txt","  .csv")
text_list = []

with open(txt_file, "r") as my_input_file:
    for line in my_input_file:
        line = string.replace(line,"    ","   ")
        line = line.split("  ")
        text_list.append(",".join(line))
    print('File read.')

with open(csv_file, "w") as my_output_file:
    for line in text_list:
        my_output_file.write(line)
    print('File Successfully written.')