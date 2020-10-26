import string
import random
s1 = string.ascii_lowercase
s2 = string.ascii_uppercase
s3 = string.digits

print("".join(map(str, random.sample(s1+s2+s3, 8))))
