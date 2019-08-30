#
# ------------  1 print number which is divisible for 7 and 5 between 1500 and 1800---------------

# for i in range (1500,1800):
#     if (i % 7 == 0) and (i % 5 ==0)
#         print (i)


# --------------------------------- 2 Count number digit----------------------------------------

# n = int(input('Enter your input'))
# count = 0
# while(n > 1) :
#     n = n / 10
#     count = count + 1

# print('result' , count )


# --------------------------------- 3 isLower case or uper case--------------------------------

# def count_lower_upper(string):
#     lower = 0
#     upper = 0
#     for character in string:
#         if character.isupper():
#             upper = upper+1
#         elif character.islower():
#             lower = lower+1
#     return lower, upper


# lower, upper = count_lower_upper("Hello world")

# print('lower case', lower)
# print('upper case', upper)


# --------------------------------- 4 perfect number--------------------------------


# def is_perfect(input) :
#     sum = 0
#     for i in range(1,input):
#         if(input % i == 0):
#             sum = sum + i

#     if(sum == input):
#         return 'true'
#     else :
#         return 'false'

# print(is_perfect(6))


# --------------------------------- 5 reverse string --------------------------------

# def reverse_string(str):
#     s = ""
#     for ch in str :
#         s = ch + s
#         print (s)

# reverse_string('123456')

# --------------------------------- 6 count number of each character in a string----------------------------

# def count_characters(str):
#     distionary = {}
#     for ch in str:
#         keys = distionary.keys()
#         if(ch in keys) :
#             distionary[ch] += 1
#         else:
#             distionary[ch] = 1

#     return distionary

# print(count_characters("Hello world"))


# --------------------------------- numpy test----------------------------


# import numpy as np

# array which increase 2 in any pos from 30 to 70
# arr = np.arange(30, 71, 2)

# print(arr)

# identity_matrix = np.identity(3)

# print(identity_matrix)

# --------------------------------- Open CV test----------------------------

import cv2

img = cv2.imread('jojo.jpg', cv2.IMREAD_COLOR)
cv2.imshow('image', img)

k = cv2.waitKey(0)
if k == 27:         # wait for ESC key to exit
    cv2.destroyAllWindows()
