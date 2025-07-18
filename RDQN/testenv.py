import numpy as np

reliability = np.arange(101)/100
_alpha2 = 1.5
reliability = _alpha2 ** (2 * reliability - 1)
print(reliability)

# CP_UNI = [
#     20, 26, 27, 32, 28, 24, 32, 24, 8, 9, 26, 26, 33, 34, 2, 31, 17, 11, 26, 24
# ]

# CP_R_UNI_10per = [
#     4, 30, 30, 30, 23, 24, 17, 21, 28, 42, 31, 27, 29, 9, 7, 28, 34, 13, 31, 23
# ]


# AC_UNI = [
#     23, 11, 29, 20, 27, 24, 22, 12, 28, 11, 29, 30, 22, 10, 13, 20, 24, 30, 11, 23,
# ]

# AC_R_UNI_10per = [
#     16, 8, 21, 7, 22, 17, 10, 11, 25, 44, 11, 14, 7, 17, 28, 13, 17, 25, 25, 16
# ]


# LL_UNI = [
#     31, 29, 24, 27, 43, 38, 17, 21, 
# ]

# LL_R_UNI_10per = [
#     30, 37, 53, 68, 30, 18, 19, 56, 
# ]

# print(np.mean(CP_UNI))
# print(np.mean(CP_R_UNI_10per))

# print(np.mean(AC_UNI))
# print(np.mean(AC_R_UNI_10per))

# print(np.mean(LL_UNI))
# print(np.mean(LL_R_UNI_10per))