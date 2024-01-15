diagonal = 0
while 0 < 1:
    try:
        diagonal = int(input("Screen diagonal \n"))
        if isinstance(diagonal, int) == True:
            break
    except ValueError: continue

length = 0
width = 0