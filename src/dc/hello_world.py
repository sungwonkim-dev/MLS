input = int(input())
answer = ""
while (input > 0):
    x = int(input / 8)
    answer = answer + str(x)
    input = input % 8

    if input < 8:
        answer = answer + str(input)
        break

print(answer)
