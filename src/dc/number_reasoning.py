import random

print("난수 생성 중")
target = random.randint(1, 100)
answer = int(input())

while target is not answer:
    if target < answer:
        print(answer, "보다 작습니다.")
    elif target > answer:
        print(answer, "보다 큽니다.")
    answer = int(input())
else :
    print("정답입니다.")