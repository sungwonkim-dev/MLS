def print_9by9(input):
    input = int(input)
    for i in range(1, 10):
        print(input, " * ", i, " = ", i * input)


if __name__ == '__main__':
    print("구구단")
    input = input()
    print_9by9(input)

