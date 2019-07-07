def convert_grade(input):
    age = 2020 - int(input) + 1
    if (age > 19 and age < 25):
        return "대학생"
    if (age > 16 and age <= 19):
        return "고등학생"
    if (age > 13 and age <= 16):
        return "중학생"
    if (age > 7 and age <= 13):
        return "초등학생"
    return "학생이 아닙니다."


if __name__ == '__main__':
    print("출생년도를 입력해주세요")
    birth_year = input()
    print(convert_grade(birth_year))
