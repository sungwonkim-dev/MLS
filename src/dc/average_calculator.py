score = [[49, 80, 20, 100, 80], [43, 60, 85, 30, 90], [49, 82, 48, 50, 100]]

for x in range(len(score)):
    subject_score = score[x]
    subject_sum = 0
    for y in range(len(subject_score)):
        subject_sum += subject_score[y]
    print(subject_sum / len(subject_score))
