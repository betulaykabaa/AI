import matplotlib.pyplot as plt

# Data
study_hours = [1, 2, 3, 4, 5, 6, 7, 8]
exam_scores = [20, 30, 45, 50, 60, 75, 85, 95]

plt.figure(figsize=(8, 5))

# DRAWING THE SCATTER PLOT
# x-axis: Study Hours
# y-axis: Exam Scores
plt.scatter(study_hours, exam_scores, color='blue', s=100, label='Students')

plt.title("Relationship Between Study Hours and Exam Score")
plt.xlabel("Study Hours")
plt.ylabel("Exam Score")
plt.grid(True, linestyle='--', alpha=0.5)
plt.legend()
plt.show()