import numpy as np
import pandas as pd

# Framingham Risk Score, based on ATP-III
# points evaluation

def trestbps_points(trestbps, sex):
    points = np.zeros_like(trestbps)
    men = (sex == 1)
    women = (sex == 0)

    points[(trestbps >= 140) & (trestbps < 160) & men] = 1
    points[(trestbps >= 160) & men] = 2

    points[(trestbps >= 120) & (trestbps < 130) & women] = 1
    points[(trestbps >= 130) & (trestbps < 140) & women] = 2
    points[(trestbps >= 140) & (trestbps < 160) & women] = 3
    points[(trestbps >= 160) & women] = 4

    return points


def age_points(age, sex):
    result = np.zeros_like(age)
    bins_male = [(20, 34, -9), (35, 39, -4), (40, 44, 0), (45, 49, 3), (50, 54, 6),
                 (55, 59, 8), (60, 64, 10), (65, 69, 11), (70, 74, 12), (75, 120, 13)]
    bins_female = [(20, 34, -7), (35, 39, -3), (40, 44, 0), (45, 49, 3), (50, 54, 6),
                   (55, 59, 8), (60, 64, 10), (65, 69, 12), (70, 74, 14), (75, 120, 16)]

    for low, high, pts in bins_male:
        mask = (sex == 1) & (age >= low) & (age <= high)
        result[mask] = pts
    for low, high, pts in bins_female:
        mask = (sex == 0) & (age >= low) & (age <= high)
        result[mask] = pts
    return result


def cholesterol_points(age, chol, sex):
    result = np.zeros_like(chol)
    age_bins = [(20, 39), (40, 49), (50, 59), (60, 69), (70, 79), (80, 120)]
    points_male = [
        [0, 4, 7, 9, 11],
        [0, 3, 5, 6, 8],
        [0, 2, 3, 4, 5],
        [0, 1, 1, 2, 3],
        [0, 0, 0, 1, 1],
        [0, 0, 0, 1, 1],
    ]
    points_female = [
        [0, 4, 8, 11, 13],
        [0, 3, 6, 8, 10],
        [0, 2, 4, 5, 7],
        [0, 1, 2, 3, 4],
        [0, 1, 1, 2, 2],
        [0, 1, 1, 2, 2],
    ]

    chol_idx = pd.cut(
        chol, bins=[-np.inf, 160, 200, 240, 280, np.inf], labels=False)

    for i, (low, high) in enumerate(age_bins):
        male_mask = (sex == 1) & (age >= low) & (age <= high)
        female_mask = (sex == 0) & (age >= low) & (age <= high)

        for j in range(5):
            result[male_mask & (chol_idx == j)] = points_male[i][j]
            result[female_mask & (chol_idx == j)] = points_female[i][j]
    return result


def diabetes_points(sex, diabetic):
    return np.where(diabetic == 1, np.where(sex == 1, 2, 4), 0)


# calculation total framingham risk score

def calculate_total_frs(age, sex, chol, trestbps, fbs):
    total = 0
    total += age_points(age, sex)
    total += cholesterol_points(age, chol, sex)
    total += trestbps_points(trestbps, sex)
    total += diabetes_points(sex, fbs)
    return total


# Other feature engineering

def calculate_age_squared(age):
    return age**2


def calculate_pain_related(cp, exang):
    return 3 * exang + 4 - cp


def calculate_food_related(chol, fbs):
    return chol * (1 + fbs)/100


def calculate_exercise_related(thalach, trestbps, thal):
    thal_mapped = thal.map({0: 0, 1: 2, 2: 1})
    result = trestbps * (250 - thalach) * (thal_mapped + 1)/10000
    return round(result, 4)


def calculate_ecg_related(restecg, oldpeak, slope):
    result = (restecg + 1) * (oldpeak + 1) * (4 - slope)
    return round(result, 4)


def calculate_dead_cells(ca, thal):
    thal_mapped = thal.map({0: 0, 1: 2, 2: 1})
    return (ca + 1) * (thal_mapped + 1)
