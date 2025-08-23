# dataset.py

import pandas as pd
import random

# ========================= CONFIG =========================
COLUMNS = ["Temperature", "Humidity", "Moisture", "Nitrogen", "Potassium", "Phosphorous", "SDI", "Level"]
IDEAL_VALUES = {"Temperature": 25, "Humidity": 60, "Moisture": 50, "Nitrogen": 50, "Potassium": 35, "Phosphorous": 20}
MAX_DEV = {"Temperature": 15, "Humidity": 40, "Moisture": 40, "Nitrogen": 50, "Potassium": 30, "Phosphorous": 20}
ROWS_PER_CLASS = 2000
RANGES = {"Healthy": (0, 10), "Low": (10, 30), "Moderate": (30, 70), "High": (70, 100)}

# ========================= FUNCTION =========================
def generate_row_by_sdi(target_range):
    while True:
        temp = random.randint(20, 40)
        hum = round(random.uniform(20, 80), 2)
        moist = round(random.uniform(10, 70), 2)
        n = round(random.uniform(5, 60), 2)
        k = round(random.uniform(5, 50), 2)
        p = round(random.uniform(1, 30), 2)

        # SDI calculation
        temp_score = abs(temp - IDEAL_VALUES["Temperature"]) / MAX_DEV["Temperature"]
        hum_score = abs(hum - IDEAL_VALUES["Humidity"]) / MAX_DEV["Humidity"]
        moist_score = abs(moist - IDEAL_VALUES["Moisture"]) / MAX_DEV["Moisture"]
        n_score = abs(n - IDEAL_VALUES["Nitrogen"]) / MAX_DEV["Nitrogen"]
        k_score = abs(k - IDEAL_VALUES["Potassium"]) / MAX_DEV["Potassium"]
        p_score = abs(p - IDEAL_VALUES["Phosphorous"]) / MAX_DEV["Phosphorous"]

        sdi = (
            0.2 * temp_score +
            0.2 * hum_score +
            0.2 * moist_score +
            0.15 * (n_score ** 1.5) +
            0.15 * (k_score ** 1.5) +
            0.1 * (p_score ** 1.5)
        ) * 100
        sdi = round(min(max(sdi, 0), 100), 2)

        if target_range[0] <= sdi <= target_range[1]:
            if sdi <= 10:
                level = "Healthy"
            elif sdi <= 30:
                level = "Low"
            elif sdi <= 70:
                level = "Moderate"
            else:
                level = "High"
            return [temp, hum, moist, n, k, p, sdi, level]

# ========================= DATASET GENERATION =========================
def generate_dataset():
    dataset = []
    for level, sdi_range in RANGES.items():
        for _ in range(ROWS_PER_CLASS):
            dataset.append(generate_row_by_sdi(sdi_range))
    df = pd.DataFrame(dataset, columns=COLUMNS)
    df = df.sample(frac=1).reset_index(drop=True)
    df.to_csv("soil_dataset.csv", index=False)
    print("✅ Dataset generated.")
    return df

# ========================= MAIN =========================
if __name__ == "__main__":
    generate_dataset()
