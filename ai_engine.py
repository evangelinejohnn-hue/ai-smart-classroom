import numpy as np

def predict(emotion):

    base = {
        "happy": (90, 20),
        "neutral": (60, 40),
        "sad": (40, 70),
        "angry": (30, 90),
        "fear": (20, 80)
    }

    focus, stress = base.get(emotion, (50, 50))

    noise = np.random.uniform(-2, 2)

    focus = max(0, min(100, focus + noise))
    stress = max(0, min(100, stress + noise))

    score = (focus * 0.7) - (stress * 0.3)

    return round(focus, 2), round(stress, 2), round(score, 2)