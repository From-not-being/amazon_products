import tensorflow as tf
import numpy as np
import csv

amazon_path = "/content/drive/MyDrive/Colab Notebooks/amazon_products.csv"

def run_amazon_prediction(file_path):
    X = []
    with open(file_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for i, r in enumerate(reader):
            if i > 50: break # Mehr Daten für besseres "Gefühl" des Modells

            try:
                # 1. Datenreinigung
                actual_price = float(r['actual_price'].replace('₹', '').replace(',', ''))

                discount_val = r.get('discount_percentage', '0')
                discount_pct = float(discount_val.replace('%', '')) if discount_val else 0.0

                X.append([actual_price, discount_pct])
            except (ValueError, KeyError):
                continue

    X_np = np.array(X)

    # --- WICHTIG: NORMALISIERUNG ---
    # Wir skalieren die Preise, damit die KI nicht mit 50.000er Werten kämpft
    X_max = X_np.max(axis=0)
    # Add a small epsilon to avoid division by zero if X_max contains zeros
    epsilon = 1e-8
    X_scaled = X_np / (X_max + epsilon)

    # Modell-Architektur (Modern ohne Input-Shape Warning)
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(2,)), # Korrekte moderne Schreibweise
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dense(16, activation='relu'),
        tf.keras.layers.Dense(1, activation='linear') # Linear für Ratings
    ])

    # Explizite Konfiguration
    mein_optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)
    meine_verlust_funktion = tf.keras.losses.MeanSquaredError()

    model.compile(optimizer=mein_optimizer, loss=meine_verlust_funktion)

    # Kurzes Training (ohne Training bleibt es bei Zufallswerten)
    # Wir faken hier ein Ziel-Rating (z.B. 4.0), damit das Modell eine Richtung hat
    y_dummy = np.full((len(X_scaled), 1), 4.0)
    model.fit(X_scaled, y_dummy, epochs=10, verbose=0)

    # Vorhersage
    predictions = model.predict(X_scaled[:5], verbose=0)

    print("\n📦 TENSORFLOW AMAZON RATING-VORHERSAGEN (Normalisiert):")
    for i, pred in enumerate(predictions):
        # Wir begrenzen das Rating logisch auf 1.0 bis 5.0 Sterne
        final_rating = max(1.0, min(5.0, pred[0]))
        print(f"Produkt {i+1}: Geschätztes Rating = {final_rating:.2f} Sterne")

run_amazon_prediction(amazon_path)
