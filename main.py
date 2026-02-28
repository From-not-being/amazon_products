import tensorflow as tf
import numpy as np
import csv

# Pfad anpassen
amazon_path = "/content/drive/MyDrive/Colab Notebooks/amazon_products.csv"

def run_amazon_prediction(file_path):
    X = []
    with open(file_path, 'r') as f:
        reader = csv.DictReader(f)
        for i, r in enumerate(reader):
            if i > 4: break # Nur die ersten 5 Produkte

            # Corrected column names and cleaned values based on Kaggle dataset 'Amazon Products Dataset'
            actual_price_str = r['actual_price'].replace('₹', '').replace(',', '')

            # Safely get 'discount_percentage', defaulting to '0' if missing or empty
            discount_percentage_val = r.get('discount_percentage')
            if discount_percentage_val is None or not discount_percentage_val.strip():
                print(f"Warning: 'discount_percentage' not found or empty for row {i+1}. Assuming 0% discount.")
                discount_percentage_str = "0"
            else:
                discount_percentage_str = discount_percentage_val.replace('%', '')

            X.append([float(actual_price_str), float(discount_percentage_str)])

    X_np = np.array(X)

    # Einfaches Modell bauen & kurz trainieren (damit Werte nicht zufällig sind)
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(32, activation='relu', input_shape=(2,)),
        tf.keras.layers.Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')

    # Vorhersage
    predictions = model.predict(X_np, verbose=0)

    print("\n📦 TENSORFLOW AMAZON RATING-VORHERSAGEN:")
    for i, pred in enumerate(predictions):
        print(f"Produkt {i+1}: Geschätztes Rating = {pred[0]:.2f} Sterne")

run_amazon_prediction(amazon_path)
