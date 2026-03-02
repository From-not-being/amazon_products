import pandas as pd
import numpy as np
import tensorflow as tf

amazon_products_path = "/content/drive/MyDrive/Colab Notebooks/amazon_products.csv"
df_amazon = pd.read_csv(amazon_products_path)

display(df_amazon.head(3))
print(df_amazon.columns)

# Pfad zu deiner Amazon CSV
path = "/content/drive/MyDrive/Colab Notebooks/amazon_products.csv"

def pure_tensorflow_amazon(file_path):
    print("Starte reine TensorFlow Engine (Zero Scikit-Learn)...\n")
    df = pd.read_csv(file_path)

    # --- 1. SPALTENNAMEN ---
    features = ['ratings', 'no_of_ratings'] # Changed 'rating' to 'ratings' and 'rating_count' to 'no_of_ratings' to match df_amazon columns
    target = 'discount_price' # Changed 'discounted_price' to 'discount_price'

    # --- 2. AMAZON DATEN BEREINIGEN (Währungen filtern) ---
    df = df.dropna(subset=features + [target])
    for col in features + [target]:
        if df[col].dtype == object:
            # Removed the r'' regex, using a simple string replacement as regex=True might not be needed for simple char removal
            df[col] = df[col].astype(str).str.replace('[^\d.]', '', regex=True)
            df[col] = pd.to_numeric(df[col], errors='coerce')
    df = df.dropna(subset=features + [target])

    # Daten mischen (Shuffle), um eine faire Aufteilung zu garantieren
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)

    X = df[features].values
    y = df[target].values

    # --- 3. MANUELLER TRAIN-TEST-SPLIT (Ohne Scikit-Learn!) ---
    # Wir schneiden das Array bei 80% einfach durch
    split_index = int(len(X) * 0.8)
    X_train, X_test = X[:split_index], X[split_index:]
    y_train, y_test = y[:split_index], y[split_index:]

    # --- 4. TENSORFLOW NORMALIZATION LAYER ---
    # Statt einem externen Scaler bauen wir die Skalierung DIREKT ins Netz ein.
    # Das Netz lernt im nächsten Schritt die Mittelwerte und Varianzen der Daten.
    normalizer = tf.keras.layers.Normalization(axis=-1)
    normalizer.adapt(X_train)

    # --- 5. NEURONALES NETZ BAUEN ---
    model = tf.keras.Sequential([
        normalizer,                                                 # Layer 1: Skaliert die Daten intern
        tf.keras.layers.Dense(16, activation='relu'),               # Layer 2: Versteckte Schicht
        tf.keras.layers.Dense(8, activation='relu'),                # Layer 3: Versteckte Schicht
        tf.keras.layers.Dense(1)                                    # Layer 4: Output (Der Preis)
    ])

    model.compile(optimizer='adam', loss='mae')

    print("Trainiere das Modell...")
    model.fit(X_train, y_train, epochs=20, batch_size=32, verbose=0)

    # --- 6. PERMUTATION IMPORTANCE (Ohne Scikit-Learn Metriken!) ---
    print("Führe Test durch, um Prozent-Gewichtung zu berechnen...\n")

    # Baseline-Fehler manuell berechnen (ohne sklearn's mean_absolute_error)
    baseline_preds = model.predict(X_test, verbose=0).flatten()
    baseline_mae = np.mean(np.abs(y_test - baseline_preds))

    impacts = {}

    for i, feature in enumerate(features):
        X_test_shuffled = X_test.copy()
        np.random.shuffle(X_test_shuffled[:, i]) # Zieht den "Stecker" für diese Spalte

        shuffled_preds = model.predict(X_test_shuffled, verbose=0).flatten()
        shuffled_mae = np.mean(np.abs(y_test - shuffled_preds))

        impacts[feature] = abs(shuffled_mae - baseline_mae)

    # In Prozente umrechnen
    total_impact = sum(impacts.values())

    print("=== Was treibt den Discount-Preis laut TensorFlow? ===")
    if total_impact > 0:
        for feature, impact in impacts.items():
            pct = (impact / total_impact) * 100
            print(f"-> {feature}: {pct:.2f} % Wirkung")
    else:
        print("Das Modell hat keinen signifikanten Zusammenhang gefunden.")

pure_tensorflow_amazon(path)
