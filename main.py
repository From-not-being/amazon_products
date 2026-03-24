import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import os

# Pfad zu deiner Amazon CSV
path = "/content/drive/MyDrive/Colab Notebooks/amazon_products.csv"

def pure_tensorflow_amazon(file_path):
    print("--- Starte reine TensorFlow Engine ---\n")
    
    # 1. DATEN LADEN
    if not os.path.exists(file_path):
        print(f"Fehler: Datei nicht gefunden unter {file_path}")
        return
    
    df = pd.read_csv(file_path)

    # --- 2. SPALTENNAMEN & BEREINIGUNG ---
    features = ['ratings', 'no_of_ratings'] 
    target = 'discount_price'

    df = df.dropna(subset=features + [target])
    
    for col in features + [target]:
        if df[col].dtype == object:
            # Bereinigt Währungszeichen und konvertiert zu Zahlen
            df[col] = df[col].astype(str).str.replace('[^\d.]', '', regex=True)
            df[col] = pd.to_numeric(df[col], errors='coerce')
            
    df = df.dropna(subset=features + [target])
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)

    X = df[features].values.astype('float32')
    y = df[target].values.astype('float32')

    # --- 3. MANUELLER TRAIN-TEST-SPLIT ---
    split_index = int(len(X) * 0.8)
    X_train, X_test = X[:split_index], X[split_index:]
    y_train, y_test = y[:split_index], y[split_index:]

    # --- 4. TENSORFLOW NORMALIZATION LAYER ---
    normalizer = tf.keras.layers.Normalization(axis=-1)
    normalizer.adapt(X_train)

    # --- 5. MODELL-ARCHITEKTUR ---
    model = tf.keras.Sequential([
        normalizer,
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dense(16, activation='relu'),
        tf.keras.layers.Dense(1)
    ])

    model.compile(optimizer='adam', loss='mae')

    # --- 6. TRAINING MIT FORTSCHRITTSANZEIGE ---
    print(f"Training läuft für {len(X_train)} Datensätze...")
    
    # history speichert den Verlauf für matplotlib
    # verbose=1 zeigt den animierten Fortschrittsbalken
    history = model.fit(
        X_train, y_train, 
        validation_split=0.2,
        epochs=50, 
        batch_size=32, 
        verbose=1 
    )

    # --- 7. VISUALISIERUNG MIT MATPLOTLIB ---
    plt.figure(figsize=(10, 6))
    plt.plot(history.history['loss'], label='Training Loss (MAE)')
    plt.plot(history.history['val_loss'], label='Validation Loss (MAE)')
    plt.title('Modell-Lernkurve (Amazon Price Prediction)')
    plt.xlabel('Epochen')
    plt.ylabel('Mittlerer absoluter Fehler (MAE)')
    plt.legend()
    plt.grid(True)
    plt.show()

    # --- 8. PERMUTATION IMPORTANCE ---
    print("\n--- Analyse der Feature-Gewichtung ---\n")
    baseline_preds = model.predict(X_test, verbose=0).flatten()
    baseline_mae = np.mean(np.abs(y_test - baseline_preds))

    impacts = {}
    for i, feature in enumerate(features):
        X_test_shuffled = X_test.copy()
        np.random.shuffle(X_test_shuffled[:, i])

        shuffled_preds = model.predict(X_test_shuffled, verbose=0).flatten()
        shuffled_mae = np.mean(np.abs(y_test - shuffled_preds))
        impacts[feature] = abs(shuffled_mae - baseline_mae)

    total_impact = sum(impacts.values())

    if total_impact > 0:
        for feature, impact in impacts.items():
            pct = (impact / total_impact) * 100
            print(f"-> {feature}: {pct:.2f} % Einfluss auf den Preis")
    else:
        print("Das Modell konnte keine klaren Abhängigkeiten feststellen.")

# Ausführen
pure_tensorflow_amazon(path)
