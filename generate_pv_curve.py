import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import matplotlib.pyplot as plt

# ============================================================================
# PV-Kurven Generator für AEM Simulation
# ============================================================================

# Parameter
start_date = datetime(2025, 6, 15, 0, 0, 0)  # Sommermonat für gute PV
num_days = 3
interval_minutes = 15
max_power = 1500  # Watt

# Sonnenzeiten
sunrise_hour = 7
sunset_hour = 20

# Zeitreihe erstellen
timestamps = []
current_time = start_date
end_time = start_date + timedelta(days=num_days)

while current_time < end_time:
    timestamps.append(current_time)
    current_time += timedelta(minutes=interval_minutes)

# Leistungswerte berechnen
power_values = []

for ts in timestamps:
    hour = ts.hour + ts.minute / 60.0
    
    # Nachts: 0 W
    if hour < sunrise_hour or hour > sunset_hour:
        power = 0.0
    else:
        # Tagesverlauf: Sinus-basierte Kurve
        # Normalisiere Stunde auf 0-1 zwischen Sonnenauf- und Untergang
        day_progress = (hour - sunrise_hour) / (sunset_hour - sunrise_hour)
        
        # Basis-Sinus (Glockenkurve)
        base_power = np.sin(day_progress * np.pi) * max_power
        
        # Zackelige Variation hinzufügen (PV-typisch durch Wolken)
        # Mehrere Frequenzen für realistische Schwankungen
        noise_slow = np.sin(day_progress * np.pi * 3 + ts.day * 1.5) * 0.15  # Langsame Schwankung
        noise_fast = np.sin(day_progress * np.pi * 12 + ts.minute / 15) * 0.08  # Schnelle Schwankung
        noise_random = (np.random.random() - 0.5) * 0.05  # Zufälliges Rauschen
        
        # Kombiniere Variationen
        variation = 1 + noise_slow + noise_fast + noise_random
        
        # Begrenze Variation auf sinnvolle Werte
        variation = np.clip(variation, 0.4, 1.15)
        
        power = base_power * variation
        
        # Stelle sicher, dass Leistung nicht negativ wird
        power = max(0, power)
        
        # Gelegentliche stärkere Einbrüche (Wolken)
        if np.random.random() < 0.05:  # 5% Chance
            power *= np.random.uniform(0.3, 0.7)
    
    power_values.append(round(power, 2))

# DataFrame erstellen
df_pv = pd.DataFrame({
    'timestamp': timestamps,
    'power_W': power_values
})

# CSV speichern
output_path = r"c:\Users\moend\Documents\Promotion\Python\aem_modelling\aem_simulator\pv_load_curve_3days.csv"
df_pv.to_csv(output_path, index=False, sep=',', decimal='.')

print(f"✅ PV-Kurve erstellt: {len(df_pv)} Datenpunkte")
print(f"📁 Gespeichert: {output_path}")
print(f"\nStatistik:")
print(f"  - Zeitraum: {df_pv['timestamp'].min()} bis {df_pv['timestamp'].max()}")
print(f"  - Min Leistung: {df_pv['power_W'].min():.2f} W")
print(f"  - Max Leistung: {df_pv['power_W'].max():.2f} W")
print(f"  - Mittelwert: {df_pv['power_W'].mean():.2f} W")
print(f"  - Gesamtenergie: {df_pv['power_W'].sum() * interval_minutes / 60 / 1000:.2f} kWh")

# ============================================================================
# Visualisierung
# ============================================================================

fig, ax = plt.subplots(figsize=(14, 6))

# Plotte die PV-Kurve
ax.plot(df_pv['timestamp'], df_pv['power_W'], linewidth=1.5, color='orange', alpha=0.8)
ax.fill_between(df_pv['timestamp'], 0, df_pv['power_W'], color='orange', alpha=0.3)

# Formatierung
ax.set_xlabel('Zeit', fontsize=12, fontweight='bold')
ax.set_ylabel('Leistung [W]', fontsize=12, fontweight='bold')
ax.set_title('Generierte PV-Lastkurve (3 Tage, 15-Minuten-Auflösung)', fontsize=14, fontweight='bold')
ax.grid(True, alpha=0.3)
ax.set_ylim(0, max_power * 1.1)

# Rotation der x-Achsen-Labels für bessere Lesbarkeit
plt.xticks(rotation=45, ha='right')

# Markiere Tag/Nacht-Zyklen
for day in range(num_days + 1):
    night_start = start_date + timedelta(days=day, hours=sunset_hour)
    night_end = start_date + timedelta(days=day+1, hours=sunrise_hour)
    ax.axvspan(night_start, night_end, color='gray', alpha=0.1, label='Nacht' if day == 0 else '')

plt.legend()
plt.tight_layout()

# Plot speichern
plot_path = r"c:\Users\moend\Documents\Promotion\Python\aem_modelling\aem_simulator\pv_load_curve_preview.png"
plt.savefig(plot_path, dpi=150, bbox_inches='tight')
print(f"\n📊 Plot gespeichert: {plot_path}")

plt.show()

print("\n✨ Fertig! Die CSV-Datei kann jetzt in der Streamlit-App verwendet werden.")
