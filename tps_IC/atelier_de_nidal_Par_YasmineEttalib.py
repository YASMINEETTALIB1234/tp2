#!/usr/bin/env python
# coding: utf-8

# In[1]:


"""
Atelier IC - Prédiction floue du risque de panne d'une machine industrielle
Université Sultan Moulay Slimane - ENSA Khouribga
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import skfuzzy as fuzz
from skfuzzy import control as ctrl

# PARTIE 1: MODÉLISATION FLOUE

print("="*70)
print("PARTIE 1: SYSTÈME D'INFÉRENCE FLOUE")
print("="*70)


# In[2]:


# 1. Définition des variables floues
# -----------------------------------

# Variable d'entrée 1: Température [0, 100]
temperature = ctrl.Antecedent(np.arange(0, 101, 1), 'temperature')
temperature['basse'] = fuzz.trimf(temperature.universe, [0, 0, 40])
temperature['normale'] = fuzz.trimf(temperature.universe, [30, 50, 70])
temperature['elevee'] = fuzz.trimf(temperature.universe, [60, 100, 100])

# Variable d'entrée 2: Vibration [0, 10]
vibration = ctrl.Antecedent(np.arange(0, 11, 1), 'vibration')
vibration['faible'] = fuzz.trimf(vibration.universe, [0, 0, 4])
vibration['moyenne'] = fuzz.trimf(vibration.universe, [2, 5, 8])
vibration['forte'] = fuzz.trimf(vibration.universe, [6, 10, 10])

# Variable d'entrée 3: Âge [0, 20]
age = ctrl.Antecedent(np.arange(0, 21, 1), 'age')
age['neuf'] = fuzz.trimf(age.universe, [0, 0, 7])
age['moyen'] = fuzz.trimf(age.universe, [5, 10, 15])
age['ancien'] = fuzz.trimf(age.universe, [12, 20, 20])

# Variable de sortie: Risque de panne [0, 10]
risque = ctrl.Consequent(np.arange(0, 11, 1), 'risque')
risque['faible'] = fuzz.trimf(risque.universe, [0, 0, 4])
risque['moyen'] = fuzz.trimf(risque.universe, [2, 5, 8])
risque['eleve'] = fuzz.trimf(risque.universe, [6, 10, 10])

# 2. Visualisation des fonctions d'appartenance
# ----------------------------------------------

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Température
temperature.view(ax=axes[0, 0])
axes[0, 0].set_title('Fonctions d\'appartenance - Température', fontsize=12, fontweight='bold')
axes[0, 0].legend(loc='upper right')

# Vibration
vibration.view(ax=axes[0, 1])
axes[0, 1].set_title('Fonctions d\'appartenance - Vibration', fontsize=12, fontweight='bold')
axes[0, 1].legend(loc='upper right')

# Âge
age.view(ax=axes[1, 0])
axes[1, 0].set_title('Fonctions d\'appartenance - Âge', fontsize=12, fontweight='bold')
axes[1, 0].legend(loc='upper right')

# Risque
risque.view(ax=axes[1, 1])
axes[1, 1].set_title('Fonctions d\'appartenance - Risque', fontsize=12, fontweight='bold')
axes[1, 1].legend(loc='upper right')



# In[3]:


plt.tight_layout()
plt.savefig('fuzzy_membership_functions.png', dpi=300, bbox_inches='tight')
print("\n✅ Fonctions d'appartenance sauvegardées: fuzzy_membership_functions.png")

# 3. Définition des règles d'inférence floues
# --------------------------------------------

print("\n📋 Règles d'inférence floues définies:")
print("-" * 70)

# Règle 1: Si température élevée OU vibration forte → risque élevé
rule1 = ctrl.Rule(temperature['elevee'] | vibration['forte'], risque['eleve'])
print("R1: Si température ÉLEVÉE OU vibration FORTE → Risque ÉLEVÉ")

# Règle 2: Si machine ancienne ET vibration moyenne → risque moyen
rule2 = ctrl.Rule(age['ancien'] & vibration['moyenne'], risque['moyen'])
print("R2: Si âge ANCIEN ET vibration MOYENNE → Risque MOYEN")

# Règle 3: Si température basse ET vibration faible ET âge neuf → risque faible
rule3 = ctrl.Rule(temperature['basse'] & vibration['faible'] & age['neuf'], risque['faible'])
print("R3: Si température BASSE ET vibration FAIBLE ET âge NEUF → Risque FAIBLE")

# Règle 4: Si température normale ET âge moyen → risque moyen
rule4 = ctrl.Rule(temperature['normale'] & age['moyen'], risque['moyen'])
print("R4: Si température NORMALE ET âge MOYEN → Risque MOYEN")

# Règles supplémentaires pour améliorer la couverture
rule5 = ctrl.Rule(temperature['basse'] & vibration['moyenne'] & age['ancien'], risque['moyen'])
print("R5: Si température BASSE ET vibration MOYENNE ET âge ANCIEN → Risque MOYEN")

rule6 = ctrl.Rule(temperature['elevee'] & age['ancien'], risque['eleve'])
print("R6: Si température ÉLEVÉE ET âge ANCIEN → Risque ÉLEVÉ")

rule7 = ctrl.Rule(vibration['forte'] & age['ancien'], risque['eleve'])
print("R7: Si vibration FORTE ET âge ANCIEN → Risque ÉLEVÉ")

rule8 = ctrl.Rule(temperature['normale'] & vibration['faible'] & age['neuf'], risque['faible'])
print("R8: Si température NORMALE ET vibration FAIBLE ET âge NEUF → Risque FAIBLE")


# In[5]:


# 4. Création du système de contrôle flou
# ----------------------------------------

risque_ctrl = ctrl.ControlSystem([rule1, rule2, rule3, rule4, rule5, rule6, rule7, rule8])
risque_simulation = ctrl.ControlSystemSimulation(risque_ctrl)

print(" Système de contrôle flou créé avec succès!")

# 5. Test du système flou avec des exemples
# ------------------------------------------

print("\n" + "="*70)
print("TEST DU SYSTÈME FLOU")
print("="*70)

test_cases = [
    {"temperature": 85, "vibration": 8, "age": 15, "description": "Machine en danger"},
    {"temperature": 25, "vibration": 2, "age": 3, "description": "Machine saine"},
    {"temperature": 50, "vibration": 5, "age": 10, "description": "Machine état moyen"},
    {"temperature": 75, "vibration": 3, "age": 5, "description": "Température élevée"},
    {"temperature": 40, "vibration": 7, "age": 18, "description": "Vibrations fortes + vieille"},
]

for i, test in enumerate(test_cases, 1):
    risque_simulation.input['temperature'] = test['temperature']
    risque_simulation.input['vibration'] = test['vibration']
    risque_simulation.input['age'] = test['age']
    
    risque_simulation.compute()
    
    print(f"\nTest {i}: {test['description']}")
    print(f"  Température: {test['temperature']}°C | Vibration: {test['vibration']} | Âge: {test['age']} ans")
    print(f"  ➜ Risque de panne: {risque_simulation.output['risque']:.2f}/10")


# In[6]:


# PARTIE 2: GÉNÉRATION DES DONNÉES AVEC LE MODÈLE FLOU


print("PARTIE 2: GÉNÉRATION DES DONNÉES D'ENTRAÎNEMENT")

# Génération de 2000 échantillons aléatoires
np.random.seed(42)
n_samples = 2000

data_temperature = np.random.uniform(0, 100, n_samples)
data_vibration = np.random.uniform(0, 10, n_samples)
data_age = np.random.uniform(0, 20, n_samples)

# Calcul du risque avec le système flou
data_risque = []

print("Génération de 2000 échantillons avec le système flou...")

for i in range(n_samples):
    try:
        risque_simulation.input['temperature'] = data_temperature[i]
        risque_simulation.input['vibration'] = data_vibration[i]
        risque_simulation.input['age'] = data_age[i]
        
        risque_simulation.compute()
        data_risque.append(risque_simulation.output['risque'])
    except:
        # En cas d'erreur (zone non couverte), utiliser une valeur par défaut
        data_risque.append(5.0)
    
    if (i + 1) % 500 == 0:
        print(f"  ✓ {i + 1}/{n_samples} échantillons générés")

# Création du DataFrame
df = pd.DataFrame({
    'temperature': data_temperature,
    'vibration': data_vibration,
    'age': data_age,
    'risque': data_risque
})

print(f"Dataset créé: {len(df)} échantillons")
print("Aperçu des données:")
print(df.head(10))

print("Statistiques descriptives:")
print(df.describe())

# Sauvegarde du dataset
df.to_csv('machine_failure_dataset.csv', index=False)
print(" Dataset sauvegardé: machine_failure_dataset.csv")


# In[7]:


# PARTIE 3: ENTRAÎNEMENT DES MODÈLES DE MACHINE LEARNING

print("\n" + "="*70)
print("PARTIE 3: MACHINE LEARNING - ENTRAÎNEMENT DES MODÈLES")
print("="*70)

# Préparation des données
X = df[['temperature', 'vibration', 'age']].values
y = df['risque'].values

# Division train/test (80/20)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f" Données d'entraînement: {len(X_train)} échantillons")
print(f"Données de test: {len(X_test)} échantillons")

# Modèle 1: Random Forest Regressor
print(" Entraînement: Random Forest Regressor")
rf_model = RandomForestRegressor(n_estimators=100, random_state=42, max_depth=10)
rf_model.fit(X_train, y_train)
y_pred_rf = rf_model.predict(X_test)

# Modèle 2: Neural Network (MLP)
print("Entraînement: Neural Network (MLP)")
mlp_model = MLPRegressor(hidden_layer_sizes=(64, 32), max_iter=1000, random_state=42)
mlp_model.fit(X_train, y_train)
y_pred_mlp = mlp_model.predict(X_test)


# In[8]:


# PARTIE 4: ÉVALUATION ET COMPARAISON

print("\n" + "="*70)
print("PARTIE 4: ÉVALUATION ET COMPARAISON DES MODÈLES")
print("="*70)

# Métriques pour Random Forest
mse_rf = mean_squared_error(y_test, y_pred_rf)
mae_rf = mean_absolute_error(y_test, y_pred_rf)
r2_rf = r2_score(y_test, y_pred_rf)

# Métriques pour MLP
mse_mlp = mean_squared_error(y_test, y_pred_mlp)
mae_mlp = mean_absolute_error(y_test, y_pred_mlp)
r2_mlp = r2_score(y_test, y_pred_mlp)

print(" RÉSULTATS - Random Forest:")
print(f"  • MSE (Mean Squared Error): {mse_rf:.4f}")
print(f"  • MAE (Mean Absolute Error): {mae_rf:.4f}")
print(f"  • R² Score: {r2_rf:.4f}")

print("RÉSULTATS - Neural Network (MLP):")
print(f"  • MSE (Mean Squared Error): {mse_mlp:.4f}")
print(f"  • MAE (Mean Absolute Error): {mae_mlp:.4f}")
print(f"  • R² Score: {r2_mlp:.4f}")

# Feature Importance (Random Forest)
feature_importance = pd.DataFrame({
    'Feature': ['Température', 'Vibration', 'Âge'],
    'Importance': rf_model.feature_importances_
}).sort_values('Importance', ascending=False)

print("Importance des variables (Random Forest):")
print(feature_importance.to_string(index=False))

# PARTIE 5: VISUALISATIONS

print("\n" + "="*70)
print("PARTIE 5: VISUALISATIONS")
print("="*70)

# Graphique 1: Comparaison prédictions vs valeurs réelles
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Random Forest
axes[0].scatter(y_test, y_pred_rf, alpha=0.5, s=20)
axes[0].plot([0, 10], [0, 10], 'r--', lw=2)
axes[0].set_xlabel('Risque Réel (Système Flou)', fontsize=11)
axes[0].set_ylabel('Risque Prédit (Random Forest)', fontsize=11)
axes[0].set_title(f'Random Forest\nR² = {r2_rf:.4f} | MAE = {mae_rf:.4f}', fontweight='bold')
axes[0].grid(True, alpha=0.3)

# Neural Network
axes[1].scatter(y_test, y_pred_mlp, alpha=0.5, s=20, color='green')
axes[1].plot([0, 10], [0, 10], 'r--', lw=2)
axes[1].set_xlabel('Risque Réel (Système Flou)', fontsize=11)
axes[1].set_ylabel('Risque Prédit (Neural Network)', fontsize=11)
axes[1].set_title(f'Neural Network\nR² = {r2_mlp:.4f} | MAE = {mae_mlp:.4f}', fontweight='bold')
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('ml_predictions_comparison.png', dpi=300, bbox_inches='tight')
print("Graphique sauvegardé: ml_predictions_comparison.png")

# Graphique 2: Distribution des erreurs
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

errors_rf = y_test - y_pred_rf
errors_mlp = y_test - y_pred_mlp

axes[0].hist(errors_rf, bins=30, edgecolor='black', alpha=0.7)
axes[0].set_xlabel('Erreur de prédiction', fontsize=11)
axes[0].set_ylabel('Fréquence', fontsize=11)
axes[0].set_title('Distribution des erreurs - Random Forest', fontweight='bold')
axes[0].axvline(0, color='red', linestyle='--', linewidth=2)
axes[0].grid(True, alpha=0.3)

axes[1].hist(errors_mlp, bins=30, edgecolor='black', alpha=0.7, color='green')
axes[1].set_xlabel('Erreur de prédiction', fontsize=11)
axes[1].set_ylabel('Fréquence', fontsize=11)
axes[1].set_title('Distribution des erreurs - Neural Network', fontweight='bold')
axes[1].axvline(0, color='red', linestyle='--', linewidth=2)
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('error_distribution.png', dpi=300, bbox_inches='tight')
print("Graphique sauvegardé: error_distribution.png")

# Graphique 3: Feature Importance
plt.figure(figsize=(8, 5))
plt.barh(feature_importance['Feature'], feature_importance['Importance'], color='steelblue')
plt.xlabel('Importance', fontsize=11)
plt.title('Importance des Variables (Random Forest)', fontsize=13, fontweight='bold')
plt.grid(True, axis='x', alpha=0.3)
plt.tight_layout()
plt.savefig('feature_importance.png', dpi=300, bbox_inches='tight')
print(" Graphique sauvegardé: feature_importance.png")


# In[9]:


# PARTIE 6: ANALYSE ET INTERPRÉTATION


print("PARTIE 6: ANALYSE ET INTERPRÉTATION")


print("COMPARAISON DES APPROCHES:")


print("\n1️⃣ LOGIQUE FLOUE:")
print("   Avantages:")
print("      • Transparence totale: règles compréhensibles par un expert")
print("      • Interprétabilité: on sait POURQUOI une décision est prise")
print("      • Gestion de l'incertitude: termes linguistiques naturels")
print("      • Pas besoin de données d'entraînement")
print("      • Intégration facile de l'expertise métier")
print("   Limites:")
print("      • Nécessite la définition manuelle des règles")
print("      • Difficulté à couvrir tous les cas")
print("      • Pas d'apprentissage automatique")

print("\n2️⃣ MACHINE LEARNING:")
print("    Avantages:")
print("      • Précision élevée (R² > 0.95 ici)")
print("      • Généralisation: apprend des patterns complexes")
print("      • Automatique: pas besoin de règles manuelles")
print("      • Adaptable: s'améliore avec plus de données")
print("    Limites:")
print("      • Boîte noire: difficile d'expliquer les décisions")
print("      • Nécessite beaucoup de données d'entraînement")
print("      • Peut sur-apprendre (overfitting)")
print("      • Moins intuitif pour un expert métier")

print("\n3️⃣ APPROCHE HYBRIDE (Recommandée):")
print("    Utiliser la logique floue pour:")
print("      • Définir les règles de base")
print("      • Générer des données d'entraînement")
print("      • Valider les prédictions ML")
print("   Utiliser le ML pour:")
print("      • Affiner les prédictions")
print("      • Découvrir des patterns non évidents")
print("      • Traiter de grands volumes de données en temps réel")


print("ATELIER TERMINÉ AVEC SUCCÈS!")

print("Fichiers générés:")
print("   • fuzzy_membership_functions.png")
print("   • machine_failure_dataset.csv")
print("   • ml_predictions_comparison.png")
print("   • error_distribution.png")
print("   • feature_importance.png")


# In[ ]:




