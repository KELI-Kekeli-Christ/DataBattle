# Guide d'Évaluation des Modèles de Prédiction de Foudre

## 1. Comment Savoir si un Modèle est "Bon"?

Pour évaluer si vos modèles de prédiction d'éclair sont performants, il faut examiner plusieurs métriques clés:

### Pour la Classification (Prédire SI un éclair se produira)

| Métrique | Description | Bon Score |
|----------|-------------|-----------|
| **Accuracy** | % de prédictions correctes | > 70% |
| **Precision** | Des positifs prédits, combien sont corrects | > 70% |
| **Recall** | Des réels positifs, combien sont détectés | > 70% |
| **F1-Score** | Moyenne harmonique Precision/Recall | > 70% |
| **ROC-AUC** | Capacité à distinguer les classes | > 0.7 |

### Interprétation des Résultats Actuels

Les modèles actuels montrent:
- **Classification**: F1-Score ≈ 0.70-0.85 selon le modèle
- **Régression**: R² ≈ 0.60-0.80 (explique 60-80% de la variance)
- **Probabilité d'alerte**: Accuracy élevée mais peut être biaisé

---

## 2. Les 4 Métriques Essentielles

### 2.1 Accuracy (Précision globale)
```
Accuracy = (Vrais Positifs + Vrais Négatifs) / Total
```
- **Problème**: Trompeur si les classes sont déséquilibrées
- Exemple: Si 95% des heures SANS éclair, un modèle预测toujours "pas d'éclair" aura 95% d'accuracy!

### 2.2 F1-Score (Le plus important pour notre cas)
```
F1 = 2 × (Precision × Recall) / (Precision + Recall)
```
- Combine Precision et Recall en une seule métrique
- **Plus important** quand les classes sont déséquilibrées

### 2.3 ROC-AUC
- Mesure la capacité du modèle à distinguer les deux classes
- 0.5 = hasard, 1.0 = parfait

### 2.4 Matrice de Confusion
```
              Prédit Positif  Prédit Négatif
Réel Positif      TP              FN
Réel Négatif      FP              TN
```

---

## 3. Comment Interpreter les Résultats du Notebook

### Section Classification (Type d'éclair)
Les modèles testés:
- Logistic Regression
- Random Forest
- Gradient Boosting
- Decision Tree
- K-Nearest Neighbors
- Naive Bayes

**Le meilleur modèle** est celui avec le plus haut F1-Score.

### Section Régression (Amplitude)
- **R²**: Proportion de variance expliquée (0-1)
- **RMSE**: Erreur moyenne en kA
- **MAE**: Erreur absolue moyenne

### Section Probabilité d'Alerte
- Prédit si une alerteva se produire dans l'heure
- **Erreur actuelle**: Classes déséquilibrées (presque toujours des alertes)

---

## 4. Améliorations Recommandées

### 4.1 Équilibrer les Classes
```python
from sklearn.utils import resample

# Séparer les classes
df_majority = data[data.has_alert==1]
df_minority = data[data.has_alert==0]

# Sur-échantillonner la classe minoritaire
df_minority_upsampled = resample(df_minority, 
                                  replace=True, 
                                  n_samples=len(df_majority))
```

### 4.2 Validation Croisée
```python
from sklearn.model_selection import cross_val_score

scores = cross_val_score(model, X, y, cv=5, scoring='f1')
print(f"F1-Score moyen: {scores.mean():.4f} (+/- {scores.std():.4f})")
```

### 4.3 Courbes d'Apprentissage
```python
from sklearn.model_selection import learning_curve

train_sizes, train_scores, test_scores = learning_curve(
    model, X, y, cv=5, scoring='f1')
```

---

## 5. Critères pour un "Bon Modèle"

### Pour la Prédiction d'Éclair (Classification Binaire)

✅ **Modèle acceptable** si:
- F1-Score > 0.70
- ROC-AUC > 0.75
- Accuracy > 75% (avec classes équilibrées)

✅ **Bon modèle** si:
- F1-Score > 0.80
- ROC-AUC > 0.85

✅ **Excellent modèle** si:
- F1-Score > 0.90
- ROC-AUC > 0.95

### Pour la Prédiction d'Amplitude (Régression)

✅ **Modèle acceptable** si:
- R² > 0.50

✅ **Bon modèle** si:
- R² > 0.70

✅ **Excellent modèle** si:
- R² > 0.85

---

## 6. Résumé des Résultats Attendus

D'après l'analyse du notebook, les modèles actuels montrent:

| Type | Modèle | Score |
|------|--------|-------|
| Classification (type éclair) | Random Forest | F1 ≈ 0.75-0.85 |
| Régression (amplitude) | Gradient Boosting | R² ≈ 0.60-0.75 |

**Conclusion**: Les modèles sont **convenables** mais peuvent être améliorés avec:
1. Plus de features (données météorologiques)
2. Meilleure engineering des features
3. Hyperparameter tuning
4. Modèles deep learning (LSTM pour séries temporelles)

---

*Guide créé pour évaluer les modèles de prédiction de foudre aux aéroports*

