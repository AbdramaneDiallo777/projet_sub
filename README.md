# Scoring de risque explicable et robuste

Projet de scoring de risque crédit avec explicabilité (SHAP) et équité (Fairlearn).

## Structure du projet

```
scoring_credit/
├── data/               # Place ici cs-training.csv et cs-test.csv
├── notebooks/          # Notebooks d'exploration libre
├── src/                # Scripts Python organisés par étape
│   └── 01_exploration.py
├── reports/            # Graphiques et rapports générés
└── requirements.txt
```

## Démarrage

1. **Installer les dépendances**
   ```bash
   pip install -r requirements.txt
   ```

2. **Placer le dataset**
   Télécharge `cs-training.csv` depuis Kaggle ("Give Me Some Credit :: 2011
   Competition Data") et place-le dans `data/`.

3. **Lancer l'exploration**
   ```bash
   cd scoring_credit
   python src/01_exploration.py
   ```
   Cela affiche un diagnostic complet du dataset (valeurs manquantes,
   déséquilibre de classes, outliers) et génère des graphiques dans `reports/`.

## Feuille de route

- [x] Étape 1 — Exploration & nettoyage
- [ ] Étape 2 — Feature engineering (traitement des manquants, SMOTE)
- [ ] Étape 3 — Modélisation (régression logistique, XGBoost)
- [ ] Étape 4 — Explicabilité (SHAP)
- [ ] Étape 5 — Équité / robustesse (Fairlearn)
- [ ] Étape 6 — API FastAPI + rapport final

## Colonnes du dataset

| Colonne | Description |
|---|---|
| `SeriousDlqin2yrs` | Target (1 = défaut de paiement dans les 2 ans) |
| `RevolvingUtilizationOfUnsecuredLines` | Taux d'utilisation du crédit renouvelable |
| `age` | Âge du client (⚠️ variable sensible, à surveiller pour l'équité) |
| `DebtRatio` | Ratio d'endettement |
| `MonthlyIncome` | Revenu mensuel (beaucoup de valeurs manquantes) |
| `NumberOfDependents` | Nombre de personnes à charge |
