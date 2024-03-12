# DEFT2013 Tâche 2 : NOMEQUIPE (optionnel)

NOM Prenom - NOM Prenom

## Description de la tâche

	1 ou 2 exemples de documents (avec leur identifiant)

## Statistiques corpus

	Nombre de document de train/dev/test
	Répartition des étiquettes dans chacun des sous-ensemble

## Méthodes proposées

### Run1: baseline (méthode de référence)

	Description de la méthode:
	- descripteurs utilisés
	- classifieur utilisé

### Run1: Word2Vec_MLP
### Run3: NOMMETHODE
### Run4: NOMMETHODE (pour aller plus loin)

## Résultats

| Run      | f1 Score |
| -------- | --------:|
| baseline |  21,1 |
| METH 1   |  80,0|
| distilbert-ft-recette-3120   | 85,7  |
| distilbert-ft-ingredients-3744   | 83,0  |	
| METH 4   |   |

### Analyse de résultats
	
	Pistes d'analyse:
	* Combien de documents ont un score de 0 ? de 0.5 ? de 1 ? (Courbe ROC)
	* Y-a-t-il des régularités dans les document bien/mal classifiés ?
	* Où est-ce que l'approche se trompe ? (matrice de confusion)
	* Si votre méthode le permet: quels sont les descripteurs les plus décisifs ?