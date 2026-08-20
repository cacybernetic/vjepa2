"""
Chapitre 5 — L'idee JEPA.

Trois petits programmes :
  1. la moyenne mobile du professeur, calculee pas a pas ;
  2. un entrainement SANS garde-fou : il s'effondre ;
  3. le meme entrainement AVEC les deux garde-fous : il tient.
"""

import torch
import torch.nn as nn

torch.manual_seed(0)


# ---------------------------------------------------------------
# 1. La moyenne mobile exponentielle, a la main
# ---------------------------------------------------------------
def demo_ema():
    m = 0.99925
    prof, eleve = 1.0, 2.0
    print("--- 1. La moyenne mobile ---")
    for pas in range(1, 4):
        prof = m * prof + (1 - m) * eleve
        print(f"pas {pas} : professeur = {prof:.7f}   ecart = {eleve - prof:.7f}")
    for pas in range(4, 5001):
        prof = m * prof + (1 - m) * eleve
        if pas in (924, 1000, 5000):
            print(f"pas {pas} : professeur = {prof:.4f}   ecart = {eleve - prof:.4f}")


# ---------------------------------------------------------------
# Les trois pieces, minuscules
# ---------------------------------------------------------------
def fabrique_encodeur():
    return nn.Sequential(nn.Linear(8, 16), nn.GELU(), nn.Linear(16, 16))


def fabrique_predicteur():
    return nn.Sequential(nn.Linear(16, 16), nn.GELU(), nn.Linear(16, 16))


def diversite(vecteurs):
    """Mesure si les sorties se ressemblent toutes. Proche de 0 = effondrement."""
    return vecteurs.std(dim=0).mean().item()


# ---------------------------------------------------------------
# 2. Sans garde-fou : le meme encodeur des deux cotes, aucun blocage
# ---------------------------------------------------------------
def sans_garde_fou(pas_total=300):
    encodeur, predicteur = fabrique_encodeur(), fabrique_predicteur()
    params = list(encodeur.parameters()) + list(predicteur.parameters())
    optim = torch.optim.Adam(params, lr=1e-3)

    x = torch.randn(64, 8)
    y = x + 0.1 * torch.randn(64, 8)

    print("\n--- 2. Sans garde-fou ---")
    for pas in range(1, pas_total + 1):
        cible = encodeur(y)                       # PAS de stop-gradient
        prediction = predicteur(encodeur(x))
        perte = (prediction - cible).abs().mean()
        optim.zero_grad()
        perte.backward()
        optim.step()
        if pas % 100 == 0 or pas == 1:
            print(f"pas {pas:4d} : perte = {perte.item():.6f}"
                  f"   diversite = {diversite(cible):.6f}")


# ---------------------------------------------------------------
# 3. Avec les deux garde-fous : professeur separe, gele, mis a jour en EMA
# ---------------------------------------------------------------
def avec_garde_fous(pas_total=300, m=0.996):
    eleve, predicteur = fabrique_encodeur(), fabrique_predicteur()
    professeur = fabrique_encodeur()
    professeur.load_state_dict(eleve.state_dict())
    for p in professeur.parameters():
        p.requires_grad_(False)

    params = list(eleve.parameters()) + list(predicteur.parameters())
    optim = torch.optim.Adam(params, lr=1e-3)

    x = torch.randn(64, 8)
    y = x + 0.1 * torch.randn(64, 8)

    print("\n--- 3. Avec les deux garde-fous ---")
    for pas in range(1, pas_total + 1):
        with torch.no_grad():                     # le gradient s'arrete ici
            cible = professeur(y)
        prediction = predicteur(eleve(x))
        perte = (prediction - cible).abs().mean()
        optim.zero_grad()
        perte.backward()
        optim.step()

        with torch.no_grad():                     # le professeur suit de loin
            for pp, pe in zip(professeur.parameters(), eleve.parameters()):
                pp.mul_(m).add_(pe, alpha=1 - m)

        if pas % 100 == 0 or pas == 1:
            print(f"pas {pas:4d} : perte = {perte.item():.6f}"
                  f"   diversite = {diversite(cible):.6f}")


if __name__ == "__main__":
    demo_ema()
    sans_garde_fou()
    avec_garde_fous()
