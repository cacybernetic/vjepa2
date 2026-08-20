"""
Chapitre 6 — La perte de V-JEPA 2, expliquee symbole par symbole.

Trois petits programmes :
  1. la perte de V-JEPA 2, calculee sur deux jetons masques ;
  2. la norme L1 comparee aux carres du chapitre 2 ;
  3. la fabrication d'un masque, et le compte des jetons caches.

Les nombres affiches sont exactement ceux du livre.
"""

import torch


# ---------------------------------------------------------------------
# 1. La perte de V-JEPA 2, sur nos deux jetons
# ---------------------------------------------------------------------
print("=== 1. la perte, sur deux jetons masques ===")

# Deux jetons masques, de trois nombres chacun.
# Ligne 0 : le jeton numero 3. Ligne 1 : le jeton numero 7.
prediction = torch.tensor([[0.5, -0.2, 1.0],
                           [0.0,  0.0, 0.0]])
cible      = torch.tensor([[0.4,  0.1, 0.8],
                           [0.1, -0.2, 0.1]])

# 1. l'ecart, terme a terme, puis on efface les signes
ecarts = (prediction - cible).abs()
print(ecarts)

# 2. la norme L1 : on additionne les nombres de chaque jeton
par_jeton = ecarts.sum(dim=1)
print(par_jeton)

# 3. la moyenne sur les jetons masques : c'est la perte
perte = par_jeton.mean()
print(perte)


# ---------------------------------------------------------------------
# 2. Norme L1 contre carres : qui punit le plus les gros ecarts ?
# ---------------------------------------------------------------------
print("\n=== 2. norme L1 contre carres ===")

ecarts = torch.tensor([0.1, 1.0])     # un petit ecart, et un gros

total_l1     = ecarts.abs().sum()
total_carres = (ecarts ** 2).sum()

print(f"norme L1 : total = {total_l1:.2f}")
print(f"   le petit ecart compte pour {100 * 0.1 / total_l1:.0f} %")
print(f"   le gros  ecart compte pour {100 * 1.0 / total_l1:.0f} %")

print(f"carres   : total = {total_carres:.2f}")
print(f"   le petit ecart compte pour {100 * 0.01 / total_carres:.0f} %")
print(f"   le gros  ecart compte pour {100 * 1.00 / total_carres:.0f} %")


# ---------------------------------------------------------------------
# 3. Le masque : d'abord la taille d'un trou, puis le compte
# ---------------------------------------------------------------------
print("\n=== 3. trois trous, du plus petit au plus grand ===")


def taille_du_trou(echelle, rapport, H=16, W=16):
    """Rend la hauteur et la largeur du rectangle, en cases."""
    aire = echelle * H * W                       # la surface visee
    hauteur = round((aire * rapport) ** 0.5)
    largeur = round((aire / rapport) ** 0.5)
    return min(hauteur, H), min(largeur, W)      # on ne sort pas de la grille


for echelle, rapport in [(0.15, 0.75), (0.473, 1.0), (0.70, 1.5)]:
    h, w = taille_du_trou(echelle, rapport)
    print(f"echelle {echelle:.3f}, rapport {rapport:.2f} -> "
          f"{h:2d} x {w:2d} = {h * w:3d} cases, soit {8 * h * w:4d} jetons")


print("\n=== 4. fabriquer un masque et compter les jetons caches ===")

T, H, W = 8, 16, 16          # tranches de temps, lignes, colonnes

masque = torch.zeros(H, W, dtype=torch.bool)   # au depart, rien n'est cache
masque[2:13, 2:13] = True                      # premier bloc : 11 x 11
masque[9:16, 10:16] = True                     # second bloc  :  7 x  6

par_tranche = int(masque.sum())      # la reunion des deux blocs
total = par_tranche * T              # le trou traverse tout le clip

print(f"cases cachees sur une tranche : {par_tranche}")
print(f"jetons masques  (|M|)         : {total}")
print(f"jetons visibles (le contexte) : {T * H * W - total}")
print(f"part jetee par la perte       : "
      f"{100 * (T * H * W - total) / (T * H * W):.1f} %")
