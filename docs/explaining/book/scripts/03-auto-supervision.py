"""Le programme complet du chapitre 3 : « Apprendre sans etiquettes ».

Il montre les deux temps de l'auto-supervision sur un monde miniature.

  1. Le pre-entrainement. Chaque image de 8x8 contient une barre, couchee ou
     debout. On coupe l'image en deux : on donne la moitie gauche, on cache la
     moitie droite, et on demande de la deviner. Aucune etiquette n'est lue.

  2. L'evaluation. On gele l'encodeur, on branche derriere lui une sonde d'une
     seule couche, entrainee avec vingt etiquettes seulement, et on compte les
     points sur mille images jamais vues.

Lancement :  python 03-auto-supervision.py
Sortie attendue (graine fixee, PyTorch 2.8) :

    etape   1 : erreur = 0.1648
    etape 100 : erreur = 0.0487
    etape 300 : erreur = 0.0483
    etape 600 : erreur = 0.0483
    moitie gauche vide          : 25.8 %
    sonde sur la representation : 80.4 %
    sonde sur les pixels bruts  : 61.8 %
"""

import torch

torch.manual_seed(0)          # pour que vous obteniez les memes nombres


# ---------------------------------------------------------------------
# 1. Le monde miniature
# ---------------------------------------------------------------------
def fabrique_images(n):
    """n images de 8x8 : soit une barre couchee, soit une barre debout."""
    images = torch.zeros(n, 8, 8)
    genre = torch.randint(0, 2, (n,))    # 0 = couchee, 1 = debout
    place = torch.randint(0, 8, (n,))    # a quelle ligne, ou a quelle colonne
    for i in range(n):
        if genre[i] == 0:
            images[i, place[i], :] = 1.0     # une ligne entiere
        else:
            images[i, :, place[i]] = 1.0     # une colonne entiere
    return images, genre.float()


images, genre = fabrique_images(2000)
gauche = images[:, :, :4].reshape(-1, 32)   # 8 lignes x 4 colonnes = 32
droite = images[:, :, 4:].reshape(-1, 32)   # l'autre moitie


# ---------------------------------------------------------------------
# 2. Le pre-entrainement, sans la moindre etiquette
# ---------------------------------------------------------------------
encodeur = torch.nn.Sequential(torch.nn.Linear(32, 16), torch.nn.ReLU())
predicteur = torch.nn.Linear(16, 32)

params = list(encodeur.parameters()) + list(predicteur.parameters())
optim = torch.optim.Adam(params, lr=0.01)

for etape in range(1, 601):
    erreur = ((predicteur(encodeur(gauche)) - droite) ** 2).mean()
    optim.zero_grad()
    erreur.backward()
    optim.step()
    if etape in (1, 100, 300, 600):
        print(f"etape {etape:3d} : erreur = {erreur.item():.4f}")


# ---------------------------------------------------------------------
# 3. L'evaluation : encodeur gele, petite sonde, vingt etiquettes
# ---------------------------------------------------------------------
def sonde(entrees, cibles, entrees_test, cibles_test):
    """Une seule couche, qui apprend a repondre : couchee ou debout ?"""
    tete = torch.nn.Linear(entrees.shape[1], 1)
    opt = torch.optim.Adam(tete.parameters(), lr=0.05)
    for _ in range(300):
        p = tete(entrees).squeeze(1)
        perte = torch.nn.functional.binary_cross_entropy_with_logits(p, cibles)
        opt.zero_grad()
        perte.backward()
        opt.step()
    with torch.no_grad():
        prediction = (tete(entrees_test).squeeze(1) > 0).float()
        return (prediction == cibles_test).float().mean().item()


test_images, test_genre = fabrique_images(1000)
test_gauche = test_images[:, :, :4].reshape(-1, 32)

with torch.no_grad():                  # l'encodeur est gele : aucun gradient
    h = encodeur(gauche)
    h_test = encodeur(test_gauche)

# La part des images ou la moitie gauche est vide : personne ne peut repondre.
vide = (gauche.abs().sum(1) == 0).float().mean().item()
print(f"moitie gauche vide          : {vide * 100:.1f} %")

torch.manual_seed(1)
print("sonde sur la representation :",
      f"{sonde(h[:20], genre[:20], h_test, test_genre) * 100:.1f} %")
torch.manual_seed(1)
print("sonde sur les pixels bruts  :",
      f"{sonde(gauche[:20], genre[:20], test_gauche, test_genre) * 100:.1f} %")
