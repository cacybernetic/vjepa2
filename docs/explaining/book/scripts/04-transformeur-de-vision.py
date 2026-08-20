"""
Chapitre 4 — Le transformeur de vision, l'oeil de la machine.

Ce programme rassemble les trois codes du chapitre. Il verifie, avec de
vrais tenseurs, tout ce que nous avons calcule a la main :

  1. le decoupage en jetons, et les quatre nombres du tableau du chapitre ;
  2. l'attention, refaite avec les nombres de l'exemple ;
  3. un bloc de transformeur complet, puis une pile de blocs.

Il tourne sur n'importe quelle machine, sans carte graphique.

    python 04-transformeur-de-vision.py
"""

import torch
import torch.nn as nn


# ---------------------------------------------------------------------
# 1. Decouper une image et un clip en jetons
# ---------------------------------------------------------------------
def compte_jetons(images, hauteur, largeur, patch=16, tubelet=2):
    """Rend le nombre de jetons. Mettre images=None pour une image seule."""
    n = (hauteur // patch) * (largeur // patch)
    if images is not None:
        n = n * (images // tubelet)
    return n


def demo_decoupage():
    print("=== 1. Le decoupage en jetons ===")

    # La voie image : une convolution 2D de 16 x 16, qui avance de 16.
    tokeniseur_image = nn.Conv2d(3, 1024, kernel_size=16, stride=16)
    image = torch.zeros(1, 3, 256, 256)          # (lot, couleurs, H, W)
    jetons_image = tokeniseur_image(image).flatten(2).transpose(1, 2)
    print("image 256x256      ->", tuple(jetons_image.shape))

    # La voie video : une convolution 3D, avec 2 images d'epaisseur.
    tokeniseur_video = nn.Conv3d(
        3, 1024, kernel_size=(2, 16, 16), stride=(2, 16, 16))
    clip = torch.zeros(1, 3, 16, 256, 256)       # (lot, couleurs, T, H, W)
    jetons_video = tokeniseur_video(clip).flatten(2).transpose(1, 2)
    print("clip 16x256x256    ->", tuple(jetons_video.shape))

    # Les quatre nombres du tableau du chapitre.
    assert compte_jetons(None, 256, 256) == 256
    assert compte_jetons(None, 512, 512) == 1024
    assert compte_jetons(16, 256, 256) == 2048
    assert compte_jetons(64, 384, 384) == 18432
    print("les quatre nombres du tableau sont bons")

    # Le prix de l'attention, qui grimpe au carre.
    print("un clip de refroidissement coute",
          18432 ** 2 / 2048 ** 2, "fois plus cher\n")


# ---------------------------------------------------------------------
# 2. L'attention, refaite avec les nombres de l'exemple
# ---------------------------------------------------------------------
def demo_attention():
    print("=== 2. L'attention, avec nos nombres ===")

    q = torch.tensor([2.0, 0.0, 0.0, 0.0])
    k = torch.tensor([[2.0, 0.0, 0.0, 0.0],     # fourrure
                      [4.0, 0.0, 0.0, 0.0],     # oreille
                      [0.0, 3.0, 0.0, 0.0]])    # flaque
    v = torch.tensor([[10.0, 0.0, 0.0, 0.0],
                      [0.0, 10.0, 0.0, 0.0],
                      [0.0, 0.0, 10.0, 0.0]])

    scores = k @ q
    print("les scores bruts   :", scores.tolist())

    scores = scores / (q.shape[0] ** 0.5)        # racine de 4, donc 2
    print("remis a l'echelle  :", scores.tolist())

    poids = torch.softmax(scores, dim=0)
    print("les parts          :", [round(p, 3) for p in poids.tolist()])
    print("leur somme         :", round(poids.sum().item(), 3))

    resultat = poids @ v
    print("le nouveau jeton   :", [round(x, 3) for x in resultat.tolist()])
    print()


# ---------------------------------------------------------------------
# 3. Un bloc de transformeur complet
# ---------------------------------------------------------------------
class BlocTransformeur(nn.Module):
    def __init__(self, dimension=1024, nb_tetes=16):
        super().__init__()
        self.norme1 = nn.LayerNorm(dimension)
        self.attention = nn.MultiheadAttention(
            dimension, nb_tetes, batch_first=True)
        self.norme2 = nn.LayerNorm(dimension)
        self.reflexion = nn.Sequential(
            nn.Linear(dimension, 4 * dimension),
            nn.GELU(),                # la fonction d'activation
            nn.Linear(4 * dimension, dimension),
        )

    def forward(self, jetons):
        # Premier etage : on ecoute les autres, et on AJOUTE le resultat.
        propre = self.norme1(jetons)
        ecoute, _ = self.attention(propre, propre, propre)
        jetons = jetons + ecoute

        # Second etage : on reflechit tout seul, et on AJOUTE encore.
        jetons = jetons + self.reflexion(self.norme2(jetons))
        return jetons


def demo_bloc():
    print("=== 3. Un bloc de transformeur ===")

    bloc = BlocTransformeur()
    jetons = torch.zeros(1, 2048, 1024)          # un clip de la phase principale
    sortie = bloc(jetons)
    print("entree :", tuple(jetons.shape))
    print("sortie :", tuple(sortie.shape))
    print("la taille ne change pas : on peut donc empiler")

    nb_boutons = sum(p.numel() for p in bloc.parameters())
    print("boutons d'un seul bloc :", f"{nb_boutons:,}".replace(",", " "))
    print("pour 48 blocs empiles  :",
          f"{48 * nb_boutons:,}".replace(",", " "))


if __name__ == "__main__":
    torch.manual_seed(0)
    demo_decoupage()
    demo_attention()
    demo_bloc()
