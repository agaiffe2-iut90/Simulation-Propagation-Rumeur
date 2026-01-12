# -*- coding: utf-8 -*-
# Simulation de la propagation d'une rumeur sur un réseau social (type X / Twitter)

import networkx as nx
import random
import matplotlib.pyplot as plt


class SimulationRumeur:
    def __init__(self, n_noeuds, m_links, p_transmission, p_recovery, modele='barabasi'):
        """
        Initialise la simulation.
        """
        self.n = n_noeuds
        self.p_trans = p_transmission
        self.p_rec = p_recovery

        # Création du graphe selon le modèle choisi
        if modele == 'barabasi':
            self.graphe = nx.barabasi_albert_graph(n_noeuds, m_links)
        elif modele == 'erdos':
            self.graphe = nx.erdos_renyi_graph(n_noeuds, 0.05)
        elif modele == 'watts':
            self.graphe = nx.watts_strogatz_graph(n_noeuds, k=4, p=0.1)

        # États : 0 = Sain, 1 = Infecté, 2 = Rétabli
        self.etats = {i: 0 for i in range(n_noeuds)}

        # Patient zéro : utilisateur le plus influent
        patient_zero = max(self.graphe.degree, key=lambda x: x[1])[0]
        self.etats[patient_zero] = 1

        # Historique pour les graphiques
        self.historique = {'S': [], 'I': [], 'R': []}
        self._maj_historique()

    def _maj_historique(self):
        """Met à jour les statistiques S / I / R"""
        valeurs = list(self.etats.values())
        self.historique['S'].append(valeurs.count(0))
        self.historique['I'].append(valeurs.count(1))
        self.historique['R'].append(valeurs.count(2))

    def step(self):
        """Avance la simulation d'un pas de temps"""
        nouveaux_etats = self.etats.copy()

        for noeud in self.graphe.nodes():
            etat = self.etats[noeud]

            # Utilisateur sain
            if etat == 0:
                voisins_infectes = sum(
                    1 for v in self.graphe.neighbors(noeud) if self.etats[v] == 1
                )
                if voisins_infectes > 0:
                    proba = 1 - (1 - self.p_trans) ** voisins_infectes
                    if random.random() < proba:
                        nouveaux_etats[noeud] = 1

            # Utilisateur infecté
            elif etat == 1:
                if random.random() < self.p_rec:
                    nouveaux_etats[noeud] = 2

        self.etats = nouveaux_etats
        self._maj_historique()

    def run(self, steps=50):
        """Lance la simulation"""
        for _ in range(steps):
            self.step()
            if self.historique['I'][-1] == 0:
                break


# --- PARAMÈTRES ---
N_NOEUDS = 200
M_LINKS = 2
P_TRANS = 0.2
P_REC = 0.05
STEPS = 60

# --- LANCEMENT ---
sim = SimulationRumeur(N_NOEUDS, M_LINKS, P_TRANS, P_REC, modele='barabasi')
sim.run(STEPS)

# --- VISUALISATION ---
plt.figure(figsize=(15, 6))

# Courbes S / I / R
plt.subplot(1, 2, 1)
x = range(len(sim.historique['S']))
plt.plot(x, sim.historique['S'], label='Sains', color='blue')
plt.plot(x, sim.historique['I'], label='Infectés', color='red')
plt.plot(x, sim.historique['R'], label='Rétablis', color='green')
plt.xlabel("Temps")
plt.ylabel("Nombre d'utilisateurs")
plt.title("Propagation de la rumeur")
plt.legend()

# Graphe final
plt.subplot(1, 2, 2)
couleurs = ['blue' if s == 0 else 'red' if s == 1 else 'green' for s in sim.etats.values()]
tailles = [v * 10 for v in dict(sim.graphe.degree()).values()]
pos = nx.spring_layout(sim.graphe, seed=42)

nx.draw_networkx_nodes(sim.graphe, pos, node_color=couleurs, node_size=tailles, alpha=0.8)
nx.draw_networkx_edges(sim.graphe, pos, alpha=0.2)
plt.title("État final du réseau")
plt.axis('off')

plt.tight_layout()
plt.show()
