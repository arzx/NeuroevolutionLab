from typing import List, Dict, Type, Callable
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import inspect
import random
import os
from copy import deepcopy

class LearningPlotter():
    """
    Ein Hilfsklasse zur Visualisierung des Trainingsverlaufs neuronaler Netzwerke über mehrere Seeds.

    Die Klasse erlaubt es, für jeden Seed ein neues, unabhängiges neuronales Netzwerk zu generieren und zu trainieren.
    Die Ergebnisse des Trainings können dann grafisch dargestellt werden.

    Attribute:
        num_seeds (int): Anzahl der Seeds, die für das Training verwendet werden.
        nn_class (Type): Die Klasse des neuronalen Netzwerks.
        nn_kwargs (Dict): Wörterbuch mit Schlüsselwortargumenten für die Initialisierung des neuronalen Netzwerks.
        layer_constructer (Callable): Funktion zur Erstellung der Netzwerkschichten.
        train_func (Callable): Trainingsfunktion für das neuronale Netzwerk.
        train_kwargs (Dict): Wörterbuch mit Schlüsselwortargumenten für die Trainingsfunktion.
        nn_required (bool): Gibt an, ob das neuronale Netzwerk als Argument für die Trainingsfunktion benötigt wird.
        colors (List[Type]): Liste von Farben für die Visualisierung.
        results (List[float] or None): Speichert die Ergebnisse nach dem Plotting.

    Methoden:
        __init__: Initialisiert eine Instanz von LearningPlotter.
        _get_vivid_colors: Erzeugt eine Liste lebendiger Farben.
        plot: Führt das Training für verschiedene Seeds durch und stellt die Ergebnisse grafisch dar.
    """

    def __init__(self, num_seeds: int, nn_class: Type, nn_kwargs: Dict, layer_constructer: Callable, train_func: Callable, train_kwargs: Dict, nn_required: bool=False) -> None:
        """
        Initialisiert die LearningPlotter-Instanz.

        Args:
            num_seeds (int): Anzahl der zu verwendenden Seeds für das Training des Netzwerks.
            nn_class (Type): Die Klasse des neuronalen Netzwerks. Diese Klasse sollte die Initialisierung des Netzwerks mit den über `nn_kwargs` und ggf. `layer_constructer` bereitgestellten Parametern unterstützen.
            nn_kwargs (Dict): Wörterbuch mit Schlüsselwortargumenten für die Initialisierung des neuronalen Netzwerks. 
                              Beinhaltet Argumente wie 'input_size', die für das Netzwerk benötigt werden, aber nicht die Layer selbst.
            layer_constructer (Callable, optional): Eine Methode, die die Schichten (Layer) des Netzwerks erstellt. 
                                                    Sollte `None` sein, wenn die Netzwerkklasse die Layer intern erstellt.
            train_func (Callable): Die Trainingsfunktion des Netzwerks. Kann eine Methode der Klasse `nn_class` oder eine externe Funktion sein.
            train_kwargs (Dict): Argumente für die Trainingsfunktion. Sollten alle notwendigen Parameter für das Training enthalten, außer das Netzwerk selbst, falls `nn_required=True` gesetzt ist.
            nn_required (bool, optional): Gibt an, ob das Netzwerk als Argument für die Trainingsfunktion benötigt wird. 
                                         Sollte `True` sein, wenn die Trainingsfunktion das Netzwerk als Parameter erwartet. Standardmäßig `False`.

        Beispiele für die Verwendung:
            - Wenn das Netzwerk 'layers' bei der Initialisierung erwartet und diese extern erzeugt werden, sollte `layer_constructer` eine Methode sein, die diese Layer erstellt.
            - `nn_kwargs` sollte die Parameter enthalten, die für die Initialisierung der Netzwerkklasse benötigt werden, abgesehen von den Layern.
            - `train_func` kann entweder eine Methode innerhalb der `nn_class` sein oder eine externe Trainingsfunktion.
            - Wenn `train_func` ein integraler Bestandteil der `nn_class` ist und das Netzwerkobjekt bereits kennt, sollte `nn_required` auf `False` gesetzt werden. Andernfalls, wenn `train_func` das Netzwerk als Argument benötigt, sollte `nn_required` auf `True` gesetzt werden.
        """
        self.num_seeds = num_seeds
        self.nn_class = nn_class
        self.nn_kwargs = nn_kwargs
        self.layer_constructer = layer_constructer
        self.train_func = train_func
        self.train_kwargs = train_kwargs
        self.nn_required = nn_required
        self.colors = self._get_vivid_colors(self.num_seeds)
        self.results = None

    def _get_vivid_colors(self, num_colors: int) -> List[Type]:
        """
        Erzeugt eine Liste lebendiger Farben.

        Args:
            num_colors (int): Anzahl der zu erzeugenden Farben.

        Returns:
            List[Type]: Liste der erzeugten Farben.
        """
        colors = [mcolors.hsv_to_rgb((i/num_colors, 1, 1)) for i in range(num_colors)]
        # to get the same colors consistently
        random.seed(314159265359)
        random.shuffle(colors)
        # disable seed
        random.seed(None)
        return colors
    
    def _fill_lists_with_mean(self, list_of_lists: List[List[float]]) -> np.ndarray:
        """
        Füllt jede Liste in einer Liste von Listen bis zur Länge der längsten Liste auf, indem der Mittelwert 
        aller Listen am jeweiligen Index verwendet wird. Wenn ein Index in einer Liste nicht existiert, wird er 
        ignoriert, um den Mittelwert zu berechnen.

        Args:
            list_of_lists (List[List[int]]): Eine Liste von Listen mit unterschiedlicher Länge.

        Returns:
            List[List[float]]: Die modifizierte Liste von Listen, bei der jede Liste auf die Länge der längsten Liste 
            aufgefüllt wurde. Fehlende Werte werden mit dem Mittelwert der vorhandenen Werte am entsprechenden Index ersetzt.
        """

        max_length = max(len(lst) for lst in list_of_lists)

        means = []
        for i in range(max_length):
            sum_values = 0
            count = 0
            for lst in list_of_lists:
                if i < len(lst):
                    sum_values += lst[i]
                    count += 1
            means.append(sum_values / count if count != 0 else 0)

        return np.array([lst + [means[i] for i in range(len(lst), max_length)] for lst in list_of_lists])



    def plot(self, y_name: str="Loss", save_plot: bool=False, plot_path: str="LearningPlotterPlot") -> List[float]:
        """
        Führt das Training für verschiedene Seeds durch und stellt die Ergebnisse grafisch dar.

        Args:
            y_name (str, optional): Bestimmt den Achsbeschriftung der y-Achse. Stadardmäßig "Loss".
            save_plot (bool, optional): Gibt an, ob der Plot gespeichert werden soll. Standardmäßig False.
            plot_path (str, optional): Pfad, unter dem der Plot gespeichert wird. Standardmäßig "LearningPlotterPlot".

        Returns:
            List[float]: Liste der Verlustwerte für jedes Seed.
        """
        all_losses = []

        for seed in range(self.num_seeds):
            # seed RNGs
            print(f"Starting training with seed {seed}...")
            np.random.seed(seed)
            random.seed(seed)
            os.environ["PYTHONHASHSEED"] = str(seed)

            if self.layer_constructer is not None:
                network = self.nn_class(layers=self.layer_constructer(), **self.nn_kwargs)
            else:
                network = self.nn_class(**self.nn_kwargs)
                
            if self.nn_required:
                losses = self.train_func(network=network, **self.train_kwargs)
            else:
                if "self" in inspect.signature(self.train_func).parameters: # check if train_func is inside the nn class
                    network.train_func = self.train_func
                    losses = network.train_func(self=network, **self.train_kwargs)
                else:
                    losses = self.train_func(**self.train_kwargs)
            all_losses.append(losses)

        # 'unseed' RNGs
        np.random.seed(None)
        random.seed(None)
        os.environ["PYTHONHASHSEED"] = "random"

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

        for i, losses in enumerate(all_losses):
            ax1.plot(losses, color=self.colors[i], label=f'Seed {i}')
        ax1.set_title('Trainingsverlauf für verschiedene Seeds')
        ax1.set_xlabel('Epochen')
        ax1.set_ylabel(y_name)
        ax1.legend()

        all_losses = self._fill_lists_with_mean(all_losses)
        
        mean_ls = np.mean(all_losses, axis=0)
        std_ls = np.std(all_losses, axis=0)
        ax2.plot(mean_ls, label='Mittelwert')
        ax2.fill_between(range(len(mean_ls)), mean_ls-std_ls, mean_ls+std_ls, alpha=0.2)
        ax2.set_title('Mittelwert und Standardabweichung über alle Seeds')
        ax2.set_xlabel('Epochen')
        ax2.set_ylabel(y_name)
        ax2.legend()

        plt.tight_layout()
        if save_plot:
            plt.save(plot_path)
        plt.show()

        self.results = all_losses