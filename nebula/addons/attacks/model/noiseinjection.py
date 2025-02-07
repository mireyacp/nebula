import logging

import torch

from nebula.addons.attacks.model.modelattack import ModelAttack


class NoiseInjectionAttack(ModelAttack):
    """
    Implements a noise injection attack on the received model weights.

    This attack introduces noise into the model weights by adding random values
    scaled by a specified strength, potentially disrupting the model’s behavior.

    Args:
        engine (object): The training engine object that manages the aggregator.
        attack_params (dict): Parameters for the attack, including:
            - strength (int): The strength of the noise to be injected into the weights.
    """

    def __init__(self, engine, attack_params):
        """
        Initializes the NoiseInjectionAttack with the specified engine and parameters.

        Args:
            engine (object): The training engine object.
            attack_params (dict): Dictionary of attack parameters, including strength.
        """
        super().__init__(engine)
        self.strength = int(attack_params["strength"])
        self.round_start_attack = int(attack_params["round_start_attack"])
        self.round_stop_attack = int(attack_params["round_stop_attack"])

    def model_attack(self, received_weights):
        """
        Performs the noise injection attack by adding random noise to the model weights.

        The noise is generated from a normal distribution and scaled by the
        specified strength, modifying each layer's weights in the model.

        Args:
            received_weights (dict): The aggregated model weights to be modified.

        Returns:
            dict: The modified model weights after applying the noise injection attack.
        """
        logging.info(f"[NoiseInjectionAttack] Performing noise injection attack with a strength of {self.strength}")
        lkeys = list(received_weights.keys())
        for k in lkeys:
            logging.info(f"Layer noised: {k}")
            received_weights[k].data += torch.randn(received_weights[k].shape) * self.strength
        return received_weights
