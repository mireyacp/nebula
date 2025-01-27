from abc import abstractmethod
import logging
import types
from nebula.addons.attacks.attacks import Attack


class CommunicationAttack(Attack):
    def __init__(self, engine, target_class, target_method, round_start_attack, round_stop_attack, decorator_args=None):
        super().__init__()
        self.engine = engine
        self.target_class = target_class
        self.target_method = target_method
        self.decorator_args = decorator_args
        self.round_start_attack = round_start_attack
        self.round_stop_attack = round_stop_attack
        self.original_method = getattr(target_class, target_method, None)

        if not self.original_method:
            raise AttributeError(f"Method {target_method} not found in class {target_class}")

    @abstractmethod
    def decorator(self, *args):
        """Decorator that adds malicious behavior to the execution of the original method."""
        pass

    async def _inject_malicious_behaviour(self):
        """Inject malicious behavior into the target method."""
        logging.info("Injecting malicious behavior")
        
        decorated_method = self.decorator(self.decorator_args)(self.original_method)
        
        setattr(
            self.target_class,
            self.target_method,
            types.MethodType(decorated_method, self.target_class),
        )

    async def _restore_original_behaviour(self):
        """Restore the original behavior of the target method."""
        logging.info(f"Restoring original behavior of {self.target_class}.{self.target_method}")
        setattr(self.target_class, self.target_method, self.original_method)

    async def attack(self):
        """Perform the attack logic based on the current round."""
        if self.engine.round == self.round_stop_attack:
            logging.info(f"[{self.__class__.__name__}] Restoring original behavior")
            await self._restore_original_behaviour()
        elif self.engine.round == self.round_start_attack:
            logging.info(f"[{self.__class__.__name__}] Injecting malicious behavior")
            await self._inject_malicious_behaviour()
        