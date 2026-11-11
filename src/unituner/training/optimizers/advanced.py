"""Isolate adv_optm stochastic-rounding RNG per optimizer for exact resume."""

from collections.abc import Callable

import torch
from adv_optm.optim.AdamW_adv import AdamW_adv
from adv_optm.util import param_update


class ResumableAdamW(AdamW_adv):
    """Keep upstream's module-global random generators out of checkpoint semantics."""

    def __init__(self, *args: object, **kwargs: object) -> None:
        super().__init__(*args, **kwargs)
        self._unituner_generators: dict[torch.device, torch.Generator] = {}
        for group in self.param_groups:
            for parameter in group["params"]:
                device = parameter.device
                if device not in self._unituner_generators:
                    generator = torch.Generator(device=device)
                    generator.set_state(torch.get_rng_state() if device.type == "cpu" else torch.cuda.get_rng_state(device))
                    self._unituner_generators[device] = generator

    def step(self, closure: Callable[[], float] | None = None) -> object:
        previous = {device: param_update._generators.get(device) for device in self._unituner_generators}
        try:
            param_update._generators.update(self._unituner_generators)
            return super().step(closure)
        finally:
            for device, generator in previous.items():
                if generator is None:
                    param_update._generators.pop(device, None)
                else:
                    param_update._generators[device] = generator

    def state_dict(self) -> dict:
        state = super().state_dict()
        state["unituner_rng"] = {str(device): generator.get_state() for device, generator in self._unituner_generators.items()}
        return state

    def load_state_dict(self, state_dict: dict) -> None:
        state = dict(state_dict)
        random_states = state.pop("unituner_rng")
        if set(random_states) != {str(device) for device in self._unituner_generators}:
            raise ValueError("Advanced optimizer RNG devices differ from checkpoint")
        super().load_state_dict(state)
        # adv_optm 2.6.3 restores signed INT8 first moments as UINT8.
        for saved_group, group in zip(state["param_groups"], self.param_groups, strict=True):
            for saved_id, parameter in zip(saved_group["params"], group["params"], strict=True):
                saved_state = state["state"].get(saved_id, {})
                loaded_state = self.state[parameter]
                for name, value in saved_state.items():
                    if isinstance(value, torch.Tensor) and value.dtype == torch.int8:
                        loaded_state[name] = loaded_state[name].to(torch.int8)
        for device, generator in self._unituner_generators.items():
            generator.set_state(random_states[str(device)].cpu())
