from .runtime import RuntimeParams
from .speculation import SpecParams

def default_runtime_params() -> RuntimeParams:
    return RuntimeParams()

def default_spec_params() -> SpecParams:
    return SpecParams()
