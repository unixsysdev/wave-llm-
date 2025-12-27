# Code Style and Conventions

## General Style
- Python 3 with type hints for function signatures
- Module-level docstrings describing purpose and references
- Class and function docstrings with Args/Returns documentation

## Naming Conventions
- Classes: PascalCase (`OneLayerTransformer`, `SinusoidalModulusEncoding`)
- Functions/methods: snake_case (`compute_loss_and_accuracy`, `create_modular_addition_data`)
- Variables: snake_case (`d_model`, `n_heads`, `train_frac`)
- Constants: snake_case (not UPPER_CASE in this project)
- Private/internal: underscore prefix not commonly used

## Type Hints
- Used in function signatures: `def forward(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:`
- Docstrings complement type hints with shape information

## Documentation Style
```python
def function_name(
    param1: Type1,
    param2: Type2 = default,
) -> ReturnType:
    """
    Brief description.
    
    Args:
        param1: Description of param1
        param2: Description with default info
        
    Returns:
        Description of return value
    """
```

## Tensor Operations
- Use `einops` for readable tensor manipulations:
  ```python
  from einops import rearrange, einsum
  q = einsum(x, self.W_Q, 'b s d, h d k -> b h s k')
  ```

## Model Architecture Patterns
- Store intermediate activations in `self.cache` dict for analysis
- Use `nn.Parameter` for custom weight matrices
- Use `nn.Linear` for standard layers

## File Organization
- One main model class per `model*.py` file
- Separate training (`train*.py`) from analysis (`analyze*.py`)
- Shell scripts (`run*.sh`) for common experiment pipelines

## Import Order
1. Standard library (`math`, `json`, `pathlib`)
2. Third-party (`torch`, `numpy`, `matplotlib`, `tqdm`)
3. Local modules (`from model import ...`)
