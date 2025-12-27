# Task Completion Checklist

## Before Committing Changes

### Code Quality
- [ ] Code follows project conventions (see `code_style.md`)
- [ ] Type hints present on function signatures
- [ ] Docstrings for classes and public functions
- [ ] No debug print statements left behind

### Testing
- [ ] Run the modified script to verify it works:
  ```bash
  python <script>.py
  ```
- [ ] For model changes, verify forward pass works:
  ```python
  python -c "from model import OneLayerTransformer; m = OneLayerTransformer(); print('OK')"
  ```

### No Formal Linting/Formatting Tools
This project does not have configured linting or formatting tools. Manually ensure:
- Consistent indentation (4 spaces)
- Line length reasonable (~100 chars)
- Import grouping as per style guide

### Analysis Scripts
If modifying analysis scripts:
- [ ] Generated plots are saved correctly
- [ ] Output directories are created if needed (`Path(...).mkdir(exist_ok=True)`)

## After Task Completion

### Documentation
- [ ] Update README.md if adding new experiments or significant features
- [ ] Add/update docstrings for new functions

### Checkpoints
- [ ] If training was modified, document any new hyperparameters
- [ ] Trained model checkpoints go in `checkpoints_*/` directories
- [ ] Analysis output goes in `analysis_*/` or `results*/` directories

### Git
```bash
git status
git diff
git add <files>
git commit -m "descriptive message"
```
