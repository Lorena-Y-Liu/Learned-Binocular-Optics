# Contributing to Deep Stereo

Thank you for your interest in contributing to Deep Stereo! This document provides guidelines and instructions for contributing.

## Getting Started

1. **Fork the repository** on GitHub
2. **Clone your fork** locally:
   ```bash
   git clone https://github.com/Lorena-Y-Liu/deep_stereo.git
   cd deep_stereo
   ```
3. **Create a virtual environment** and install dependencies:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```

## Development Workflow

1. **Create a new branch** for your feature or bugfix:
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Make your changes** and ensure:
   - Code follows the existing style (PEP 8)
   - All functions have docstrings
   - Type hints are used where appropriate

3. **Test your changes** before submitting

4. **Commit your changes** with clear, descriptive messages:
   ```bash
   git add .
   git commit -m "Add: brief description of changes"
   ```

5. **Push to your fork** and create a Pull Request:
   ```bash
   git push origin feature/your-feature-name
   ```

## Code Style Guidelines

- Follow [PEP 8](https://pep8.org/) style guide
- Use meaningful variable and function names
- Add docstrings to all public functions and classes
- Use type hints for function parameters and return values
- Keep functions focused and modular

### Example Function Style

```python
def compute_depth(
    left_image: torch.Tensor,
    right_image: torch.Tensor,
    max_disparity: int = 192
) -> torch.Tensor:
    """
    Compute depth map from stereo image pair.
    
    Args:
        left_image: Left image tensor (B, C, H, W)
        right_image: Right image tensor (B, C, H, W)
        max_disparity: Maximum disparity to search
    
    Returns:
        Depth map tensor (B, 1, H, W)
    """
    # Implementation here
    pass
```

## Project Structure

```
deep_stereo/
├── configs/          # Configuration files
├── core/             # Stereo matching network
├── datasets/         # Dataset loaders
├── models/           # Neural network models
├── optics/           # DOE camera simulation
├── util/             # Utility functions
└── examples/         # Usage examples
```

## Reporting Issues

When reporting issues, please include:

1. **Description**: Clear description of the problem
2. **Steps to reproduce**: Minimal code to reproduce the issue
3. **Expected behavior**: What you expected to happen
4. **Actual behavior**: What actually happened
5. **Environment**: Python version, PyTorch version, OS, GPU

## Pull Request Guidelines

- Keep PRs focused on a single feature or fix
- Update documentation if needed
- Add tests for new functionality
- Ensure all existing tests pass
- Reference related issues in PR description

## Questions?

Feel free to open an issue for questions or discussions about the project.

## License

By contributing, you agree that your contributions will be licensed under the MIT License.
