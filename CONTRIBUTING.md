# Contributing to Cross-Chain Anomaly Detection

We welcome contributions to the Cross-Chain Anomaly Detection project! This document provides guidelines for contributing to the project.

## 🤝 How to Contribute

### Reporting Issues

1. **Check existing issues** first to avoid duplicates
2. **Use the issue templates** when available
3. **Provide detailed information** including:
   - Operating system and Python version
   - Steps to reproduce the issue
   - Expected vs actual behavior
   - Error messages and stack traces
   - Sample data (if applicable and non-sensitive)

### Submitting Changes

1. **Fork the repository** and create a new branch
2. **Follow the coding standards** outlined below
3. **Add tests** for new functionality
4. **Update documentation** as needed
5. **Submit a pull request** with a clear description

## 🔧 Development Setup

### Environment Setup
```bash
# Clone your fork
git clone https://github.com/YOUR_USERNAME/CrossChainAnomalyDetection.git
cd CrossChainAnomalyDetection

# Create development environment
conda env create -f environment.yml
conda activate cc-AD-1

# Install development dependencies
pip install pytest black flake8 pre-commit
```

### Pre-commit Hooks
```bash
# Install pre-commit hooks
pre-commit install

# Run hooks manually
pre-commit run --all-files
```

## 📝 Coding Standards

### Python Style Guide
- Follow **PEP 8** style guidelines
- Use **Black** for code formatting
- Maximum line length: **88 characters**
- Use **type hints** for function parameters and returns
- Write **descriptive docstrings** for all public methods

### Code Organization
```python
# Good: Clear, descriptive naming
def calculate_anomaly_score(transaction_features: pd.DataFrame) -> np.ndarray:
    """
    Calculate anomaly scores for transaction features.
    
    Args:
        transaction_features: DataFrame with preprocessed features
        
    Returns:
        Array of anomaly scores between 0 and 1
    """
    pass

# Bad: Unclear naming and no documentation
def calc_score(df):
    pass
```

### Documentation
- **Class docstrings**: Describe purpose and key methods
- **Method docstrings**: Include Args, Returns, and Raises sections
- **Inline comments**: Explain complex logic or business rules
- **README updates**: Update documentation for new features

## 🧪 Testing Guidelines

### Test Structure
```python
import pytest
import pandas as pd
from src.bridge_compliance_ml import BridgeComplianceML

class TestBridgeComplianceML:
    def test_data_loading(self):
        """Test that data loading works correctly."""
        ml = BridgeComplianceML('test_data.csv')
        df = ml.load_and_explore_data()
        assert isinstance(df, pd.DataFrame)
        assert len(df) > 0

    def test_preprocessing(self):
        """Test data preprocessing pipeline."""
        # Test implementation
        pass
```

### Running Tests
```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src

# Run specific test file
pytest tests/test_bridge_compliance.py
```

## 📊 Data Guidelines

### Data Privacy
- **Never commit** real transaction data
- **Use synthetic data** for examples and tests
- **Anonymize** any data used in documentation
- **Follow institutional** data handling policies

### Sample Data Format
```python
# Example of acceptable sample data structure
sample_data = pd.DataFrame({
    'transaction_id': ['tx_001', 'tx_002'],
    'amount': [100.0, 250.0],
    'timestamp': ['2025-01-01 12:00:00', '2025-01-01 13:00:00'],
    'label': [0, 1]  # 0: normal, 1: suspicious
})
```

## 🚀 Feature Development

### New Features
1. **Create an issue** describing the feature
2. **Discuss the approach** with maintainers
3. **Implement in small commits** with clear messages
4. **Add comprehensive tests**
5. **Update documentation**

### Machine Learning Models
- **Document model architecture** and hyperparameters
- **Include performance metrics** on test data
- **Provide feature importance** analysis
- **Compare against baseline** models

### Visualization Features
- **Use consistent styling** across plots
- **Include meaningful titles** and labels
- **Save plots** in appropriate formats
- **Document plot interpretations**

## 🔍 Code Review Process

### Pull Request Guidelines
- **Clear title** summarizing the change
- **Detailed description** explaining the motivation
- **Link related issues** using keywords (fixes #123)
- **Include screenshots** for UI/visualization changes
- **Small, focused changes** are preferred

### Review Checklist
- [ ] Code follows style guidelines
- [ ] Tests pass and coverage is maintained
- [ ] Documentation is updated
- [ ] No sensitive data is exposed
- [ ] Performance impact is considered
- [ ] Backward compatibility is maintained

## 🐛 Bug Fix Guidelines

### Bug Reports
- **Reproducible steps** to trigger the bug
- **Expected vs actual behavior**
- **Environment details** (OS, Python version, dependencies)
- **Sample data** or code to reproduce (if safe to share)

### Bug Fixes
- **Write a test** that reproduces the bug
- **Fix the minimal code** necessary
- **Verify the test passes** after the fix
- **Consider edge cases** and similar issues

## 📈 Performance Considerations

### Optimization Guidelines
- **Profile before optimizing** using tools like cProfile
- **Focus on bottlenecks** identified through profiling
- **Maintain readability** while improving performance
- **Benchmark improvements** with realistic data sizes

### Memory Management
- **Use generators** for large datasets
- **Release references** to large objects when done
- **Monitor memory usage** during development
- **Consider chunking** for very large files

## 🔒 Security Guidelines

### Data Security
- **Validate inputs** to prevent injection attacks
- **Sanitize file paths** to prevent directory traversal
- **Use secure defaults** for configuration
- **Log security events** appropriately

### Dependency Management
- **Keep dependencies updated** to patch vulnerabilities
- **Review new dependencies** for security issues
- **Pin dependency versions** for reproducibility
- **Use virtual environments** to isolate dependencies

## 📞 Getting Help

### Communication Channels
- **GitHub Issues**: Technical questions and bug reports
- **Discussions**: General questions and feature ideas
- **Email**: Sensitive security issues or private concerns

### Response Times
- **Issues**: Acknowledged within 2-3 business days
- **Pull Requests**: Initial review within 1 week
- **Security Issues**: Addressed within 24 hours

## 📄 License

By contributing to this project, you agree that your contributions will be licensed under the MIT License.

## 🙏 Recognition

Contributors will be acknowledged in:
- Repository README.md
- Release notes for significant contributions
- Academic publications (if applicable)

Thank you for contributing to Cross-Chain Anomaly Detection! 🚀
