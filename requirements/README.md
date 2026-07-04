# Requirements

The core CLI intentionally uses Python standard library only.

Optional installs:

```bash
pip install -e '.[analysis,monitoring,clients,dev]'
```

Serving engines should usually run through Docker images, not direct local pip installs, until BTC releases final constraints.
