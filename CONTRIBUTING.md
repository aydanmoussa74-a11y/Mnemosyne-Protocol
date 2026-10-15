# Contributing to the Mnemosyne Protocol

Thank you for wanting to contribute. This is a production-grade agent memory
engine, and we keep the bar high: clean, typed, well-tested code with zero
tolerance for bloat or "AI slop."

---

## Guiding principles

1. **No AI slop / no bloat.** Every line of code should earn its place. If a
   feature can be expressed in fewer lines, do it. Avoid unnecessary
   abstractions, framework churn, or dependencies that solve a problem we
   don't have.
2. **Strict typing.** All public functions and methods must be fully typed.
   Use `from __future__ import annotations` and run the code through a type
   checker where practical.
3. **Tests are not optional.** Every PR must pass `pytest tests/`. New behavior
   requires new tests. Security-relevant changes require security tests.
4. **SQLite concurrency is sacred.** The consolidation worker and the FastAPI
   threadpool share one SQLite file. Any change to the store must preserve WAL
   mode, busy timeouts, and lock discipline. Never hold the in-memory RLock
   across a long-running DB commit.
5. **Tenant isolation is non-negotiable.** Every memory entity is namespaced by
   `tenant_id`. Cross-tenant leaks are treated as critical security bugs.
6. **Graceful degradation.** External model providers fail. Adapters must wrap
   provider errors in `AdapterError` and never leak stack traces or API keys
   to clients.

---

## Getting started

```bash
# Clone
git clone https://github.com/aydanmoussa74-a11y/Mnemosyne-Protocol.git
cd Mnemosyne-Protocol

# Install core + dev dependencies
pip install -e .[dev]

# Run the test suite
pytest tests/

# Run a single test file
pytest tests/test_security.py -q
```

---

## Submitting a pull request

1. Fork the repository and create a feature branch from `main`.
2. Make your changes. Keep diffs small and focused.
3. Add or update tests in `tests/` covering your change.
4. Run the full suite locally: `pytest tests/ -q`. All tests must pass.
5. Open a pull request against `main` with a clear description of the problem
   and the fix.

### PR checklist

- [ ] `pytest tests/` passes
- [ ] New code is fully typed
- [ ] Tests added for new behavior
- [ ] No new heavy dependencies (prefer stdlib / existing deps)
- [ ] SQLite concurrency / thread-safety rules respected
- [ ] Tenant isolation preserved
- [ ] CHANGELOG / docs updated if user-facing

---

## Code style

* Follow the conventions already in `pyproject.toml` (Black line length 100,
  Ruff select E/F/I/N/S).
* Docstrings on all public classes and methods.
* Prefer `edit_file`/targeted patches over full-file rewrites for existing
  files.

---

## Security

If you discover a security vulnerability, do **not** open a public issue.
See [SECURITY_AUDIT.md](./SECURITY_AUDIT.md) for the current threat model and
contact the maintainers privately.

---

## License

By contributing, you agree that your contributions will be licensed under the
MIT License ([LICENSE](./LICENSE)).
