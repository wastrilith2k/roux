# Admin, Operations & Testing

[Back to Architecture Index](../ARCHITECTURE.md)

---

## Admin CLI (`scripts/admin.py`)

```bash
python scripts/admin.py status              # System health check
python scripts/admin.py errors              # Recent errors (--companion, --module, --limit)
python scripts/admin.py costs               # Token usage (--companion, --days)
python scripts/admin.py companions          # List all companions
python scripts/admin.py companion kai       # Companion details
python scripts/admin.py memory kai          # Memory search (--search "term")
python scripts/admin.py goals kai           # Active goals
python scripts/admin.py db                  # Database stats
```

## Ops Bot (`ops/telegram_ops_bot.py`)

Admin Telegram bot for system notifications and commands. Separate from the companion's Telegram bridge.

## Report Generator (`scripts/generate_report.py`)

Self-contained HTML report for simulation results:
```bash
python scripts/generate_report.py --output report.html
python scripts/generate_report.py --output report.html --json data.json
python scripts/generate_report.py --companions kai mira
```

Renders: messages, opinions, curiosities, episodes, reflections, fact counts in dark-themed HTML.

## Database Seeding (`scripts/seed_database.py`)

Bootstraps companion database with initial knowledge and conversation history from `instances/<id>/seed_history.json`:
```bash
python scripts/seed_database.py --companion kai
python scripts/seed_database.py --all
```

## Other Scripts

| Script | Purpose |
|--------|---------|
| `backup_neo4j.py` / `restore_neo4j.py` | Knowledge graph backup/restore |
| `google_oauth_setup.py` | Google Calendar/Gmail OAuth flow |
| `check_environment.py` | Environment validation |
| `safe_docker_compose.py` | Safe Docker Compose wrapper |

---

## Testing

### Test Suite

13 test files + `conftest.py` with ~50+ test methods:

| Test File | Coverage |
|-----------|----------|
| `test_persona_config.py` | PersonaConfig loading, singleton, env var overrides, switching |
| `test_relationship_types.py` | RelationshipType enum, inverse/symmetric relationships |
| `test_error_tracker.py` | Error recording resilience, buffering, buffer limits |
| `test_clock.py` | SystemClock vs SimulationClock, time advancement |
| `test_reach_out_pressure.py` | Accumulation, diminishing returns, bump/release, capping |
| `test_confidence_decay.py` | Effective confidence, decay curves, mention boosts |
| `test_firebase_auth.py` | Firebase authentication |
| `test_admin_cli.py` | Admin CLI commands |
| `test_companion_isolation.py` | Companion data isolation |
| `test_fact_store_utils.py` | Fact storage utilities |
| `test_message_bundling.py` | Message bundling logic |
| `test_report_xss.py` | XSS prevention in reports |
| `test_connection.py` | CLI connection |

### Running Tests

```bash
pytest                           # All tests
pytest tests/ -v                 # Verbose
pytest tests/test_pipeline.py    # Specific file
```

Note: 3 tests require PostgreSQL. All others are pure unit tests with mocked externals.

### Test Patterns

- Fixture-based setup/teardown via `conftest.py`
- Mock external services (DB, Redis)
- Time-based testing with `SimulationClock` pinning
- Configuration isolation between tests
