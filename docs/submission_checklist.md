# Competition Submission Checklist

Use this only after BTC releases official requirements.

## Before Packaging

- [ ] Official problem statement saved.
- [ ] Official model identifier/path known.
- [ ] Docker Compose starts from clean machine.
- [ ] No private API keys committed.
- [ ] No contest data leaked.
- [ ] README contains exact run command.
- [ ] Healthcheck endpoint works.
- [ ] Benchmark command is reproducible.
- [ ] Logs do not include secrets.

## Validation

- [ ] Cold start test.
- [ ] Small workload smoke test.
- [ ] Burst workload test.
- [ ] Long context test.
- [ ] Timeout test.
- [ ] OOM resistance test.
- [ ] Restart/recovery test.

## Final Artifacts

- [ ] `docker-compose.yml`
- [ ] `.env.example`
- [ ] source code or image reference, depending on rules.
- [ ] final benchmark report.
- [ ] decision log.

## Current Skeleton

Skeleton path:

```text
submissions/phase1/
```

Validate structure:

```bash
scripts/validate_submission.sh submissions/phase1
```
