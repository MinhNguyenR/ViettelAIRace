# Phase 1 Submission Skeleton

This directory is a placeholder for Viettel AI Race Phase 1 Docker Compose submission.

Current status:

- No official model is configured.
- No contest data is included.
- Runtime is not locked.
- This compose file is for structure validation only.

Validate:

```bash
scripts/validate_submission.sh submissions/phase1
```

Run only after `MODEL_ID` points to an available official model:

```bash
docker compose --env-file .env -f docker-compose.yml up
```
