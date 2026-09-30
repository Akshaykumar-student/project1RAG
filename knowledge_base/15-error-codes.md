# Common FlowDesk error codes

- **FD-401 — Authentication failed:** Sign in again. For API requests, check that the key is active and sent in the Authorization header.
- **FD-403 — Permission denied:** Your role cannot perform the action. Ask an Admin or owner for the required permission.
- **FD-409 — Update conflict:** The record changed in another session. Refresh the page and repeat the edit.
- **FD-422 — Invalid import:** A CSV file has missing required columns or invalid values. Download the row-level error report.
- **FD-429 — Too many requests:** The API rate limit was reached. Wait for the `Retry-After` duration before retrying.
- **FD-503 — Service temporarily unavailable:** Check status.flowdesk.example and retry after service is restored.

