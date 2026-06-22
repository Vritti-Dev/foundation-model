# Vritti AI Labs - Web

FastAPI app with email/password auth, server-rendered landing/dashboard pages, and Postgres for user + progress storage. Sits in front of the existing Jupyter Book course.

```
web/
  app/
    main.py            FastAPI app, mounts static and the (optional) course
    config.py          pydantic-settings; reads .env
    db.py              SQLAlchemy 2.0 engine + session
    models.py          User, Progress
    schemas.py         Pydantic request/response types
    security.py        bcrypt password hashing + JWT helpers
    deps.py            FastAPI auth dependencies
    routes/
      auth.py          /auth/signup, /auth/login, /auth/logout
      pages.py         /, /login, /signup, /dashboard
      progress.py      /api/progress  (GET, POST)
    templates/         base.html, landing.html, login.html, signup.html, dashboard.html
    static/manifesto.css
  migrations/          Alembic env + versions/0001_init.py
  docker-compose.yml   Postgres 16 for local dev
  alembic.ini
  requirements.txt
  .env.example
```

## Run it locally

```bash
# 1. start postgres
docker compose -f web/docker-compose.yml up -d

# 2. install python deps (use a venv; py3.11+ recommended)
pip install -r web/requirements.txt

# 3. point at the local DB and a real secret
cp web/.env.example web/.env
# edit web/.env if you want (defaults match docker-compose)

# 4. run the migration
alembic -c web/alembic.ini upgrade head

# 5. start the app
uvicorn web.app.main:app --reload --host 127.0.0.1 --port 8000
```

Then visit:

- <http://127.0.0.1:8000/> -> the sovereign-AI landing page
- <http://127.0.0.1:8000/signup> -> create an account
- <http://127.0.0.1:8000/login> -> sign in
- <http://127.0.0.1:8000/dashboard> -> your progress (requires login)
- <http://127.0.0.1:8000/course/> -> the existing Jupyter Book (if `book/_build/html` exists)

## How auth works

- Passwords hashed with bcrypt (12 rounds).
- On signup/login a JWT is issued (HS256, 7-day expiry) and stored in an httpOnly cookie named `vritti_session`.
- `require_user` dependency decodes the cookie, looks up the user, returns 401 otherwise.
- Logout clears the cookie.

## How progress works

The browser-side gating script in `book/_static/gating.js` already tracks completion in IndexedDB. When the user is signed in, you can POST to `/api/progress` with `{lesson_id, status}` to mirror the same flag server-side, which is what the dashboard reads.

## Production checklist

- Set `JWT_SECRET` to a real value (`openssl rand -hex 32`).
- Set `COOKIE_SECURE=true` when serving over HTTPS.
- Point `DATABASE_URL` at a managed Postgres (Neon, RDS, Cloud SQL, your own).
- Run behind a reverse proxy (nginx, Caddy, or a platform's edge).
- Run migrations on deploy: `alembic -c web/alembic.ini upgrade head`.
